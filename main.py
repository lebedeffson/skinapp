import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# ==========================================
# 0. СЛОВАРИ КЛАССОВ (НАЗВАНИЯ ВМЕСТО ИНДЕКСОВ)
# ==========================================
CLASSES_SKIN_KERAS = {
    0: "Базалиома",
    1: "Кератоз",
    2: "Меланома",
    3: "Родинка"
}

# Частоты соответствующих классов в HAM10000. Модель была сильно смещена к
# доминирующему классу невусов, поэтому перед показом результата убираем prior.
SKIN_KERAS_CLASS_COUNTS = np.array([514, 1099, 1113, 6705], dtype=np.float32)

CLASSES_SKIN_PYTORCH = {
    0: "Экзема",
    1: "Грибок ногтей",
    2: "Бородавки",
    3: "Угревая болезнь"
}

CLASSES_NAILS = {
    0: "Алопеция ареата",
    1: "Голубоватый оттенок ногтя",
    2: "Отслоение ногтевой пластины",
    3: "Спонтанное кровоизлияние под ногтем"
}

CLASSES_EMOTIONS = {
    0: "😡 Злость",
    1: "🤢 Отвращение",
    2: "😨 Страх",
    3: "😁 Счастье",
    4: "😐 Нейтральность",
    5: "😢 Грусть",
    6: "😲 Удивление"
}

# ==========================================
# 1. ЛЕНИВАЯ ЗАГРУЗКА ML-ФРЕЙМВОРКОВ
# ==========================================
@st.cache_resource(show_spinner=False)
def get_torch_modules():
    """PyTorch грузится только после выбора режима анализа."""
    import torch
    import torchvision.transforms as transforms
    import torchvision.models as models

    return torch, transforms, models


@st.cache_resource(show_spinner=False)
def get_keras_modules():
    """Keras/TensorFlow всегда импортируются после PyTorch.

    Обратный порядок импортов падает в текущем окружении при загрузке Triton.
    """
    get_torch_modules()
    import keras
    import tensorflow as tf

    if not getattr(keras.layers.Dense, "_mira_safe_init", False):
        original_dense_init = keras.layers.Dense.__init__

        def safe_dense_init(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            original_dense_init(self, *args, **kwargs)

        keras.layers.Dense.__init__ = safe_dense_init
        keras.layers.Dense._mira_safe_init = True

    return keras, tf


# ==========================================
# 2. ФУНКЦИИ ГЕНЕРАЦИИ ТЕПЛОВЫХ КАРТ (Grad-CAM)
# ==========================================
def overlay_heatmap(pil_img, heatmap, alpha=0.4):
    """Накладывает тепловую карту внимания на оригинальное изображение"""
    from matplotlib import colormaps

    img_np = np.array(pil_img.convert('RGB'))
    h, w, _ = img_np.shape

    # Изменяем размер карты под оригинал
    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize((w, h), Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img) / 255.0

    # Накладываем цветовое палитра JET (красный = максимум внимания)
    cmap = colormaps.get_cmap('jet')
    color_heatmap = cmap(heatmap_resized)[:, :, :3]
    color_heatmap = np.uint8(color_heatmap * 255)

    # Смешиваем изображения
    overlay = np.uint8(img_np * (1 - alpha) + color_heatmap * alpha)
    return Image.fromarray(overlay)


def generate_pytorch_gradcam(model, input_tensor, target_class=None):
    """Grad-CAM для PyTorch (ResNet18)"""
    torch, _, _ = get_torch_modules()
    try:
        model.eval()
        target_layer = model.layer4[-1]  # Последний сверточный слой ResNet18

        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)

        model.zero_grad()
        output = model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward()

        h1.remove()
        h2.remove()

        grads = gradients[0].cpu().data.numpy()[0]
        acts = activations[0].cpu().data.numpy()[0]

        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam
    except Exception:
        return None


def generate_keras_gradcam(model, input_array, target_class=None):
    """Grad-CAM для Keras моделей"""
    keras, tf = get_keras_modules()
    try:
        conv_layers = [
            layer for layer in model.layers
            if isinstance(layer, keras.layers.Conv2D)
        ]
        last_conv_layer = conv_layers[-1] if conv_layers else None

        if last_conv_layer is None:
            return None

        if isinstance(model, keras.Sequential):
            with tf.GradientTape() as tape:
                outputs = tf.convert_to_tensor(input_array)
                conv_outputs = None
                for layer in model.layers:
                    try:
                        outputs = layer(outputs, training=False)
                    except TypeError:
                        outputs = layer(outputs)
                    if layer is last_conv_layer:
                        conv_outputs = outputs
                predictions = outputs
                if target_class is None:
                    target_class = tf.argmax(predictions[0])
                loss = predictions[:, target_class]
        else:
            grad_model = keras.Model(
                inputs=model.inputs[0],
                outputs=[last_conv_layer.output, model.outputs[0]],
            )
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(input_array, training=False)
                if target_class is None:
                    target_class = tf.argmax(predictions[0])
                loss = predictions[:, target_class]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except Exception:
        return None


# ==========================================
# 3. ФУНКЦИИ ЗАГРУЗКИ И ПРЕДОБРАБОТКИ
# ==========================================
@st.cache_resource(show_spinner=False)
def load_keras_model(model_path):
    keras, _ = get_keras_modules()
    if not os.path.exists(model_path):
        return None
    try:
        return keras.models.load_model(model_path, compile=False)
    except Exception as e:
        return f"ERROR: {e}"


@st.cache_resource(show_spinner=False)
def load_pytorch_model(model_path, num_classes):
    torch, _, models = get_torch_modules()
    if not os.path.exists(model_path):
        return None
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        if isinstance(checkpoint, dict):
            model = models.resnet18()
            if "fc.1.weight" in checkpoint and "fc.4.weight" in checkpoint:
                w1, w4 = checkpoint["fc.1.weight"], checkpoint["fc.4.weight"]
                model.fc = torch.nn.Sequential(
                    torch.nn.Dropout(),
                    torch.nn.Linear(w1.shape[1], w1.shape[0]),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(),
                    torch.nn.Linear(w4.shape[1], w4.shape[0])
                )
            else:
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

            model.load_state_dict(checkpoint)
            model.eval()
            return {"type": "model", "model": model}
        else:
            checkpoint.eval()
            return {"type": "model", "model": checkpoint}
    except Exception as e:
        return {"type": "error", "message": str(e)}


def preprocess_keras_image(
    image,
    target_size=(224, 224),
    grayscale=False,
    normalization="zero_one",
):
    if grayscale:
        image = image.convert('L')
    else:
        image = image.convert('RGB')
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)
    if normalization == "mobilenet_v2":
        image_array = image_array / 127.5 - 1.0
    else:
        image_array = image_array / 255.0
    if grayscale:
        image_array = np.expand_dims(image_array, axis=-1)
    input_tensor = np.expand_dims(image_array, axis=0)
    return input_tensor


def correct_skin_class_bias(probabilities):
    """Корректирует перекос HAM10000 к доминирующему классу невусов."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    priors = SKIN_KERAS_CLASS_COUNTS / np.sum(SKIN_KERAS_CLASS_COUNTS)
    corrected = probabilities / priors
    return corrected / np.sum(corrected)


def image_quality_warnings(image):
    """Находит явно плохие снимки без дополнительных библиотек."""
    sample = np.asarray(
        image.convert("L").resize((256, 256)),
        dtype=np.float32,
    )
    brightness = float(np.mean(sample))
    contrast = float(np.std(sample))
    center = sample[1:-1, 1:-1]
    laplacian = (
        sample[:-2, 1:-1]
        + sample[2:, 1:-1]
        + sample[1:-1, :-2]
        + sample[1:-1, 2:]
        - 4.0 * center
    )
    sharpness = float(np.var(laplacian))

    warnings = []
    if brightness < 25:
        warnings.append("снимок слишком тёмный")
    elif brightness > 235:
        warnings.append("снимок пересвечен")
    if contrast < 10:
        warnings.append("слишком низкий контраст")
    if sharpness < 15:
        warnings.append("изображение выглядит размытым")
    return warnings


def preprocess_pytorch_image(image, target_size=(224, 224)):
    _, transforms, _ = get_torch_modules()
    image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def get_image_from_user(label, key):
    """Камера или файл — удобнее с телефона в браузере."""
    img = None
    tab_camera, tab_file = st.tabs(["📷 Камера", "📁 Файл"])
    with tab_camera:
        camera_photo = st.camera_input(label, key=f"{key}_cam")
        if camera_photo is not None:
            with Image.open(camera_photo) as source:
                img = ImageOps.exif_transpose(source).convert("RGB").copy()
    with tab_file:
        uploaded = st.file_uploader(label, type=["jpg", "png", "jpeg"], key=f"{key}_file")
        if uploaded is not None and img is None:
            with Image.open(uploaded) as source:
                img = ImageOps.exif_transpose(source).convert("RGB").copy()
    return img


# ==========================================
# 4. ИНТЕРФЕЙС STREAMLIT
# ==========================================
st.set_page_config(page_title="MIRA", layout="centered", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    @media (max-width: 600px) {
        .block-container {
            padding: 1rem 1rem 3rem;
        }
        h1 {
            font-size: 2.25rem !important;
            line-height: 1.1 !important;
            margin-bottom: 0.35rem !important;
        }
        [data-testid="stSegmentedControl"] button,
        .stButton > button {
            min-height: 46px;
        }
        .stButton > button {
            width: 100%;
        }
        [data-testid="stFileUploaderDropzone"] {
            padding: 0.75rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("MIRA")
st.caption("Сделайте фото камерой или выберите готовое изображение.")

analysis_type = st.segmented_control(
    "Что анализируем?",
    ["Кожа", "Ногти", "Эмоции"],
    selection_mode="single",
    default=None,
)

if analysis_type is None:
    st.info("Выберите режим анализа — модели загрузятся только после выбора.")

# --- ВКЛАДКА 1: КОЖА (КОНСИЛИУМ МОДЕЛЕЙ) ---
if analysis_type == "Кожа":
    st.header("Анализ заболеваний кожи")

    skin_scope = st.segmented_control(
        "Что видно на фото?",
        ["Родинка или пятно", "Сыпь, угри или бородавка"],
        selection_mode="single",
        default="Родинка или пятно",
    )

    model_skin_keras = None
    pt_skin_result = None
    with st.spinner("Первый запуск модели может занять несколько секунд..."):
        if skin_scope == "Родинка или пятно":
            model_skin_keras = load_keras_model(
                'skin_disease_model_focal_recall_FINAL.keras'
            )
        else:
            pt_skin_result = load_pytorch_model(
                'final_fine_tuned_medical_model (2).pth',
                len(CLASSES_SKIN_PYTORCH),
            )

    if isinstance(model_skin_keras, str):
        st.error(f"Ошибка Keras: {model_skin_keras}")
    if pt_skin_result and pt_skin_result.get("type") == "error":
        st.error(f"Ошибка PyTorch: {pt_skin_result['message']}")

    img_skin = get_image_from_user("Снимок кожи", "skin")
    if img_skin:
        quality_issues = image_quality_warnings(img_skin)
        if quality_issues:
            st.warning(
                "Для более точного результата переснимите фото: "
                + ", ".join(quality_issues)
                + "."
            )

        include_skin_heatmap = st.checkbox(
            "Построить карту внимания (медленнее)",
            key="skin_heatmap",
        )

        if st.button("Анализировать кожу"):
            with st.spinner("Проводим анализ..."):
                if model_skin_keras and not isinstance(model_skin_keras, str):
                    input_k = preprocess_keras_image(
                        img_skin,
                        target_size=(224, 224),
                        normalization="mobilenet_v2",
                    )
                    raw_probabilities = model_skin_keras.predict(
                        input_k,
                        verbose=0,
                    )[0]
                    probabilities = correct_skin_class_bias(
                        raw_probabilities,
                    )
                    predicted_idx = int(np.argmax(probabilities))
                    confidence = float(probabilities[predicted_idx])
                    class_name = CLASSES_SKIN_KERAS.get(
                        predicted_idx,
                        f"Класс {predicted_idx}",
                    )
                    heatmap = None
                    if include_skin_heatmap:
                        heatmap = generate_keras_gradcam(
                            model_skin_keras,
                            input_k,
                            predicted_idx,
                        )
                    class_names = CLASSES_SKIN_KERAS

                elif pt_skin_result and pt_skin_result.get("type") != "error":
                    torch, _, _ = get_torch_modules()
                    model_skin_pt = pt_skin_result["model"]
                    input_pt = preprocess_pytorch_image(img_skin)

                    with torch.inference_mode():
                        outputs = model_skin_pt(input_pt)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        predicted_idx = int(torch.argmax(probabilities).item())
                        confidence = float(probabilities[predicted_idx].item())
                        probabilities = probabilities.numpy()
                    class_name = CLASSES_SKIN_PYTORCH.get(
                        predicted_idx,
                        f"Класс {predicted_idx}",
                    )
                    heatmap = None
                    if include_skin_heatmap:
                        heatmap = generate_pytorch_gradcam(
                            model_skin_pt,
                            input_pt,
                            predicted_idx,
                        )
                    class_names = CLASSES_SKIN_PYTORCH
                else:
                    probabilities = None

                if probabilities is not None:
                    sorted_probabilities = np.sort(np.asarray(probabilities))
                    margin = float(
                        sorted_probabilities[-1] - sorted_probabilities[-2]
                    )
                    if confidence < 0.55 or margin < 0.10:
                        st.warning(
                            f"**Предварительный результат:** {class_name}. "
                            "Модель не уверена — сделайте ещё одно чёткое фото "
                            "при ровном освещении."
                        )
                    else:
                        st.success(f"**Предварительный результат:** {class_name}")
                    st.info(f"Уверенность модели: **{confidence * 100:.1f}%**")

                    if heatmap is not None:
                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            st.image(img_skin, caption="Исходный снимок", width="stretch")
                        with col_img2:
                            heatmap_overlay = overlay_heatmap(img_skin, heatmap)
                            st.image(heatmap_overlay, caption="Карта внимания модели (Grad-CAM)",
                                     width="stretch")
                    else:
                        st.image(img_skin, caption="Исходный снимок", width="stretch")

                    import pandas as pd

                    chart_df = pd.DataFrame(
                        {"Уверенность": probabilities},
                        index=[
                            class_names.get(i, f"Класс {i}")
                            for i in range(len(probabilities))
                        ],
                    )
                    st.subheader("Распределение вероятностей")
                    st.bar_chart(chart_df)
                else:
                    st.error("Не удалось получить предсказание модели.")

# --- ВКЛАДКА 2: НОГТИ ---
if analysis_type == "Ногти":
    st.header("Анализ состояния ногтей")
    with st.spinner("Первый запуск модели может занять несколько секунд..."):
        pt_result = load_pytorch_model('final_nails_model (1).pth', len(CLASSES_NAILS))

    if pt_result and pt_result.get("type") != "error":
        img_nails = get_image_from_user("Снимок ногтей", "nails")
        if img_nails:
            include_nails_heatmap = st.checkbox(
                "Построить карту внимания (медленнее)",
                key="nails_heatmap",
            )

            if st.button("Анализировать ногти"):
                with st.spinner("Анализируем состояние ногтей..."):
                    torch, _, _ = get_torch_modules()
                    model_nails = pt_result["model"]

                    input_pt = preprocess_pytorch_image(img_nails)

                    with torch.inference_mode():
                        outputs = model_nails(input_pt)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        predicted_idx = torch.argmax(probabilities).item()
                        confidence = float(torch.max(probabilities).item())

                    class_name = CLASSES_NAILS.get(predicted_idx, f"Класс {predicted_idx}")
                    st.success(f"**Состояние ногтей:** {class_name} (Уверенность: {confidence * 100:.1f}%)")

                    heatmap = None
                    if include_nails_heatmap:
                        heatmap = generate_pytorch_gradcam(model_nails, input_pt, predicted_idx)

                    if heatmap is not None:
                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            st.image(img_nails, caption="Исходный снимок", width="stretch")
                        with col_img2:
                            heatmap_overlay = overlay_heatmap(img_nails, heatmap)
                            st.image(heatmap_overlay, caption="Карта внимания модели", width="stretch")
                    else:
                        st.image(img_nails, caption="Исходный снимок", width="stretch")

                    # Диаграмма с текстовыми названиями
                    import pandas as pd

                    chart_df = pd.DataFrame(
                        {"Уверенность": probabilities.numpy()},
                        index=[CLASSES_NAILS.get(i, f"Класс {i}") for i in range(len(probabilities))]
                    )
                    st.subheader("Распределение вероятностей")
                    st.bar_chart(chart_df)

# --- ВКЛАДКА 3: ЭМОЦИИ ---
if analysis_type == "Эмоции":
    st.header("Распознавание эмоций")
    model_emotions = load_keras_model('emotion_model_attempt_63_plus.keras')

    if isinstance(model_emotions, str):
        st.error(f"Ошибка загрузки: {model_emotions}")
    elif model_emotions:
        img_emotions = get_image_from_user("Фото лица", "emotions")
        if img_emotions:
            include_emotions_heatmap = st.checkbox(
                "Построить карту внимания (медленнее)",
                key="emotions_heatmap",
            )

            if st.button("Распознать эмоцию"):
                with st.spinner("Анализ лица..."):
                    input_k = preprocess_keras_image(img_emotions, target_size=(48, 48), grayscale=True)
                    preds = model_emotions.predict(input_k, verbose=0)
                    predicted_idx = int(np.argmax(preds[0]))
                    confidence = float(np.max(preds[0]))
                    class_name = CLASSES_EMOTIONS.get(predicted_idx, f"Класс {predicted_idx}")

                    st.success(f"**Распознанная эмоция:** {class_name} (Уверенность: {confidence * 100:.1f}%)")

                    heatmap = None
                    if include_emotions_heatmap:
                        heatmap = generate_keras_gradcam(model_emotions, input_k, predicted_idx)

                    if heatmap is not None:
                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            st.image(img_emotions, caption="Исходный снимок", width="stretch")
                        with col_img2:
                            heatmap_overlay = overlay_heatmap(img_emotions, heatmap)
                            st.image(heatmap_overlay, caption="Карта внимания модели", width="stretch")
                    else:
                        st.image(img_emotions, caption="Исходный снимок", width="stretch")

                    # Диаграмма с названими эмоций
                    import pandas as pd

                    chart_df = pd.DataFrame(
                        {"Уверенность": preds[0]},
                        index=[CLASSES_EMOTIONS.get(i, f"Класс {i}") for i in range(len(preds[0]))]
                    )
                    st.subheader("Распределение вероятностей")
                    st.bar_chart(chart_df)
