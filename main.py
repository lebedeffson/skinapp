import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import keras
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import tensorflow as tf
import matplotlib.cm as cm

# ==========================================
# 0. СЛОВАРИ КЛАССОВ (НАЗВАНИЯ ВМЕСТО ИНДЕКСОВ)
# ==========================================
CLASSES_SKIN_KERAS = {
    0: "Базалиома",
    1: "Кератоз",
    2: "Меланома",
    3: "Родинка"
}

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
# 1. ЖЕСТКИЙ ОБХОД ОШИБКИ KERAS
# ==========================================
original_dense_init = keras.layers.Dense.__init__


def safe_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)


keras.layers.Dense.__init__ = safe_dense_init


# ==========================================
# 2. ФУНКЦИИ ГЕНЕРАЦИИ ТЕПЛОВЫХ КАРТ (Grad-CAM)
# ==========================================
def overlay_heatmap(pil_img, heatmap, alpha=0.4):
    """Накладывает тепловую карту внимания на оригинальное изображение"""
    img_np = np.array(pil_img.convert('RGB'))
    h, w, _ = img_np.shape

    # Изменяем размер карты под оригинал
    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize((w, h), Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img) / 255.0

    # Накладываем цветовое палитра JET (красный = максимум внимания)
    cmap = cm.get_cmap('jet')
    color_heatmap = cmap(heatmap_resized)[:, :, :3]
    color_heatmap = np.uint8(color_heatmap * 255)

    # Смешиваем изображения
    overlay = np.uint8(img_np * (1 - alpha) + color_heatmap * alpha)
    return Image.fromarray(overlay)


def generate_pytorch_gradcam(model, input_tensor, target_class=None):
    """Grad-CAM для PyTorch (ResNet18)"""
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
    try:
        # Находим последний сверточный слой
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv2D) or 'conv' in layer.name.lower():
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_array)
            if target_class is None:
                target_class = tf.argmax(predictions[0])
            loss = predictions[:, target_class]

        grads = tape.gradient(loss, conv_outputs)
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
@st.cache_resource
def load_keras_model(model_path):
    if not os.path.exists(model_path):
        return None
    try:
        return keras.models.load_model(model_path, compile=False)
    except Exception as e:
        return f"ERROR: {e}"


@st.cache_resource
def load_pytorch_model(model_path):
    if not os.path.exists(model_path):
        return None
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        if isinstance(checkpoint, dict):
            return {"type": "state_dict", "data": checkpoint}
        else:
            checkpoint.eval()
            return {"type": "full_model", "model": checkpoint}
    except Exception as e:
        return {"type": "error", "message": str(e)}


def preprocess_keras_image(image, target_size=(224, 224), grayscale=False):
    if grayscale:
        image = image.convert('L')
    else:
        image = image.convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    if grayscale:
        image_array = np.expand_dims(image_array, axis=-1)
    input_tensor = np.expand_dims(image_array, axis=0)
    return input_tensor


def preprocess_pytorch_image(image, target_size=(224, 224)):
    image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


# ==========================================
# 4. ИНТЕРФЕЙС STREAMLIT
# ==========================================
st.set_page_config(page_title="Нейросети", layout="centered")
st.title("🧠 Мультимодельный анализатор")

tab_skin, tab_nails, tab_emotions = st.tabs(["🩺 Кожа", "💅 Ногти", "😊 Эмоции"])

# --- ВКЛАДКА 1: КОЖА (КОНСИЛИУМ МОДЕЛЕЙ) ---
with tab_skin:
    st.header("Анализ заболеваний кожи")

    model_skin_keras = load_keras_model('skin_disease_model_focal_recall_FINAL.keras')
    pt_skin_result = load_pytorch_model('final_fine_tuned_medical_model (2).pth')

    if isinstance(model_skin_keras, str):
        st.error(f"Ошибка Keras: {model_skin_keras}")
    if pt_skin_result and pt_skin_result.get("type") == "error":
        st.error(f"Ошибка PyTorch: {pt_skin_result['message']}")

    file_skin = st.file_uploader("Загрузите снимок кожи", type=["jpg", "png", "jpeg"], key="skin")
    if file_skin:
        img_skin = Image.open(file_skin)

        if st.button("Анализировать кожу"):
            with st.spinner("Проводим консилиум нейросетей и построение тепловой карты..."):

                best_confidence = -1.0
                best_class_name = "Неизвестно"
                winning_model_name = ""
                winning_chart_df = None
                winning_heatmap = None

                # 1. Спрашиваем Keras
                if model_skin_keras and not isinstance(model_skin_keras, str):
                    input_k = preprocess_keras_image(img_skin, target_size=(224, 224))
                    preds_k = model_skin_keras.predict(input_k)
                    max_prob_k = float(np.max(preds_k[0]))
                    idx_k = int(np.argmax(preds_k[0]))

                    if max_prob_k > best_confidence:
                        best_confidence = max_prob_k
                        best_class_name = CLASSES_SKIN_KERAS.get(idx_k, f"Класс {idx_k}")
                        winning_model_name = "Keras"

                        # Собираем DataFrame с текстовыми названиями вместо индексов
                        winning_chart_df = pd.DataFrame(
                            {"Уверенность": preds_k[0]},
                            index=[CLASSES_SKIN_KERAS.get(i, f"Класс {i}") for i in range(len(preds_k[0]))]
                        )
                        winning_heatmap = generate_keras_gradcam(model_skin_keras, input_k, idx_k)

                # 2. Спрашиваем PyTorch
                if pt_skin_result and pt_skin_result.get("type") != "error":
                    if pt_skin_result["type"] == "full_model":
                        model_skin_pt = pt_skin_result["model"]
                    else:
                        state_dict = pt_skin_result["data"]
                        model_skin_pt = models.resnet18()
                        if "fc.1.weight" in state_dict and "fc.4.weight" in state_dict:
                            w1, w4 = state_dict["fc.1.weight"], state_dict["fc.4.weight"]
                            model_skin_pt.fc = torch.nn.Sequential(
                                torch.nn.Dropout(),
                                torch.nn.Linear(w1.shape[1], w1.shape[0]),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(),
                                torch.nn.Linear(w4.shape[1], w4.shape[0])
                            )
                        else:
                            model_skin_pt.fc = torch.nn.Linear(model_skin_pt.fc.in_features, len(CLASSES_SKIN_PYTORCH))

                        model_skin_pt.load_state_dict(state_dict)
                        model_skin_pt.eval()

                    input_pt = preprocess_pytorch_image(img_skin)

                    # Предсказание
                    with torch.no_grad():
                        outputs = model_skin_pt(input_pt)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        max_prob_pt = float(torch.max(probabilities).item())
                        idx_pt = torch.argmax(probabilities).item()

                    if max_prob_pt > best_confidence:
                        best_confidence = max_prob_pt
                        best_class_name = CLASSES_SKIN_PYTORCH.get(idx_pt, f"Класс {idx_pt}")
                        winning_model_name = "PyTorch"

                        winning_chart_df = pd.DataFrame(
                            {"Уверенность": probabilities.numpy()},
                            index=[CLASSES_SKIN_PYTORCH.get(i, f"Класс {i}") for i in range(len(probabilities))]
                        )
                        # Генерация Grad-CAM для PyTorch
                        winning_heatmap = generate_pytorch_gradcam(model_skin_pt, input_pt, idx_pt)

                # 3. Вывод результатов
                if best_confidence > -1.0:
                    st.success(f"**Итоговый диагноз:** {best_class_name}")
                    st.info(f"Уверенность: **{best_confidence * 100:.1f}%** (Выбрана модель {winning_model_name})")

                    # Выводим изображения: Исходник и Тепловая карта
                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        st.image(img_skin, caption="Исходный снимок", use_container_width=True)
                    with col_img2:
                        if winning_heatmap is not None:
                            heatmap_overlay = overlay_heatmap(img_skin, winning_heatmap)
                            st.image(heatmap_overlay, caption="Карта внимания модели (Grad-CAM)",
                                     use_container_width=True)
                        else:
                            st.image(img_skin, caption="Тепловая карта недоступна", use_container_width=True)

                    st.subheader("Распределение вероятностей")
                    st.bar_chart(winning_chart_df)
                else:
                    st.error("Не удалось получить предсказания ни от одной из моделей.")

# --- ВКЛАДКА 2: НОГТИ ---
with tab_nails:
    st.header("Анализ состояния ногтей")
    pt_result = load_pytorch_model('final_nails_model (1).pth')

    if pt_result and pt_result.get("type") != "error":
        file_nails = st.file_uploader("Загрузите снимок ногтей", type=["jpg", "png", "jpeg"], key="nails")
        if file_nails:
            img_nails = Image.open(file_nails)

            if st.button("Анализировать ногти"):
                with st.spinner("Анализ состояния ногтей и построение карты внимания..."):
                    if pt_result["type"] == "full_model":
                        model_nails = pt_result["model"]
                    else:
                        state_dict = pt_result["data"]
                        model_nails = models.resnet18()
                        if "fc.1.weight" in state_dict and "fc.4.weight" in state_dict:
                            w1, w4 = state_dict["fc.1.weight"], state_dict["fc.4.weight"]
                            model_nails.fc = torch.nn.Sequential(
                                torch.nn.Dropout(),
                                torch.nn.Linear(w1.shape[1], w1.shape[0]),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(),
                                torch.nn.Linear(w4.shape[1], w4.shape[0])
                            )
                        else:
                            model_nails.fc = torch.nn.Linear(model_nails.fc.in_features, len(CLASSES_NAILS))

                        model_nails.load_state_dict(state_dict)
                        model_nails.eval()

                    input_pt = preprocess_pytorch_image(img_nails)

                    with torch.no_grad():
                        outputs = model_nails(input_pt)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        predicted_idx = torch.argmax(probabilities).item()
                        confidence = float(torch.max(probabilities).item())

                    class_name = CLASSES_NAILS.get(predicted_idx, f"Класс {predicted_idx}")
                    st.success(f"**Состояние ногтей:** {class_name} (Уверенность: {confidence * 100:.1f}%)")

                    # Накладываем тепловую карту
                    heatmap = generate_pytorch_gradcam(model_nails, input_pt, predicted_idx)
                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        st.image(img_nails, caption="Исходный снимок", use_container_width=True)
                    with col_img2:
                        if heatmap is not None:
                            heatmap_overlay = overlay_heatmap(img_nails, heatmap)
                            st.image(heatmap_overlay, caption="Карта внимания модели", use_container_width=True)
                        else:
                            st.image(img_nails, caption="Карта внимания недоступна", use_container_width=True)

                    # Диаграмма с текстовыми названиями
                    chart_df = pd.DataFrame(
                        {"Уверенность": probabilities.numpy()},
                        index=[CLASSES_NAILS.get(i, f"Класс {i}") for i in range(len(probabilities))]
                    )
                    st.subheader("Распределение вероятностей")
                    st.bar_chart(chart_df)

# --- ВКЛАДКА 3: ЭМОЦИИ ---
with tab_emotions:
    st.header("Распознавание эмоций")
    model_emotions = load_keras_model('emotion_model_attempt_63_plus.keras')

    if isinstance(model_emotions, str):
        st.error(f"Ошибка загрузки: {model_emotions}")
    elif model_emotions:
        file_emotions = st.file_uploader("Загрузите фото лица", type=["jpg", "png", "jpeg"], key="emotions")
        if file_emotions:
            img_emotions = Image.open(file_emotions)

            if st.button("Распознать эмоцию"):
                with st.spinner("Анализ лица..."):
                    input_k = preprocess_keras_image(img_emotions, target_size=(48, 48), grayscale=True)
                    preds = model_emotions.predict(input_k)
                    predicted_idx = int(np.argmax(preds[0]))
                    confidence = float(np.max(preds[0]))
                    class_name = CLASSES_EMOTIONS.get(predicted_idx, f"Класс {predicted_idx}")

                    st.success(f"**Распознанная эмоция:** {class_name} (Уверенность: {confidence * 100:.1f}%)")

                    # Тепловая карта для эмоций
                    heatmap = generate_keras_gradcam(model_emotions, input_k, predicted_idx)
                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        st.image(img_emotions, caption="Исходный снимок", use_container_width=True)
                    with col_img2:
                        if heatmap is not None:
                            heatmap_overlay = overlay_heatmap(img_emotions, heatmap)
                            st.image(heatmap_overlay, caption="Карта внимания модели", use_container_width=True)
                        else:
                            st.image(img_emotions, caption="Карта внимания недоступна", use_container_width=True)

                    # Диаграмма с названими эмоций
                    chart_df = pd.DataFrame(
                        {"Уверенность": preds[0]},
                        index=[CLASSES_EMOTIONS.get(i, f"Класс {i}") for i in range(len(preds[0]))]
                    )
                    st.subheader("Распределение вероятностей")
                    st.bar_chart(chart_df)