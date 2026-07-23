# Skin-model inference

The Keras pigmented-lesion model is a MobileNetV2 transfer-learning model. Its
frozen ImageNet backbone expects pixels in `[-1, 1]`, not the generic `0..1`
normalization used by the emotion model.

The four output classes follow the HAM10000 subset order:

1. basal cell carcinoma (`bcc`, 514);
2. benign keratosis (`bkl`, 1099);
3. melanoma (`mel`, 1113);
4. melanocytic nevus (`nv`, 6705).

Correct the output by those training priors before normalization. Without the
correction, the model strongly defaults to the dominant nevus class. If the
model is later retrained on balanced data or calibrated probabilities, remove
or re-estimate this correction from a held-out validation set.

Do not choose between the Keras and PyTorch skin models by comparing raw
softmax confidence. Their class sets are disjoint and their probabilities are
not calibrated against each other. Route pigmented lesions to Keras and
rash/acne/warts to PyTorch.

On 2026-07-23, a balanced 16-image ISIC smoke set (four public images per class)
improved from 6/16 with the old pipeline to 11/16 with MobileNetV2 preprocessing
and prior correction. This is regression evidence only, not a clinical accuracy
claim; use a larger held-out, patient-separated set before changing thresholds
or making performance claims.
