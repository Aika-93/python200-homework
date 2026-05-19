# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# ### Task 1: Environment Setup and Data Loading

# %%
import torch
import torchvision
from torchvision import models, transforms
from torchvision.models import (
    ResNet18_Weights,
    MobileNet_V3_Small_Weights,
    EfficientNet_B0_Weights,
)
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import copy
import os
from pathlib import Path
from sklearn.decomposition import PCA

os.makedirs("outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = Path("/kaggle/input/datasets/puneet6060/intel-image-classification/seg_test/seg_test")
LABELS   = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

random.seed(42)

# %%
def load_images(n_per_class=10):
    """Load n images per class. Returns a list of (PIL.Image, label_string) tuples."""
    image_set = []
    for label in LABELS:
        class_dir = DATA_DIR / label
        paths = random.sample(list(class_dir.glob("*.jpg")), n_per_class)
        for path in paths:
            img = Image.open(path).convert("RGB")
            image_set.append((img, label))
    random.shuffle(image_set)
    return image_set

image_set = load_images(n_per_class=10)
print(f"Total images loaded: {len(image_set)}")

# %%
fig, axes = plt.subplots(2, 3, figsize=(10, 7))

for ax, label in zip(axes.flatten(), LABELS):
    class_dir = DATA_DIR / label
    img_path = random.choice(list(class_dir.glob("*")))
    img = Image.open(img_path).convert("RGB")

    ax.imshow(img)
    ax.set_title(label)
    ax.axis("off")

plt.tight_layout()

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/dataset_sample.png")
plt.show()

# It is a good starting point because the model has already learned useful visual features (edges, textures, shapes).
# These features can still be useful for recognizing new scene types, even if the exact classes are different.

# %% [markdown]
# ### Task 2: Baseline Inference with ResNet18

# %%
resnet_weights   = ResNet18_Weights.DEFAULT
resnet           = models.resnet18(weights=resnet_weights).to(device).eval()
resnet_preproc   = resnet_weights.transforms()
imagenet_classes = resnet_weights.meta["categories"]


def run_inference(model, preprocess, image, device, class_labels, top_k=5):
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    probs = torch.softmax(output[0], dim=0)
    top_probs, top_idxs = torch.topk(probs, top_k)
    results=[]
    for prob, idx in zip(top_probs, top_idxs):
        results.append((class_labels[idx], prob.item()))

    return results

print(f"ResNet18 parameters: {sum(p.numel() for p in resnet.parameters()):,}")
resnet_results = []
for img, true_label in image_set:
    preds = run_inference(resnet, resnet_preproc, img, device, imagenet_classes)
    resnet_results.append({
        "true_label":   true_label,
        "top1_class":   preds[0][0],
        "top1_prob":    preds[0][1],
        "top5_classes": [p[0] for p in preds],
        "top5_probs":   [p[1] for p in preds],
    })

print(f"Processed {len(resnet_results)} images.")

df = pd.DataFrame(resnet_results)

# Overall mean
print(df["top1_prob"].mean())
# Mean top-1 probability per class
print(df.groupby("true_label")["top1_prob"].mean())

data = []
labels = []

for label in LABELS:
    probs = [r["top1_prob"] for r in resnet_results if r["true_label"] == label]
    data.append(probs)
    labels.append(label)

plt.figure(figsize=(10,6))
plt.boxplot(data, tick_labels=labels)
plt.title("ResNet18 Top-1 Confidence by Class")
plt.ylabel("Top-1 Probability")
plt.xticks(rotation=30)

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/resnet18_confidence_by_class.png")
plt.show()

# High confidence doesn't always mean the prediction is correct, we use a threshold (around 0.7-0.8):
# above it we auto-accept predictions, below it we send them for human review.

# %% [markdown]
# ### Task 3: Multi-Model Comparison
# 

# %%
# MobileNetV3-Small — designed for mobile and edge deployment
mobile_weights = MobileNet_V3_Small_Weights.DEFAULT
mobilenet      = models.mobilenet_v3_small(weights=mobile_weights).to(device).eval()
mobile_preproc = mobile_weights.transforms()

# EfficientNet-B0 — designed to maximize accuracy per unit of compute
effnet_weights = EfficientNet_B0_Weights.DEFAULT
efficientnet   = models.efficientnet_b0(weights=effnet_weights).to(device).eval()
effnet_preproc = effnet_weights.transforms()

# Print parameter counts for all three
for name, m in [("ResNet18",          resnet),
                ("MobileNetV3-Small", mobilenet),
                ("EfficientNet-B0",   efficientnet)]:
    params = sum(p.numel() for p in m.parameters())
    print(f"{name:22s}  {params:>12,} parameters")

# Smaller models have lower capacity, meaning they are less expressive but more efficient. They are better suited for resource-limited devices like phones.
# Larger models have higher capacity and usually better accuracy, but require more computation and are better suited for cloud deployment.

mobilenet_results = []
for img, true_label in image_set:
    preds = run_inference(mobilenet, mobile_preproc, img, device, imagenet_classes)
    mobilenet_results.append({
        "true_label":   true_label,
        "top1_class":   preds[0][0],
        "top1_prob":    preds[0][1],
        "top5_classes": [p[0] for p in preds],
        "top5_probs":   [p[1] for p in preds],
    })

print(f"Processed {len(mobilenet_results)} images.")

effnet_results = []
for img, true_label in image_set:
    preds = run_inference(efficientnet, effnet_preproc, img, device, imagenet_classes)
    effnet_results.append({
        "true_label":   true_label,
        "top1_class":   preds[0][0],
        "top1_prob":    preds[0][1],
        "top5_classes": [p[0] for p in preds],
        "top5_probs":   [p[1] for p in preds],
    })

print(f"Processed {len(effnet_results)} images.")

os.makedirs("outputs", exist_ok=True)

sample_images = image_set[:6]

fig, axes = plt.subplots(len(sample_images), 4, figsize=(18,18))
for i, (img, true_label) in enumerate(sample_images):

    # inference for all model
    res_preds = run_inference(resnet, resnet_preproc, img, device, imagenet_classes)
    mob_preds = run_inference(mobilenet, mobile_preproc, img, device, imagenet_classes)
    eff_preds = run_inference(efficientnet, effnet_preproc, img, device, imagenet_classes)

    # image
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"True: {true_label}")
    axes[i, 0].axis("off")

    # helper function for text
    def format_preds(preds):
        return "\n".join([f"{p[0]}: {p[1]:.2f}" for p in preds[:3]])
    
    # ResNet
    axes[i, 1].text(0.1, 0.5, format_preds(res_preds), fontsize=9)
    axes[i, 1].set_title("ResNet18")
    axes[i, 1].axis("off")

    # mobileNet
    axes[i, 2].text(0.1, 0.5, format_preds(mob_preds), fontsize=9)
    axes[i, 2].set_title("MobileNet")
    axes[i, 2].axis("off")

    # efficientNet
    axes[i, 3].text(0.1, 0.5, format_preds(eff_preds), fontsize=9)
    axes[i, 3].set_title("EfficientNet")
    axes[i, 3].axis("off")

plt.tight_layout()
plt.savefig("outputs/model_comparison_grid.png")
plt.show()

# Yes, often give similar answers.
# Yes, they disagree on some images, combining models could give better results.
# EfficientNet usually gives the most meaningful and realistic top-3 predictions

# %% [markdown]
# ### Task 4: Speed vs. Accuracy Tradeoff

# %%
def benchmark_model(model, preprocess, image_set, device, n_warmup=5):
    """
    Benchmark single-image inference speed.
    Returns mean latency in milliseconds per image.
    """
    # Warm up the GPU — the first few calls are slower due to CUDA initialization
    for img, _ in image_set[:n_warmup]:
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(tensor)

    # Timed run — synchronize before and after to get accurate GPU timing
    torch.cuda.synchronize()
    start = time.time()

    for img, _ in image_set:
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(tensor)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    return (elapsed / len(image_set)) * 1000  # milliseconds per image

resnet_ms  = benchmark_model(resnet,       resnet_preproc,  image_set, device)
mobile_ms  = benchmark_model(mobilenet,    mobile_preproc,  image_set, device)
effnet_ms  = benchmark_model(efficientnet, effnet_preproc,  image_set, device)

print(f"ResNet18:           {resnet_ms:.2f} ms/image")
print(f"MobileNetV3-Small:  {mobile_ms:.2f} ms/image")
print(f"EfficientNet-B0:    {effnet_ms:.2f} ms/image")

os.makedirs("outputs", exist_ok=True)

models = ["ResNet", "MobileNet", "EfficientNet"]
latency = [resnet_ms, mobile_ms, effnet_ms]

plt.figure(figsize=(8,5))
plt.bar(models, latency)
plt.title("Model Inference Latency(ms per image)")
plt.xlabel("Model")
plt.ylabel("Latency")

plt.tight_layout()
plt.savefig("outputs/inference_speed.png")
plt.show()

summary = pd.DataFrame({
    "Model": ["ResNet", "MobileNet", "EfficientNet"],
    "Parameters": [
        sum(p.numel() for p in resnet.parameters()),
        sum(p.numel() for p in mobilenet.parameters()),
        sum(p.numel() for p in efficientnet.parameters())
    ],
    "ms_per_image": [
        resnet_ms, mobile_ms, effnet_ms
    ]
})
print(summary)

# To process 50 images per second, the maximum tolerable latency is 20 ms per image.
# All three models meet this requirement.
# ResNet is the fastest and best suited for real-time constraints.

# For (a) a high-throughput cloud pipeline I would choose ResNet because it offers the best balance of speed and performance.
# For (b) an on-device mobile app, MobileNet is the best choice due to its efficiency and low latency.
# For (c) a safety-critical system, I would choose EfficientNet because accuracy is more important than speed.

# %% [markdown]
# ### Task 5: Pretrained Features as a Window into Transfer Learning

# %%
import copy

feature_extractor = copy.deepcopy(resnet)
feature_extractor.fc = torch.nn.Identity()   # remove the classification head
feature_extractor    = feature_extractor.to(device).eval()

def extract_features(model, preprocess, image, device):
    """Extract a feature vector from an image using the truncated CNN."""
    tensor   = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor)
    return features.squeeze().cpu().numpy()

# Extract features for all images
feature_vectors = []
true_labels     = []

for img, label in image_set:
    feat = extract_features(feature_extractor, resnet_preproc, img, device)
    feature_vectors.append(feat)
    true_labels.append(label)

feature_matrix = np.array(feature_vectors)
print(f"Feature matrix shape: {feature_matrix.shape}")
# Expected: (60, 512) — 60 images, 512-dimensional feature vector each

pca          = PCA(n_components=2)
features_2d  = pca.fit_transform(feature_matrix)

fig, ax = plt.subplots(figsize=(8, 6))
colors  = plt.cm.tab10(np.linspace(0, 1, len(LABELS)))

for i, label in enumerate(LABELS):
    mask = [l == label for l in true_labels]
    ax.scatter(
        features_2d[mask, 0],
        features_2d[mask, 1],
        label=label, color=colors[i], s=60, alpha=0.75
    )

ax.legend()
ax.set_title("ResNet18 Feature Embeddings (PCA to 2D)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.tight_layout()
plt.savefig("outputs/feature_embeddings.png")
plt.show()

# Images from the same class are often close to each other on the plot.
# This means pretrained model already know how to recognize similar visual patterns(like shapes, colors, and textures),
# even without being trained on these specific classes.Урок по обучению на трансфере описал две стратегии: извлечение особенностей (заморозить все предварительно подготовленные слои, тренировать только новый финальный слой) и точная настройка (позволить некоторым или всем предварительно подготовленным весам обновляться во время обучения). Если бы вы адаптировали ResNet18 для новой задачи - скажем, классификацию рентгеновских изображений на нормальные/ненормальные образцы - и у вас было только 500 обозначенных примеров, с какой стратегии вы бы начали и почему?

# I would start with feature extraction. 
# Freezing the pretrained ResNet and trainig only the final layer helps prevent overfitting when data is limited.


# %% [markdown]
# ### Task 6: Summary and Recommendation

# %% [markdown]
# Model Comparison: EfficientNet performs best in terms of prediction quality, producing the most semantically meaningful top-3 predictions across the dataset. However, it has highest latency, making it the slowest model.
# 
# Confidence Calibration: ResNet is more confident when the image is easier to recognize, like mountains or sea. It is less confident for more confusing scenes like forest or street, where images can look more different from each other. This make sense because some scenes have clearer shapes and structure than others.
# 
# Production Recommendation: 
#     - EfficientNet is a good choice because it gives the best results.
#     - The images should be resized, normalized, and prepared the same way as during training, and          also converted to RGB before the model sees them.
#     - One problem is that the model can still be confidently wrong. So if the confidence is low(like       under 0.7-0.8), the result should be checked by a person
#     


