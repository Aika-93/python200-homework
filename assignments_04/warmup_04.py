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

# %%
import os
os.makedirs("outputs", exist_ok=True)

# %%
import torch
import torchvision
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

# The standard device check — you'll use this pattern in every PyTorch notebook
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version:     {torch.__version__}")
print(f"TorchVision version: {torchvision.__version__}")

# %% [markdown]
# ## PyTorch Tensors 

# %% [markdown]
# ### Q1

# %%
a = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

b = torch.zeros(2, 3)
c = torch.ones(4)

# Tensor a 
print("a:", a)
print("shape:",a.shape)
print("dtype:", a.dtype)
print("device:", a.device)

print("\nb:", b)
print("shape:", b.shape)
print("dtype:", b.dtype)
print("device:", b.device)

print("\nc:", c)
print("shape:",c.shape)
print("dtype:", c.dtype)
print("device:", c.device)

# These tensors currently on the CPU

# It uis important that both the model and input tensors are on the same device
# because PyTorch cannot perform operations between CPU and GPU tensors.
# IF they are on different devices, it will cause an error

# %% [markdown]
# ### Q2

# %%
x = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0])

print("sqrt:", torch.sqrt(x))
print("sum:", x.sum())
print("mean:", x.mean())
print("argmax:", x.argmax())

# .argmax() returns the class with the highest predicted score.

# %% [markdown]
# ### Q3

# %%
a_gpu   = a.to(device)
print(f"a_gpu device: {a_gpu.device}")

a_back  = a_gpu.cpu()
a_numpy = a_back.numpy()
print(f"numpy type: {type(a_numpy)}")
print(f"numpy values:\n{a_numpy}")

# PyTorch requires .cpu() because NumPy cannot work with GPU tensors.
# NumPy arrays live in CPU memory, not on the GPU

# %% [markdown]
# ### Q4

# %%
t = torch.arange(24).float()

t_reshaped_1 = t.reshape(4,6)
print(t_reshaped_1.shape)

t_reshaped_2 = t.reshape(2, 3, 4)
print(t_reshaped_2.shape)

t_reshaped_3 = t_reshaped_1.unsqueeze(0)
print(t_reshaped_3.shape)

# The operation that accompishes is unsqueeze(0).
# This matters because neural networks are designed to always process inputs in batches
# even when there is only one image.

# %% [markdown]
# ### Q5

# %%
np_a = np.array([[1.0, 2.0], [3.0, 4.0]])
np_b = np.array([[5.0, 6.0], [7.0, 8.0]])

t_a  = torch.tensor(np_a, dtype=torch.float32)
t_b  = torch.tensor(np_b, dtype=torch.float32)

np_math = np.matmul(np_a, np_b)
t_math = torch.matmul(t_a, t_b)

print(np_math)
print(t_math)
print(np.allclose(np_math, t_math.numpy()))

# Matrix multiplication transforms input data into learned features inside a neural network layer.

# %% [markdown]
# ## Pretrained Models

# %% [markdown]
# ## Q1

# %%
weights = ResNet18_Weights.DEFAULT
model   = models.resnet18(weights=weights)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Using pretrained model is very practical because it saves a huge amount of time, data, and computational recources.
# Instead of training ResNet18 from scratch on millions of images, we can start from already learned features 
# and fine-tune the model for our task
# This is especially important when we have limited budget and tight deadline

# %% [markdown]
# ## Q2

# %%
print(model)

# The final layer in ResNet-18 is the fully connected (fc) layer.
# Its output size is 1000, corresponding to the 1000 ImageNet classes the model can predict.

# A deep network has many layers (like layer1 to layer4).
# Each layer learns something from the data step by step.
# Early layers find simple things like edges,
# and deeper layers find more complex things like shapes or objects.
# This is why it is called "deep".


# %% [markdown]
# ## Q3

# %% [markdown]
# 

# %%
model = model.to(device)
model.eval()
print("Model ready for inference.")

# .to(device) ensures model and data are on the same device so computation works correctly

# model.eval() puts model in prediction mode.
# BatchNorm behave differently in train and eval.

# %% [markdown]
# ## Q4

# %%
preprocess = weights.transforms()
print(preprocess)

# Resize/crop makes all images the same size (e.g., 224x224),
# so they match what the model expects and can be processed without errors.

# ToTensor() changes pixels from 0-255 to 0-1.
# This makes the data easier for neural networks to process.

# Normalization rescales pixel values using mean and std.
# It uses ImageNet values because the model was trained on ImageNet,
# so inputs must match the same distribution for good performance.

# %% [markdown]
# ## Running Inference

# %%
import random
random.seed(42)

DATA_DIR = Path("/kaggle/input/datasets/puneet6060/intel-image-classification/seg_test/seg_test")
LABELS   = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

def load_sample_image(label):
    """Load a random image file from the given class folder."""
    class_dir = DATA_DIR / label
    img_path  = random.choice(list(class_dir.glob("*.jpg")))
    return Image.open(img_path).convert("RGB"), img_path.name

imagenet_classes = weights.meta["categories"]
print(f"Number of classes: {len(imagenet_classes)}")
print(f"First 5 labels: {imagenet_classes[:5]}")


# %% [markdown]
# ## Q1

# %%
def get_top5_predictions(model, preprocess, image, device, class_labels):
    """
    Run inference on a PIL image and return the top-5 predictions.
    Returns a list of (class_name, probability) tuples.
    """
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    probs = torch.softmax(output[0], dim=0)
    
    top_probs, top_idxs = torch.topk(probs, 5)

    results=[]
    for prob, idx in zip(top_probs, top_idxs):
        results.append((class_labels[idx], prob.item()))

    return results

img, img_name = load_sample_image("mountain")
preds         = get_top5_predictions(model, preprocess, img, device, imagenet_classes)

print(f"\nTop-5 predictions for '{img_name}':")
for class_name, prob in preds:
    print(f"  {class_name:30s}  {prob:.4f}")

# Yes, the top prediction make sense.
# Several of the top-5 predictions (especially "alp", and "valley") clearly describe a montainous scene.


# %% [markdown]
# ### Q2

# %%
for label in LABELS:
    img, img_name = load_sample_image(label)
    preds = get_top5_predictions(model, preprocess, img, device, imagenet_classes)[:3]
    print(f"\n[{label}]  {img_name}")
    for class_name, prob in preds:
        print(f"  {class_name:30s}  {prob:.4f}")

# The model seems most confident on classes like "mountain" or "sea", where the visual features are clear
# It is less confident on classes like "street" or "buildings"
# The pattern is that the model performs better on natural scenes and worse on mixed environments.

# %% [markdown]
# ### Q3

# %%
img, _ = load_sample_image("forest")
input_tensor = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(input_tensor)

probs = torch.nn.functional.softmax(logits[0], dim=0)

print(f"Logit  range: min={logits.min():.2f}, max={logits.max():.2f}")
print(f"Prob   range: min={probs.min():.6f}, max={probs.max():.4f}")
print(f"Probs sum to: {probs.sum():.6f}")
print(f"Top prediction: {imagenet_classes[probs.argmax().item()]}  ({probs.max():.4f})")

# Neural networks output logits because they are easier and more stable for computation during training.
# Logits are raw scores that are later converted into probabilities using softmax.

# In a production pipeline, I use probabilities, because they are easier to interpret 
# and allow us to filter predictions based on confidence.

# %% [markdown]
# ### Q4

# %%
labels = [p[0] for p in preds]
probs = [p[1] for p in preds]

fig, axes = plt.subplots(1, 2, figsize=(12,5))

axes[0].imshow(img)
axes[0].set_title(img_name)
axes[0].axis("off")

axes[1].barh(labels[::-1], probs[::-1])
axes[1].set_title("Top-5 Predictions")
axes[1].set_xlim(0,1)

plt.tight_layout()

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/warmup_inference_viz.png")

plt.show()

# This visualization is helpful for non-technical people 
# because it shows the image and how sure the model is about its predictions.

# In a real system, we can mark low confidence predictions ( for example, if top-1 probability is less than 0.8)
# so a person can check them.

# A good threshold for automatic decision is usually 0.7 to 0.9. If threshold is higher, it is more safe.



