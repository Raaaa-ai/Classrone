import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm
from torchvision.transforms import Resize

MODEL_NAME = "microsoft/resnet-50"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_SAMPLES = 1000

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

dataset = load_dataset("ylecun/mnist", split="test")
if NUM_SAMPLES:
    dataset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))

# 显式定义，ResNet‑50要求输入224×224
resize_transform = Resize((224, 224))

def collate_fn(batch):
    images = [resize_transform(item["image"].convert("RGB")) for item in batch]
    labels = [item["label"] for item in batch]
    inputs = processor(images=images, return_tensors="pt")
    return inputs["pixel_values"], torch.tensor(labels)


loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

correct = 0
total = 0
with torch.no_grad():
    for pixel_values, labels in tqdm(loader, desc="Inference"):
        pixel_values = pixel_values.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(pixel_values).logits
        preds = logits[:, :10].argmax(-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Accuracy on MNIST (ResNet-50, no finetune): {accuracy:.4f} ({correct}/{total})")
