import os
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as T

CLASSES = ["bharat", "hp", "indane", "unknown"]

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CONFIDENCE_THRESHOLD = 0.70

# ── Attention architecture ────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.max(dim=1, keepdim=True).values
        pooled   = avg_pool + max_pool
        att_map  = self.sigmoid(self.conv(pooled))
        return x * att_map

class LPGClassifierAttention(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        backbone        = models.efficientnet_b2(weights=None)
        self.features   = backbone.features
        self.avgpool    = backbone.avgpool
        feat_dim        = backbone.classifier[1].in_features
        self.attention  = SpatialAttention()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat_maps = self.features(x)
        attended  = self.attention(feat_maps)
        pooled    = self.avgpool(attended)
        flat      = torch.flatten(pooled, 1)
        return self.classifier(flat)

# ── Load models ───────────────────────────────────────────────────────────
def load_models(detector_path=None, classifier_path=None):
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    if not os.path.exists(base):
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if detector_path is None:
        detector_path   = os.path.join(base, "yolov11x_lpg_v1_best.pt")
    if classifier_path is None:
        classifier_path = os.path.join(base, "classifier_best_v6_attention.pth")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector   = YOLO(detector_path)

    classifier = LPGClassifierAttention(num_classes=4)
    checkpoint = torch.load(classifier_path, map_location=device)
    if "model_state_dict" in checkpoint:
        classifier.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded v6 attention — best val acc: {checkpoint.get('best_val_acc', 'N/A'):.1f}%")
    else:
        classifier.load_state_dict(checkpoint)

    classifier.eval().to(device)
    return detector, classifier, device

# ── Inference ─────────────────────────────────────────────────────────────
def predict(image: Image.Image, detector, classifier, device, conf=0.45):
    image = image.convert("RGB")
    image.save("temp.jpg")

    results = detector("temp.jpg", conf=conf, verbose=False)
    boxes   = results[0].boxes
    n       = len(boxes)

    if n == 0:
        return {"status": "no_cylinder", "brand": None, "confidence": None,
                "probabilities": None, "crop": None}
    if n > 1:
        return {"status": "multi_cylinder", "brand": None, "confidence": None,
                "probabilities": None, "crop": None}

    box  = boxes[0].xyxy[0].cpu().numpy()
    crop = image.crop((
        max(0, int(box[0])),
        max(0, int(box[1])),
        min(image.width,  int(box[2])),
        min(image.height, int(box[3]))
    ))

    tensor = TRANSFORM(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        output = classifier(tensor)
        probs  = torch.softmax(output, dim=1)[0]

    brand_idx  = probs.argmax().item()
    confidence = probs[brand_idx].item()

    if confidence < CONFIDENCE_THRESHOLD:
        brand = "unknown"
    else:
        brand = CLASSES[brand_idx]

    return {
        "status":        "ok",
        "brand":         brand,
        "confidence":    round(confidence * 100, 1),
        "probabilities": {CLASSES[i]: round(probs[i].item() * 100, 1) for i in range(4)},
        "crop":          crop
    }