import os
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as T

# Fallback class order, used only when a checkpoint has no "classes" key.
# NOTE: the models do NOT all share this list — the side classifier
# (classifier_best_v6_attention.pth) is 4-class and includes "unknown", while the
# bottom-ring and top-view classifiers are 3-class. Each model's real class list is
# read from its own checkpoint at load time and carried alongside the model, so never
# index a model's logits with this constant.
CLASSES = ["bharat", "hp", "indane"]

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CONFIDENCE_THRESHOLD = 0.50   # below this, final composed brand is overridden to "unknown"
MIN_CROP_SIZE        = 30     # skip a view's classifier if its crop is smaller than this

# ── Attention architecture (side classifier) ───────────────────────────────
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
    def __init__(self, num_classes=3):
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

# ── Load all ensemble models ───────────────────────────────────────────────
def load_ensemble_models():
    # HF Spaces runs from /app/ — models are at /app/models/
    # Local: src/../models/ works fine
    # HF: /app/../models/ doesn't exist, use /app/models/ instead
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    if not os.path.exists(base):
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = YOLO(os.path.join(base, "yolov11x_lpg_v1_best.pt"))

    # Side classifier — EfficientNetB2 + spatial attention
    side_path       = os.path.join(base, "classifier_best_v6_attention.pth")
    side_ckpt       = torch.load(side_path, map_location=device)
    side_classes    = list(side_ckpt.get("classes", CLASSES))
    side_clf        = LPGClassifierAttention(num_classes=len(side_classes))
    side_clf.load_state_dict(side_ckpt["model_state_dict"])
    side_clf.classes = side_classes
    side_clf.eval().to(device)
    print(f"Loaded side classifier — best val acc: {side_ckpt.get('best_val_acc', 'N/A')}%")

    # Bottom ring classifier — EfficientNetB0
    bottom_path       = os.path.join(base, "classifier_bottomring_v2.pth")
    bottom_ckpt       = torch.load(bottom_path, map_location=device)
    bottom_classes    = list(bottom_ckpt.get("classes", CLASSES))
    bottom_clf  = models.efficientnet_b0(weights=None)
    bottom_clf.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(bottom_clf.classifier[1].in_features, len(bottom_classes))
    )
    bottom_clf.load_state_dict(bottom_ckpt["model_state_dict"])
    bottom_clf.classes = bottom_classes
    bottom_clf.eval().to(device)
    print(f"Loaded bottom ring classifier — best val acc: {bottom_ckpt.get('best_val_acc', 'N/A')}%")

    # Top view classifier — EfficientNetB0
    top_path       = os.path.join(base, "classifier_topview_v1.pth")
    top_ckpt       = torch.load(top_path, map_location=device)
    top_classes    = list(top_ckpt.get("classes", CLASSES))
    top_clf  = models.efficientnet_b0(weights=None)
    top_clf.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(top_clf.classifier[1].in_features, len(top_classes))
    )
    top_clf.load_state_dict(top_ckpt["model_state_dict"])
    top_clf.classes = top_classes
    top_clf.eval().to(device)
    print(f"Loaded top view classifier — best val acc: {top_ckpt.get('best_val_acc', 'N/A')}%")

    return detector, side_clf, bottom_clf, top_clf, device

# ── Composer ────────────────────────────────────────────────────────────────
def compose(predictions):
    """
    predictions: {view: (brand, confidence_0_to_100)} for views present among "side", "bottom", "top".
    Weighted vote across views; low-confidence views are dropped; missing top view
    causes side+bottom weights to be renormalised back up to 1.0.

    Note: the side classifier is 4-class, so "unknown" can arrive here as a brand and is
    treated as an ordinary vote — a confident side-view "unknown" can therefore win the
    vote and be returned as the final label. The bottom/top classifiers are 3-class and
    can never vote "unknown".
    """
    weights = {"side": 0.40, "bottom": 0.45, "top": 0.15}

    # Drop views whose classifier wasn't confident enough to trust its vote.
    usable = {view: (brand, conf) for view, (brand, conf) in predictions.items() if conf >= 60.0}

    # Renormalise weights over only the views actually present (e.g. no top camera).
    active_weight_sum = sum(weights[view] for view in usable)
    if active_weight_sum == 0:
        return ("unknown", 0.0)

    scores = {}
    for view, (brand, conf) in usable.items():
        w = weights[view] / active_weight_sum
        scores[brand] = scores.get(brand, 0.0) + (conf / 100.0) * w

    final_brand = max(scores, key=scores.get)
    final_conf  = round(scores[final_brand] * 100, 1)

    if final_conf < 25.0:
        return ("unknown", final_conf)
    return (final_brand, final_conf)

# ── Helpers ─────────────────────────────────────────────────────────────────
def _classify_crop(crop, classifier, device):
    # Each classifier carries its own class list (set in load_ensemble_models) because
    # the side model is 4-class and the bottom/top models are 3-class. Indexing all of
    # them with a shared 3-item constant would raise IndexError on an "unknown"
    # prediction and silently hide the unknown probability mass.
    classes = getattr(classifier, "classes", CLASSES)
    tensor  = TRANSFORM(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        output = classifier(tensor)
        probs  = torch.softmax(output, dim=1)[0]
    idx  = probs.argmax().item()
    conf = probs[idx].item()
    return (
        classes[idx],
        round(conf * 100, 1),
        {classes[i]: round(probs[i].item() * 100, 1) for i in range(len(classes))},
    )

def _crop_from_box(image, box):
    return image.crop((
        max(0, int(box[0])),
        max(0, int(box[1])),
        min(image.width,  int(box[2])),
        min(image.height, int(box[3]))
    ))

# ── Main ensemble inference ────────────────────────────────────────────────
def predict_ensemble(image, detector, side_clf, bottom_clf, top_clf,
                      device, side_conf_thresh=0.45, top_image=None):
    image = image.convert("RGB")
    image.save("temp.jpg")

    # Step 1 — detect cylinder in the side image.
    results = detector("temp.jpg", conf=side_conf_thresh, verbose=False)
    boxes   = results[0].boxes
    n       = len(boxes)

    if n == 0:
        return {"status": "no_cylinder", "brand": None, "confidence": None,
                "full_crop": None, "side_crop": None, "bottom_crop": None, "top_crop": None,
                "side_pred": None, "side_conf": None,
                "bottom_pred": None, "bottom_conf": None,
                "top_pred": None, "top_conf": None,
                "probabilities": None}
    if n > 1:
        return {"status": "multi_cylinder", "brand": None, "confidence": None,
                "full_crop": None, "side_crop": None, "bottom_crop": None, "top_crop": None,
                "side_pred": None, "side_conf": None,
                "bottom_pred": None, "bottom_conf": None,
                "top_pred": None, "top_conf": None,
                "probabilities": None}

    box       = boxes[0].xyxy[0].cpu().numpy()
    full_crop = _crop_from_box(image, box)

    # Step 2 — slice the bbox into side (middle 60%) and bottom ring (bottom 25%) regions.
    x1, y1, x2, y2 = box
    box_h = y2 - y1

    side_box   = (x1, y1 + 0.25 * box_h, x2, y1 + 0.85 * box_h)
    bottom_box = (x1, y1 + 0.75 * box_h, x2, y2)

    side_crop   = _crop_from_box(image, side_box)
    bottom_crop = _crop_from_box(image, bottom_box)

    predictions = {}

    # Step 3 — side classifier (primary signal).
    side_pred = side_conf = None
    probabilities = None
    if side_crop.width >= MIN_CROP_SIZE and side_crop.height >= MIN_CROP_SIZE:
        side_pred, side_conf, probabilities = _classify_crop(side_crop, side_clf, device)
        predictions["side"] = (side_pred, side_conf)

    # Step 4 — bottom ring classifier.
    bottom_pred = bottom_conf = None
    if bottom_crop.width >= MIN_CROP_SIZE and bottom_crop.height >= MIN_CROP_SIZE:
        bottom_pred, bottom_conf, _ = _classify_crop(bottom_crop, bottom_clf, device)
        predictions["bottom"] = (bottom_pred, bottom_conf)

    # Step 5 — optional top view classifier.
    top_pred = top_conf = None
    top_crop = None
    if top_image is not None:
        top_image = top_image.convert("RGB")
        top_image.save("temp_top.jpg")
        top_results = detector("temp_top.jpg", conf=side_conf_thresh, verbose=False)
        top_boxes   = top_results[0].boxes
        if len(top_boxes) == 1:
            top_box  = top_boxes[0].xyxy[0].cpu().numpy()
            top_crop = _crop_from_box(top_image, top_box)
            if top_crop.width >= MIN_CROP_SIZE and top_crop.height >= MIN_CROP_SIZE:
                top_pred, top_conf, _ = _classify_crop(top_crop, top_clf, device)
                predictions["top"] = (top_pred, top_conf)
            else:
                top_crop = None
        # If no single detection in the top image, skip the top view entirely.

    # Step 6 — combine per-view predictions into a final brand call.
    brand, confidence = compose(predictions)
    if confidence < CONFIDENCE_THRESHOLD * 100:
        brand = "unknown"

    return {
        "status":        "ok",
        "brand":         brand,
        "confidence":    confidence,
        "full_crop":     full_crop,
        "side_crop":     side_crop,
        "bottom_crop":   bottom_crop,
        "top_crop":      top_crop,
        "side_pred":     side_pred,
        "side_conf":     side_conf,
        "bottom_pred":   bottom_pred,
        "bottom_conf":   bottom_conf,
        "top_pred":      top_pred,
        "top_conf":      top_conf,
        "probabilities": probabilities,
    }
