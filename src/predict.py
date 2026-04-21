import os
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as T

CLASSES = ["Bharat Gas", "HP Gas", "Indane", "Unknown"]

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_models(detector_path=None, classifier_path=None):
    base = os.path.join(os.path.dirname(__file__), "..", "models")
    if detector_path is None:
        detector_path = os.path.join(base, "best.pt")
    if classifier_path is None:
        classifier_path = os.path.join(base, "classifier_best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLO(detector_path)

    classifier = models.efficientnet_b0(weights=None)
    classifier.classifier[1] = nn.Linear(
        classifier.classifier[1].in_features, 4
    )
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval().to(device)

    return detector, classifier, device

def predict(image: Image.Image, detector, classifier, device, conf=0.6):
    image = image.convert("RGB")
    image.save("temp.jpg")

    results = detector("temp.jpg", conf=conf)
    boxes = results[0].boxes
    n = len(boxes)

    if n == 0:
        return {"status": "no_cylinder", "brand": None, "confidence": None,
                "probabilities": None, "crop": None}
    if n > 1:
        return {"status": "multi_cylinder", "brand": None, "confidence": None,
                "probabilities": None, "crop": None}

    box = boxes[0].xyxy[0].cpu().numpy()
    crop = image.crop((
        max(0, int(box[0])),
        max(0, int(box[1])),
        min(image.width, int(box[2])),
        min(image.height, int(box[3]))
    ))

    tensor = TRANSFORM(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        output = classifier(tensor)
        probs = torch.softmax(output, dim=1)[0]

    brand_idx = probs.argmax().item()

    return {
        "status": "ok",
        "brand": CLASSES[brand_idx],
        "confidence": round(probs[brand_idx].item() * 100, 1),
        "probabilities": {CLASSES[i]: round(probs[i].item() * 100, 1) for i in range(4)},
        "crop": crop
    }