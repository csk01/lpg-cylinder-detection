import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import traceback
import gradio as gr
from PIL import Image
from predict import load_models, predict

try:
    detector, classifier, device = load_models()
    print("Models loaded!")
except Exception as e:
    traceback.print_exc()
    raise

def run(image):
    if image is None:
        return None, "Please upload an image."
    img = Image.fromarray(image)
    result = predict(img, detector, classifier, device)
    if result["status"] == "no_cylinder":
        return None, "❌ No LPG cylinder detected in image."
    if result["status"] == "multi_cylinder":
        return None, "⚠️ Multiple cylinders detected.\n\nPlease present ONE cylinder.\n\nClassification skipped."
    prob_breakdown = "\n".join([
        f"  {brand}: {conf}%"
        for brand, conf in result["probabilities"].items()
    ])
    result_text = f"""✅ Single cylinder detected
🏷️  Brand: {result['brand']}, Confidence: {result['confidence']}%

All probabilities:
{prob_breakdown}"""
    return result["crop"], result_text

demo = gr.Interface(
    fn=run,
    inputs=gr.Image(label="📷 Upload or drag & drop a cylinder image here, then click Submit"),
    outputs=[
        gr.Image(label="🔍 Cropped cylinder sent to classifier"),
        gr.Textbox(label="📊 Result & Probability Breakdown", lines=12, max_lines=12),
    ],
    title="🛢️ lpg_identificationv1",
    description="""
## How to use
1. **Upload** a photo of an LPG cylinder using the image box on the left.
2. **Click Submit** — the detector will run automatically.
3. **View results** on the right — cropped cylinder region + brand, confidence, and full probability breakdown.

---
**Pipeline:** YOLOv11n (detection) → EfficientNetB0 (classification)
**Brands:** Indane | Bharat Gas | HP Gas | Unknown
    """,
)

if __name__ == "__main__":
    demo.launch(theme="soft", share=True)