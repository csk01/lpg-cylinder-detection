import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import traceback
import gradio as gr
from PIL import Image
from predict_ensemble import load_ensemble_models, predict_ensemble


try:
    ens_detector, side_clf, bottom_clf, top_clf, ens_device = load_ensemble_models()
    print("Ensemble models loaded!")
except Exception:
    traceback.print_exc()
    raise

def run(side_image, top_image):
    if side_image is None:
        return None, None, None, None, "Side camera image is required."

    side_img = Image.fromarray(side_image)
    top_img  = Image.fromarray(top_image) if top_image is not None else None

    result = predict_ensemble(
        side_img, ens_detector, side_clf, bottom_clf, top_clf, ens_device,
        top_image=top_img
    )

    if result["status"] == "no_cylinder":
        return None, None, None, None, "No LPG cylinder detected."

    if result["status"] == "multi_cylinder":
        return None, None, None, None, "Multiple cylinders detected. Please present one cylinder at a time."

    # Resize crops to avoid content length issues
    full_crop   = result["full_crop"]
    side_crop   = result["side_crop"]
    bottom_crop = result["bottom_crop"]
    top_crop    = result["top_crop"]

    full_crop.thumbnail((512, 512))
    side_crop.thumbnail((512, 512))
    bottom_crop.thumbnail((512, 512))
    if top_crop is not None:
        top_crop.thumbnail((512, 512))

    top_pred_text = f"{result['top_pred']} ({result['top_conf']}%)" if result["top_pred"] is not None else "N/A"

    result_text = f"""═══════════════════════════
Brand: {result['brand'].title()}
Confidence: {result['confidence']}%
═══════════════════════════
Side view:   {result['side_pred']} ({result['side_conf']}%)
Bottom ring: {result['bottom_pred']} ({result['bottom_conf']}%)
Top view:    {top_pred_text}"""

    return full_crop, side_crop, bottom_crop, top_crop, result_text

demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Image(sources=["webcam"],
                  streaming=True,
                  label="Side Camera (required)"),
        gr.Image(sources=["webcam", "upload"], 
                 streaming=False, 
                 label="Top Camera (leave empty if unavailable, streaming set to False now)"),
    ],
    outputs=[
        gr.Image(label="Detected cylinder"),
        gr.Image(label="Side region"),
        gr.Image(label="Bottom ring region"),
        gr.Image(label="Top view crop"),
        gr.Textbox(label="Result", lines=10, max_lines=10),
    ],
    title="LPG Cylinder Identification — Ensemble v1",
    description="Side camera required. Top camera optional — ensemble automatically adjusts weights.",
    theme="soft",
    flagging_mode="manual",
    flagging_options=["Wrong brand", "Not detected", "Multiple cylinders", "Other"],
    flagging_dir="flagged_predictions"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
