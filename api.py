from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from seegull import Image, YOLOMultiLabel
import shutil
import os
import uuid
import torch
from PIL import Image as PILImage, ImageDraw

# ---- FastAPI app + Static Files ----
app = FastAPI()
app.mount("/static", StaticFiles(directory="."), name="static")  # <-- serve current directory

# ---- Load model ----
model = YOLOMultiLabel("./model/yolov8m_object_material_best.pt")

# ---- Function to draw bounding boxes ----
def draw_boxes_on_image(orig_img, boxes, labels):
    image = PILImage.fromarray(orig_img)
    draw = ImageDraw.Draw(image)

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), label, fill="red")

    return image

# ---- Endpoint ----
@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    temp_path = f"temp_{file_id}.jpg"

    # Save uploaded image
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    img = Image(path=temp_path)
    img.predict(model, conf=0.01, nms_threshold=0.05)

    # Fix detection tensor shape if needed
    if hasattr(img, "_yolo_result") and img._yolo_result is not None:
        data = img._yolo_result.data
        if isinstance(data, torch.Tensor) and data.shape[1] > 7:
            print("Fixing detection tensor with shape", data.shape)
            img._yolo_result.data = data[:, :7]  # Keep only valid columns

    results = img.yolo_result
    detections = []

    boxes = []
    labels = []
    print("results.boxes ...", results.boxes)

    if results.boxes is not None:
        for i, box in enumerate(results.boxes.data):
            x1, y1, x2, y2, conf1, cls1, conf2 = box[:7].tolist()

            # Get labels
            object_label = results.names[0].get(int(cls1), "Unknown")
            material_label = results.names[1].get(int(box[7])) if box.shape[0] > 7 else None

            detections.append({
                "label": object_label,
                "category": material_label,
                "confidence": round(conf1, 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            })

            boxes.append([x1, y1, x2, y2])
            labels.append(object_label)

    # Draw and save image
    if results.orig_img is not None:
        image_with_boxes = draw_boxes_on_image(results.orig_img, boxes, labels)
        output_path = f"detected_{file_id}.jpg"
        image_with_boxes.save(output_path)
        preview_url = f"/static/{output_path}"
    else:
        preview_url = None

    os.remove(temp_path)

    return {
        "message": "Detection successful. Awaiting confirmation.",
        "detections": detections,
        "preview_image_url": preview_url
    }
