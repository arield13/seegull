from huggingface_hub import hf_hub_download
from seegull import YOLOMultiLabel

# Download model from Hugging Face
model_path = hf_hub_download(
    repo_id="BowerApp/bowie-yolov8-multihead-trash-detection",
    filename="yolov8m_object_material_best.pt"
)

# Load the model
model = YOLOMultiLabel(model_path)

print("✅ Model downloaded and loaded successfully!")
print(f"📍 Model path: {model_path}")
