

from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import AutoProcessor, ViTForImageClassification
from PIL import Image, UnidentifiedImageError
import torch
import io

# Initialize FastAPI app
app = FastAPI()

# Load the model once (to improve performance)
model_name = "google/vit-base-patch16-224"
processor = AutoProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

@app.get("/")
async def home():
    return {"message": "FastAPI Animal Classifier is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Convert to RGB to avoid issues

        # Preprocess image
        inputs = processor(images=image, return_tensors="pt")

        # Model prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = logits.argmax(-1).item()

        # Get class label (handle unknown labels gracefully)
        animal_name = model.config.id2label.get(predicted_label, f"Class {predicted_label}")

        return {"Predicted Animal": animal_name}

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


# to run animal classifier with respect to fastapi: [uvicorn Mainapp:app --reload  (or)  uvicorn Mainapp:app --host 0.0.0.0 --port 8000 --reload ]
# And run Aniapp.py file also:  [streamlit run Aniapp.py ]
