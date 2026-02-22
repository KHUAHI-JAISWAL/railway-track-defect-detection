from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

#  add this line
from huggingface_hub import hf_hub_download

app = FastAPI()

#  CHANGE ONLY THIS PART (model loading)
model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/YOUR_MODEL_REPO",
    filename="railway_defect_cnn_model.h5"
)

model = load_model(model_path)
#  end change

labels = ["Non Defective", "Defective"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # check file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    image_bytes = await file.read()

    # open image safely
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # preprocess
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    pred = model.predict(img_array)
    prob = float(pred[0][0])   # probability of Defective

    # final decision + correct confidence
    if prob >= 0.5:
        label = labels[1]          # Defective
        confidence = prob
    else:
        label = labels[0]          # Non Defective
        confidence = 1 - prob

    return {
        "class": label,
        "confidence": float(confidence)
    }