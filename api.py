from fastapi import FastAPI, UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
from io import BytesIO
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model1_path = "./dog_cat_version_01_accuracy_80.h5"
model = tf.keras.models.load_model(model1_path)


@app.get("/")
def hello():
    return {"message":"to /docs route for "}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    uploaded_file = await file.read()
    img = Image.open(BytesIO(uploaded_file))
    img = img.convert("L")
    img = img.resize((200,200))
    img = np.array(img)
    img = img/255.0
    img = img.reshape((1,200,200))
    prediction=float(model.predict(img)[0][0])
    print(prediction)
    return {"label":"dog" if prediction>=0.5 else "cat",
            "prediction":prediction}



