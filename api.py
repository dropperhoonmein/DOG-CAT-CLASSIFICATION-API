from fastapi import FastAPI, UploadFile,File,Query
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
model1_path = "./dog_cat_version_01_accuracy_85.h5"
model2_path = "./dog_cat_version_02_feature_learning_accuracy_93.h5"
model3_path = "./dog_cat_version_03_fine_turning_accuracy_96.h5"

model = tf.keras.models.load_model(model1_path,compile=False)
model_path_list = [model1_path,model2_path,model3_path]
models = []
for model_path in model_path_list:
    models.append( tf.keras.models.load_model(model_path,compile=False))


@app.get("/")
def hello():
    return {"message":"to /docs route for "}

@app.post("/predict")
async def predict(file: UploadFile = File(...),model_no=Query(default=1,example=1,description="choose a model")):
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




    model_no = int(model_no)
    if model_no == 1:
        print(1)
        img = Image.open(BytesIO(uploaded_file))
        img = img.convert("L")
        img = img.resize((200,200))
        img = np.array(img)
        img = img/255.0
        img = img.reshape((1,200,200))
        prediction=float(models[model_no-1].predict(img)[0][0])
        print(prediction)
        return {"label":"dog" if prediction>=0.5 else "cat",
                "prediction":prediction}
    elif model_no ==2 or model_no==3:
        img = Image.open(BytesIO(uploaded_file))
        img = img.convert("RGB")
        img = img.resize((224,224))
        img = np.array(img)
        img = img/255.0
        img = img.reshape((1,224,224,3))
        prediction=float(models[model_no-1].predict(img)[0][0])
        print(prediction)
        return {"label":"dog" if prediction>=0.5 else "cat",

                "prediction":prediction}
