from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from deepface import DeepFace
import os, datetime

app = FastAPI()

PHOTO_DIR = "photos"
UNKNOWN_DIR = "unknown"
EMP_DIR = "employees"

os.makedirs(PHOTO_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)
os.makedirs(EMP_DIR, exist_ok=True)

app.mount("/photos", StaticFiles(directory="photos"), name="photos")

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    contents = await file.read()

    filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    filepath = os.path.join(PHOTO_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(contents)

    try:
        result = DeepFace.find(
            img_path=filepath,
            db_path=EMP_DIR,
            model_name="ArcFace",
            enforce_detection=False
        )

        if len(result[0]) > 0:
            match = result[0].iloc[0]
            name = match["identity"].split("/")[-1].split(".")[0]

            return {
                "status": "success",
                "name": name
            }

    except:
        pass

    with open(os.path.join(UNKNOWN_DIR, filename), "wb") as f:
        f.write(contents)

    return {"status": "error"}
