from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model import predict_image
import uvicorn

app = FastAPI(title="Brain Tumor Classification API")

# Allow CORS for local frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Brain Tumor Classification API. Use POST /predict to classify an image."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image bytes
    contents = await file.read()
    
    # Process image and get predictions
    result = predict_image(contents)
    
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
