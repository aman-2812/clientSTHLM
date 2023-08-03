from fastapi import FastAPI
import uvicorn
import pickle
from fastapi import FastAPI, HTTPException
import base64
from pydantic import BaseModel
from logger_config import logger

import client_sthlm

app = FastAPI()

bucket_name = 'sthlm-5g-nw'
object_name = 'Traffic_Train_Data.csv'
local_file_path = 'Traffic_Train_Data.csv'
client_sthlm.download_file_from_s3(bucket_name, object_name, local_file_path)

class WeightsData(BaseModel):
    weights: str

@app.get("/")
async def root():
    return {"message": "Hello from STHLM server"}

@app.get("/download_dataset")
async def download_dataset():
    logger.info("Api hit for download")
    response = client_sthlm.download_file_from_s3(bucket_name, object_name, local_file_path)
    return {"message": f"Dataset download result '{response}'!"}

@app.post("/local_training")
async def test(weights_data: WeightsData):
    base64_encoded_weights = weights_data.weights
    # Convert the base64-encoded string back to bytes.
    serialized_weights = base64.b64decode(base64_encoded_weights)
    # Deserialize the weights using Pickle.
    try:
        global_weights = pickle.loads(serialized_weights)
    except pickle.UnpicklingError as e:
        raise HTTPException(status_code=400, detail="Invalid weights data")
    result = client_sthlm.local_training(global_weights)
    logger.info(f"Returning result for {result[0]}")
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8020)