from fastapi import FastAPI
from fastapi import BackgroundTasks
import uvicorn
app = FastAPI()


@app.get("/health")
def read_root():
    return {"health": "True"}

@app.get("/metrics")
def get_model_metrics():
    return {"metrics": "True"}

@app.post("/train")
async def train_model():
    background_tasks.add_task()
    return {"train": "True"}

@app.post("/predict")
async def predict():
    background_tasks.add_task()
    return {"train": "True"}



if __name__ == "__main__":
    uvicorn.run(app=app)