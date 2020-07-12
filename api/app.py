from fastapi import FastAPI
from fastapi import BackgroundTasks
from src.train import TrainLightGbmModel
from src.pipeline import CsvFilePipeline
from src.config import num_samples, identity_dir, transaction_dir, hyperparams_space
from api.api_scheme import ModelInput, ModelOutput, HealthCheckOutput
import lightgbm as lgb
import uvicorn
import json
import os
import logging


app = FastAPI()
package_dir = os.path.dirname(os.path.abspath(__file__))
train_light_gbm_model = TrainLightGbmModel()
global model
model = lgb.Booster(model_file=os.path.join(
    package_dir, "..", "models/model.txt"))


@app.get("/health", response_model=HealthCheckOutput)
def health_check():
    return {"health": "True"}


@app.get("/model/metrics")
def get_model_metrics():
    with open(
            os.path.join(
                package_dir,
                "..",
                "models/model_metrics.json"), "r") as f:
        model_metrics = json.load(f)
    return model_metrics


@ app.put("/model")
async def retrain_model(model_name: str, background_tasks: BackgroundTasks):
    def task_retrain_model():

        csv_file_pipeline = CsvFilePipeline()
        raw_data = csv_file_pipeline.query(
            identity_dir=hyperparams_space["identity_dir"],
            transaction_dir=hyperparams_space["transaction_dir"])
        _, _ = csv_file_pipeline.parse_data(
            raw_data=raw_data)
        test_data_x, test_data_y = csv_file_pipeline.get_test_data()

        best_model = train_light_gbm_model.get_best_model(
            hyperparams_space=hyperparams_space,
            num_samples=num_samples)
        res = train_light_gbm_model.save_model(
            model=best_model,
            save_model_dir=os.path.join(package_dir, "..", "models"),
            test_data_x=test_data_x,
            test_data_y=test_data_y)
        logging.critical("Training result: {}".format(res))

    background_tasks.add_task(task_retrain_model)

    return {"train": "True"}


# @ app.post("/predict", response_model=ModelOutput)
@ app.post("/model:predict", response_model=ModelOutput)
async def get_model_predict(data: ModelInput):
    # async def get_model_predict(data):
    try:
        # model = lgb.Booster(model_file=os.path.join(
        #     package_dir, "..", "models/model.txt"))

        prediction = train_light_gbm_model.predict(
            model=model, data=data.dict())
    except Exception as e:
        logging.error("Error in makeing prediction: {}".format(e))
        prediction = 0.5

    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run(app=app)
