from api.api_scheme import ModelInput, ModelOutput, HealthCheckOutput, MetricsOutput
from fastapi import BackgroundTasks
from fastapi import FastAPI
from src.config import num_samples, identity_dir, transaction_dir, hyperparams_space
from src.pipeline import CsvFilePipeline
from src.train import TrainLightGbmModel
import gc
import json
import lightgbm as lgb
import logging
import os
import uvicorn

tags_metadata = [
    {
        "name": "Model",
        "description": "Operations for the lightgbm model",
    },
    {
        "name": "Default",
        "description": "Basic function for API"
    }
]

app = FastAPI(
    title="Lightgbm Project Demo",
    description="This is machine learning project API",
    version="0.0.1",
    openapi_tags=tags_metadata)
package_dir = os.path.dirname(os.path.abspath(__file__))
train_light_gbm_model = TrainLightGbmModel()
global model
model = lgb.Booster(model_file=os.path.join(
    package_dir, "..", "models/model.txt"))


@app.get("/health", response_model=HealthCheckOutput, tags=["Default"])
def health_check():
    return {"health": "True"}


@app.get("/model/metrics", response_model=MetricsOutput, tags=["Model"])
def get_model_metrics():
    try:
        with open(
                os.path.join(
                    package_dir,
                    "..",
                    "models/model_metrics.json"), "r") as f:
            model_metrics = json.load(f)
    except Exception as e:
        logging.error("Error in geting model metrics: {}".format(e))
    return model_metrics


@ app.put("/model", tags=["Model"])
async def retrain_model(model_name: str, background_tasks: BackgroundTasks):
    def task_retrain_model():

        try:
            logging.info("Start Searching Best Model")
            best_model = train_light_gbm_model.get_best_model(
                hyperparams_space=hyperparams_space,
                num_samples=num_samples)
        except Exception as e:
            logging.error("Error in searching best model: {}".format(e))

        try:
            logging.info("Get the testing data")
            csv_file_pipeline = CsvFilePipeline()
            raw_data = csv_file_pipeline.query(
                identity_dir=hyperparams_space["identity_dir"],
                transaction_dir=hyperparams_space["transaction_dir"])

            gc.collect()
            data_x, data_y = csv_file_pipeline.parse_data(
                raw_data=raw_data)
            del data_x, data_y, raw_data
            gc.collect()
            
            test_data_x, test_data_y = csv_file_pipeline.get_test_data()
        except Exception as e:
            logging.error("Error in create data pipeline: {}".format(e))

        try:
            logging.info("Start Saving Model")
            res = train_light_gbm_model.save_model(
                model=best_model,
                model_name=model_name,
                save_model_dir=os.path.join(package_dir, "..", "models"),
                test_data_x=test_data_x,
                test_data_y=test_data_y)
        except Exception as e:
            logging.error("Error in saving model: {}".format(e))

        try:
            logging.info("Model Reload")
            model = lgb.Booster(model_file=os.path.join(
                package_dir, "..", "models/model.txt"))
        except Exception as e:
            logging.error("Error in reload model {}".format(e))

        logging.critical("Retrain Finish. Training result: {}".format(res))
    background_tasks.add_task(task_retrain_model)

    return {"train": "True"}


@ app.post("/model:predict", response_model=ModelOutput, tags=["Model"])
async def get_model_predict(data: ModelInput):
    try:

        prediction = train_light_gbm_model.predict(
            model=model, data=data.dict())
    except Exception as e:
        logging.error("Error in makeing prediction: {}".format(e))
        prediction = 0.5

    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run(app=app)
