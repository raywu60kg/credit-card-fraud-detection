from src.train import TrainLightGbmModel
from src.pipeline import CsvFilePipeline
from src.callback import LightGBMCallback
import lightgbm as lgb
import os
import pandas as pd
from ray import tune
import json
from src.eval import eval_average_precision
from tests.resources.test_config import (
    train_data_dir,
    val_data_dir,
    test_data_x_dir,
    test_data_y_dir,
    test_identity_dir,
    test_transaction_dir)

test_params = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "verbose": 1,
    "scale_pos_weight": 1,
    "num_iterations": 1
}


train_data = lgb.Dataset(train_data_dir)
val_data = lgb.Dataset(val_data_dir, reference=train_data)
test_data_x = pd.read_csv(test_data_x_dir)
test_data_y = pd.read_csv(test_data_y_dir)
train_light_gbm_model = TrainLightGbmModel()
test_hyperparams_space = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    # "metric": "binary_error",
    "verbose": 1,
    "num_iterations": 1,
    "num_leaves": tune.randint(10, 1000),
    "learning_rate": tune.loguniform(1e-8, 1e-1),
    "identity_dir": test_identity_dir,
    "transaction_dir": test_transaction_dir
}
test_data_json = {
    "TransactionAmt": 261.95,
    "ProductCD": 4,
    "card1": 16982,
    "C1": 2.0,
    "C2": 2.0,
    "C3": 0.0,
    "C4": 0.0,
    "C5": 2.0,
    "C6": 3.0,
    "C7": 0.0,
    "C8": 0.0,
    "C9": 2.0,
    "C10": 0.0,
    "C11": 2.0,
    "C12": 0.0,
    "C13": 3.0,
    "C14": 2.0
}


class TestLightGbmModel:
    def test_simple_train(self):
        model = train_light_gbm_model.simple_train(
            params=test_params,
            train_data=train_data,
            val_data=val_data)
        print(model.best_score["valid_0"])
        predictions = model.predict(test_data_x)
        for k, v in model.best_score["valid_0"].items():
            print(k, v)
        assert len(model.best_score["valid_0"]) == 3
        assert len(predictions) == len(test_data_x)

    def test_get_best_model(self):

        best_model = train_light_gbm_model.get_best_model(
            hyperparams_space=test_hyperparams_space, num_samples=2)
        predictions = best_model.predict(test_data_x)
        assert len(best_model.best_score["valid_0"]) == 3
        assert len(predictions) == len(test_data_x)

    def test_save_model(self):
        model = train_light_gbm_model.simple_train(
            params=test_params,
            train_data=train_data,
            val_data=val_data)
        res = train_light_gbm_model.save_model(
            model=model,
            model_name = "test",
            save_model_dir="tests/",
            test_data_x=test_data_x,
            test_data_y=test_data_y)
        for key in ["model_name", "log_loss", "auc", "average_precision"]:
            assert key in res.keys()
        os.remove("tests/model.txt")
        os.remove("tests/model_metrics.json")

    def test_predict(self):
        model = train_light_gbm_model.simple_train(
            params=test_params,
            train_data=train_data,
            val_data=val_data)
        prediction = train_light_gbm_model.predict(
            model=model,
            data=test_data_json)
        print(prediction)
        assert prediction >= 0 and prediction <= 1

