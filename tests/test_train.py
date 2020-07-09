from src.train import TrainLightGbmModel
from src.pipeline import CsvFilePipeline
from src.call_back import LightGBMCallback
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

        analysis = train_light_gbm_model.get_best_model(
            hyperparams_space=test_hyperparams_space, num_samples=2)
        print(analysis)
        # print(analysis.get_best_logdir("~/ray_results/tune_light_gbm"))
        print("++++++++")
        print([k for k in analysis.dataframe()])
        print(analysis.get_best_logdir('binary_logloss', mode="min"))
        print(analysis.get_best_logdir('average_precision', mode="max"))
        log_dir = analysis.get_best_logdir('average_precision', mode="max")
        with open(os.path.join(log_dir, "params.json")) as f:
            params = json.load(f)
        print(params)

        csv_file_pipeline = CsvFilePipeline()
        raw_data = csv_file_pipeline.query(
            identity_dir=params["identity_dir"],
            transaction_dir=params["transaction_dir"])
        csv_file_pipeline.parse_data(raw_data=raw_data)

        train_data = csv_file_pipeline.get_train_data()
        val_data = csv_file_pipeline.get_val_data()
        best_model = lgb.train(
            params,
            train_data,
            valid_sets=val_data,
            feval=eval_average_precision,
            # verbose_eval=False,
            callbacks=[LightGBMCallback])
        print("@@@@", best_model)
        assert analysis == 1
