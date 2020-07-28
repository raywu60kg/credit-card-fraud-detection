from ray import tune
from ray.tune.schedulers import ASHAScheduler
from src.callback import LightGBMCallback
from src.eval import eval_average_precision
from src.pipeline import CsvFilePipeline
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
import ray
import json
import lightgbm as lgb
import os
import pandas as pd
import logging
import gc


class TrainModel:

    def get_best_model(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def get_model_metrics(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class TrainLightGbmModel(TrainModel):

    def simple_train(self, params, train_data, val_data):
        """Train one lightgbm model
        Args:
            params: The parameter set for lightgbm model.
            train_data: lightgbm dataset for training.
            val_data: lightgbm dataset for validating model.
        Returns:
            model: Trained lightgbm model.
        """

        model = lgb.train(
            params,
            train_data,
            valid_sets=val_data,
            feval=eval_average_precision)
        return model

    def get_best_model(self, hyperparams_space, num_samples):
        """Use ray tune to train mutiple lightgbm model and
         return the best model in our hyperparameters search space.
        Args:
            hyperparams_space: The hyperparameters space for searching.
            num_samples: Number of trial for ray tune.
        Returns:
            best_model: The best trained lightgbm model in hyperparameter space.
        """

        # define the tuning function for ray tune
        def tune_light_gbm(hyperparams_space):

            # init the data pipeline
            csv_file_pipeline = CsvFilePipeline()
            raw_data = csv_file_pipeline.query(
                identity_dir=hyperparams_space["identity_dir"],
                transaction_dir=hyperparams_space["transaction_dir"])

            data_x, data_y = csv_file_pipeline.parse_data(raw_data=raw_data)
            del data_x, data_y, raw_data
            gc.collect()

            train_data = csv_file_pipeline.get_train_data()
            val_data = csv_file_pipeline.get_val_data()
            del csv_file_pipeline
            gc.collect()

            # Train the model
            model = lgb.train(
                hyperparams_space,
                train_data,
                valid_sets=val_data,
                feval=eval_average_precision,
                verbose_eval=False,
                callbacks=[LightGBMCallback])

            # Report metrics
            metrics = {}
            for k, v in model.best_score["valid_0"].items():
                metrics[k] = v

            tune.report(
                binary_logloss=metrics["binary_logloss"],
                auc=metrics["auc"],
                average_precision=metrics["average_precision"],
                done=True)
            tune.report(done=True)

        # Start tuning the hyperparams
        ray.init(
            memory=2000 * 1024 * 1024,
            object_store_memory=200 * 1024 * 1024,
            driver_object_store_memory=100 * 1024 * 1024)
        analysis = tune.run(
            tune_light_gbm,
            verbose=1,
            config=hyperparams_space,
            num_samples=num_samples,
            scheduler=ASHAScheduler(metric="binary_logloss", mode="min"))

        # Get the best params setup
        log_dir = analysis.get_best_logdir('average_precision', mode="max")
        with open(os.path.join(log_dir, "params.json")) as f:
            best_params = json.load(f)

        # Get the model using best params
        csv_file_pipeline = CsvFilePipeline()
        raw_data = csv_file_pipeline.query(
            identity_dir=best_params["identity_dir"],
            transaction_dir=best_params["transaction_dir"])
        
        data_x, data_y = csv_file_pipeline.parse_data(raw_data=raw_data)
        del data_x, data_y, raw_data
        gc.collect()

        train_data = csv_file_pipeline.get_train_data()
        val_data = csv_file_pipeline.get_val_data()
        del csv_file_pipeline
        gc.collect()

        best_model = lgb.train(
            best_params,
            train_data,
            valid_sets=val_data,
            feval=eval_average_precision,
            verbose_eval=False)
        return best_model

    def save_model(
            self,
            model,
            model_name,
            save_model_dir,
            test_data_x,
            test_data_y):
        """Save the model and the metrics
        Args:
            model: Lightgbm model.
            model_name: The model name for the model.
            save_model_dir: The directory for model saving.
            test_data_x: Pandas DataFrame with training features for calculating metrics.
            test_data_y: Pandas DataFrame with label for calculating metrics.
        Returns:
            metrics: A dictionary with model metrics.
        """

        y_pred = model.predict(test_data_x)
        metrics = {
            "model_name": model_name,
            "log_loss": log_loss(y_true=test_data_y, y_pred=y_pred),
            "auc": roc_auc_score(y_true=test_data_y, y_score=y_pred),
            "average_precision": average_precision_score(
                y_true=test_data_y, y_score=y_pred)
        }
        with open(os.path.join(save_model_dir, "model_metrics.json"), "w") as f:
            json.dump(metrics, f)

        model.save_model(os.path.join(save_model_dir, 'model.txt'))
        return metrics

    def predict(self, model, data):
        """Calculate the prediction with given data
        Args:
            model: lightgbm model.
            data: A dictionary with key and value for all the training features
        Returns:
            prediction: A float number which is the estimate of the probability 
                for the postive label.
        """
        
        prediction = model.predict(pd.DataFrame(data, index=[0]))[0]
        return prediction
