import lightgbm as lgb
from src.eval import eval_average_precision
from ray.tune.schedulers import ASHAScheduler
from src.call_back import LightGBMCallback
from src.eval import eval_average_precision
from src.pipeline import CsvFilePipeline
from ray import tune


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
        model = lgb.train(
            params,
            train_data,
            valid_sets=val_data,
            feval=eval_average_precision)
        return model

    def get_best_model(self, hyperparams_space, num_samples):
        # print(hyperparams_space["ident"])
        def tune_light_gbm(hyperparams_space):
            csv_file_pipeline = CsvFilePipeline()
            raw_data = csv_file_pipeline.query(
                identity_dir=hyperparams_space["identity_dir"],
                transaction_dir=hyperparams_space["transaction_dir"])
            csv_file_pipeline.parse_data(raw_data=raw_data)

            train_data = csv_file_pipeline.get_train_data()
            val_data = csv_file_pipeline.get_val_data()
            model = lgb.train(
                hyperparams_space,
                train_data,
                valid_sets=val_data,
                feval=eval_average_precision,
                verbose_eval=False,
                callbacks=[LightGBMCallback])
            # model = lgb.train(
            #     hyperparams_space,
            #     train_data,
            #     valid_sets=val_data,
            #     feval=eval_average_precision,
            #     verbose_eval=False)
            metrics = {}
            for k, v in model.best_score["valid_0"].items():
                metrics[k] = v

            tune.report(
                binary_logloss=metrics["binary_logloss"],
                auc=metrics["auc"],
                average_precision=metrics["average_precision"],
                done=True)
            tune.report(done=True)
        analysis = tune.run(
            tune_light_gbm,
            verbose=1,
            config=hyperparams_space,
            num_samples=num_samples,
            scheduler=ASHAScheduler(metric="binary_logloss", mode="min"))

        return analysis

    def save_model(self):
        return 0

    def predict(self):
        return 0
