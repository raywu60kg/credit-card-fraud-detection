import lightgbm as lgb
from src.eval import eval_average_precision
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

    def get_best_model(self):
        return 0

    def save_model(self):
        return 0

    def predict(self):
        return 0
