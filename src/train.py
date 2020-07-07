class TrainModel:

    def get_best_model(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError
    def get_model_metrics(self):
        raise NotImplementedError
    def predict(self):
        raise NotImplementedError

class TrainLightgbmModel(TrainModel):
    def get_best_model(self):
        return 0
    def save_model(self):
        return 0
    def predict(self):
        return 0

