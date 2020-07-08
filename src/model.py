from sklearn.model_selection import GridSearchCV
class Model:
    def create_model(self, hp):
        raise NotImplementedError

class LightGbmModel(Model):
    def create_model(self):
        return 0