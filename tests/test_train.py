from src.train import TrainLightGbmModel
import lightgbm as lgb
import os
import pandas as pd

package_dir = os.path.dirname(os.path.abspath(__file__))
test_params = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "verbose": 1,
    "scale_pos_weight": 1,
    "num_iterations": 1
}
train_data_dir = os.path.join(package_dir, "resources", "train_data.bin")
val_data_dir = os.path.join(package_dir, "resources", "val_data.bin")
test_data_x_dir = os.path.join(package_dir, "resources", "test_data_x.csv")
test_data_y_dir = os.path.join(package_dir, "resources", "test_data_y.csv")
train_data = lgb.Dataset(train_data_dir)
val_data = lgb.Dataset(val_data_dir, reference=train_data)
test_data_x = pd.read_csv(test_data_x_dir)
test_data_y = pd.read_csv(test_data_y_dir)
train_light_gbm_model = TrainLightGbmModel()


class TestLightGbmModel:
    def test_simple_train(self):
        model = train_light_gbm_model.simple_train(
            params=test_params,
            train_data=train_data, 
            val_data=val_data)
        print(model.best_score["valid_0"])
        predictions = model.predict(test_data_x)
        assert len(model.best_score["valid_0"]) == 3
        print(predictions)
        assert len(predictions) == len(test_data_x)
