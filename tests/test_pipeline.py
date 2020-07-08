from src.pipeline import CsvFilePipeline
from src.config import feature_names, label_name, categorical_feature_names
import os
import pandas as pd


package_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_pipeline = CsvFilePipeline()
identity_dir = os.path.join(
    package_dir,
    "resources",
    "testing_identity.csv")
transaction_dir = os.path.join(
    package_dir,
    "resources",
    "testing_transaction.csv")
test_raw_data = {
    "transaction": pd.read_csv(transaction_dir),
    "identity": pd.read_csv(identity_dir)}


class TestCsvFilePipeline:
    def test_query(self):
        res = csv_file_pipeline.query(
            identity_dir=identity_dir,
            transaction_dir=transaction_dir)
        assert len(res["identity"]) == 300
        assert len(res["transaction"]) == 300

    def test_parse_data(self):

        res_x, res_y = csv_file_pipeline.parse_data(test_raw_data)
        for feature_name in feature_names:
            assert feature_name in res_x.columns
        assert label_name[0] in res_y.columns
        for element in res_x["ProductCD"]:
            assert element in [0, 1, 2, 3, 4]

    def test_get_train_data(self):
        _, _ = csv_file_pipeline.parse_data(test_raw_data)
        train_data = csv_file_pipeline.get_train_data()

        for feature_name in  train_data.categorical_feature:
            assert feature_name in categorical_feature_names
        assert len(train_data.label) == 210

    def test_get_val_data(self):
        _, _ = csv_file_pipeline.parse_data(test_raw_data)
        val_data = csv_file_pipeline.get_val_data()

        for feature_name in  val_data.categorical_feature:
            assert feature_name in categorical_feature_names
        assert len(val_data.label) == 45

    def test_get_test_data(self):
        _, _ = csv_file_pipeline.parse_data(test_raw_data)
        test_data = csv_file_pipeline.get_test_data()

        for feature_name in  test_data.categorical_feature:
            assert feature_name in categorical_feature_names
        assert len(test_data.label) == 45