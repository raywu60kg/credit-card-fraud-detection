from src.pipeline import CsvFilePipeline
from src.config import feature_names, label_name, categorical_feature_names
import os
import pandas as pd
from tests.resources.test_config import test_identity_dir, test_transaction_dir


csv_file_pipeline = CsvFilePipeline()
test_raw_data = {
    "transaction": pd.read_csv(test_transaction_dir),
    "identity": pd.read_csv(test_identity_dir)}


class TestCsvFilePipeline:
    def test_query(self):
        res = csv_file_pipeline.query(
            identity_dir=test_identity_dir,
            transaction_dir=test_transaction_dir)
        assert len(res["identity"]) == 300
        assert len(res["transaction"]) == 300

    def test_parse_data(self):

        res_x, res_y = csv_file_pipeline.parse_data(test_raw_data)
        for feature_name in feature_names:
            assert feature_name in res_x.columns
        assert label_name[0] in res_y.columns
        for element in res_x["ProductCD"]:
            assert element in [0, 1, 2, 3, 4]

        assert csv_file_pipeline.scale_pos_weight == 299

    def test_get_train_data(self):
        _, _ = csv_file_pipeline.parse_data(test_raw_data)
        train_data = csv_file_pipeline.get_train_data()

        for feature_name in train_data.categorical_feature:
            assert feature_name in categorical_feature_names
        assert len(train_data.label) == 210
        # train_data.save_binary('tests/resources/train_data.bin')

    def test_get_val_data(self):
        _, _ = csv_file_pipeline.parse_data(test_raw_data)
        val_data = csv_file_pipeline.get_val_data()

        for feature_name in val_data.categorical_feature:
            assert feature_name in categorical_feature_names
        assert len(val_data.label) == 45
        # val_data.save_binary('tests/resources/val_data.bin')

    def test_get_test_data(self):
        _, _ = csv_file_pipeline.parse_data(test_raw_data)
        test_data_x, test_data_y = csv_file_pipeline.get_test_data()

        assert len(test_data_x) == 45
        assert len(test_data_y) == 45
        # test_data_x.to_csv("tests/resources/test_data_x.csv", index=False)
        # test_data_y.to_csv("tests/resources/test_data_y.csv", index=False)
