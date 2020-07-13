import psycopg2
import lightgbm as lgb
import pandas as pd
from src.config import data_primary_key, feature_names, label_name, productCD_categories, categorical_feature_names
from sklearn.model_selection import train_test_split
import numpy as np

class Pipeline:
    def query(self):
        raise NotImplementedError

    def get_train_data(self):
        raise NotImplementedError

    def get_val_data(self):
        raise NotImplementedError

    def get_test_data(self):
        raise NotImplementedError


class PostgreSqlPipeline(Pipeline):
    """TODO"""

    def query(self):
        return 0

    def get_train_data(self):
        return 0

    def get_val_data(self):
        return 0

    def get_test_data(self):
        return 0


class CsvFilePipeline(Pipeline):

    def query(self, identity_dir, transaction_dir):
        """Query two data table from two different directory.
        Args:
            identity_dir: The directory of identity data.
            transaction_dir: The directory of transaction data.
        Returns:
            A dictionary with key identity and transaction 
            and corresponding data in  pandas DataFrame.  
        """

        raw_data = {}
        raw_data["identity"] = pd.read_csv(identity_dir)
        raw_data["transaction"] = pd.read_csv(transaction_dir)
        return raw_data

    def parse_data(self, raw_data):
        """Parse the raw data to the format that model need.
        Args:
            raw_data: A dictionary with key identity and transaction 
                and corresponding data in  pandas DataFrame (The return from query).
        Returns:
            data_x: A pandas DataFrame with training features.
            data_y: A pandas DataFrame with the label.
        """

        data = raw_data["transaction"]
        data_x = data[feature_names]
        data_y = data[label_name]
        data_x["ProductCD"] = list(
            map(lambda x: productCD_categories.index(x),  data_x["ProductCD"]))
        self.train_x, test_x, self.train_y, test_y = train_test_split(
            data_x, data_y, test_size=0.3, random_state=42)
        self.val_x, self.test_x, self.val_y, self.test_y = train_test_split(
            test_x, test_y, test_size=0.5, random_state=42)
        
        self.scale_pos_weight = (len(data_x)-sum(data_y[label_name[0]])/sum(data_y[label_name[0]]))
        
        self.d_train = lgb.Dataset(
            self.train_x,
            label=self.train_y,
            categorical_feature=categorical_feature_names, 
            free_raw_data=False)
        return data_x, data_y

    def get_train_data(self):
        """Get the lightgbm dataset for training"""
        
        return self.d_train

    def get_val_data(self):
        """Get the lightgbm dataset for validation"""

        return  lgb.Dataset(
            self.val_x,
            label=self.val_y,
            categorical_feature=categorical_feature_names, 
            free_raw_data=False,
            reference=self.d_train)

    def get_test_data(self):
        """Get the two pandas DataFrame 
        which are test_x and test_y for testing because
        lightgbm model take pandas DataFrame for predicting.
        Returns:
            test_x: A pandas DataFrame with training features.
            test_y: A pandas DataFrame with the label.
        """

        return  self.test_x, self.test_y
