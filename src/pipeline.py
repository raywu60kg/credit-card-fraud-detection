import psycopg2
import lightgbm as lgb


class Pipeline:
    def query_db(self):
        raise NotImplementedError

    def get_data(self):
        raise NotImplementedError
    


class PostgreSqlPipeline(Pipeline):
    def __init__(self):
        self.config = 1

    def query_db(self):
        return 0
    
    def get_data(self):
        # train_data = lgb.Dataset(data, label=label, feature_name=['c1', 'c2', 'c3'], categorical_feature=['c3'])
        return 0