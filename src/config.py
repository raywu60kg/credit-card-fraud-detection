import os
from ray import tune
package_dir = os.path.dirname(os.path.abspath(__file__))
identity_dir = os.path.join(package_dir, "..", "data/train_identity.csv")
transaction_dir = os.path.join(package_dir, "..", "data/train_transaction.csv")

data_primary_key = "TransactionID"
label_name = ["isFraud"]
feature_names = [
    'TransactionAmt',
    'ProductCD',
    'card1',
    'C1',
    'C2',
    'C3',
    'C4',
    'C5',
    'C6',
    'C7',
    'C8',
    'C9',
    'C10',
    'C11',
    'C12',
    'C13',
    'C14']
productCD_categories = ['C', 'H', 'R', 'S', 'W']
categorical_feature_names = ["card1", "ProductCD"]

num_samples = 20
hyperparams_space = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "verbose": 1,
    "num_threads": 1,
    "num_iterations": 100,
    "num_leaves": tune.randint(10, 1000),
    "learning_rate": tune.loguniform(1e-8, 1e-1),
    "identity_dir": identity_dir,
    "transaction_dir": transaction_dir
}
