param = {'num_leaves': 31, 'objective': 'binary'}
param['metric'] = 'auc'

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

