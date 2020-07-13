from sklearn.metrics import average_precision_score

def eval_average_precision(y_pred, train_data):
    """Evaluate the average precision for the lightgbm model"""
    y_true = train_data.get_label()
    average_precision = average_precision_score(y_true, y_pred)
    return 'average_precision', average_precision, True