from sklearn.metrics import average_precision_score

def eval_average_precision(y_pred, train_data):
    y_true = train_data.get_label()
    average_precision = average_precision_score(y_true, y_pred)
    return 'average precision', average_precision, True