from ray import tune

def LightGBMCallback(env):
    """Assumes that `valid_0` is the target validation score."""
    _, metric, score, _ = env.evaluation_result_list[0]
    tune.report(**{metric: score})