from ray import tune


def LightGBMCallback(env):
    """Assumes that `valid_0` is the target validation score."""
    for _, metric, score, _ in env.evaluation_result_list:
        tune.report(**{metric: score})
