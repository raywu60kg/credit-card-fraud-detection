# Dev mode
I use pipenv and pyenv for developing. If you never use them, you can check [here](https://medium.com/wu-hao-hsiang/how-to-set-up-python-working-environment-233a8a894c0a) to have basic understand. Anyway, you can also use conda or venv and install package from requirements.txt

## My dev setup
```
pipenv local 3.6.9
pipenv install --dev
pipenv shell
```

## Jupyter notebook
Initial the jupyter extension if this is the firt time
```
jupyter contrib nbextension install --sys-prefix
```

Activate jupyter notebook
```
jupyter notebook
```

## Test
```
pytest
```

## Run dev app
```
make run-app-dev
```

