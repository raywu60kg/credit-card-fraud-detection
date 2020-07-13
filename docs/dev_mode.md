# Dev mode
I used `pipenv` and `pyenv` for developing. If you have not used them before, you can check [here](https://medium.com/wu-hao-hsiang/how-to-set-up-python-working-environment-233a8a894c0a) to have some basic understanding. Also, you can use `conda` or `venv` and install package from requirements.txt

## My dev setup
```
pipenv local 3.6.9
pipenv install --dev
pipenv shell
```

## Jupyter notebook
Initialize the jupyter extension if this is the first time
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

