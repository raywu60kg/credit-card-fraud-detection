# lightgbm-project-demo
We are using the kaggle competition [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) as our dataset. We also want this project to be like a project to solve real world problem and it is able to put into production. 
Therefore, we use postgresSQL as our database and FastAPI as our API framework to simulate the real world scenario. However, a real world problem will also facing **Domain Adaptation Problems** 

## Requirement
- docker
- docker-compose
- make 
- kaggle.json 

## dev mode
## Requirement
- pipenv 
```bash
pipenv install --dev
```
## Note
```bash
jupyter contrib nbextension install --sys-prefix
```
to enable jupyter notebook extension