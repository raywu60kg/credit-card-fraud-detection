# API example
After we run the app, we can use **http://localhost:8000/docs** to check 

## Health Check
```
curl localhost:8000/health
```

## Get Model Metrics
```
curl localhost:8000/model/metrics
```

## Retrain Model
```
curl -X PUT "http://localhost:8000/model?model_name=default" -H "accept: application/json"
```

## Get Model Prediction
```
curl -X POST "http://localhost:8000/model:predict" -H "accept: application/json" -H "Content-Type: application/json" -d '{"TransactionAmt":0,"ProductCD":0,"card1":0,"C1":0,"C2":0,"C3":0,"C4":0,"C5":0,"C6":0,"C7":0,"C8":0,"C9":0,"C10":0,"C11":0,"C12":0,"C13":0,"C14":0}'
```