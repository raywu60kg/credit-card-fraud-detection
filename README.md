# lightgbm-project-demo
![image](pictures/api-ui.png)
I used the kaggle competition [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) as our dataset.  In this project, I used **fastapi**, **lightgbm** and **ray tune** build an api server with retrainable model. Also, I want this project to be like a project real world project and it is able to put into production. Therefore I have some:

1. EDA or visualization in notebooks folder
2. A api Server in api folder
3. Dockerfile and docker-compose.yml for the deployment
4. Documents in docs
5. Some scripts in scripts folder

## Run this demo
I have two services app and script. App is the machine learning api open on 8000 port. Because I used fastapi for api server, you can check the document on **http://localhost:8000**. Script is send the request ask for predictions for kaggle testing data. After the "script" get all the responce it will write the file on **/tmp/submission.csv** (on host and container) but it gonna take a lot of time. Use **docker logs -f lightgbm-project-demo_script_1** to check the process.

### 1. Install Requirement
- docker
- docker-compose
- make 

### 2. Download data
#### option 1. 
setup kaggle api and use 
```
make init-data
```

#### option 2.
1. mkdir data
2. Download the data from kaggle [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)

3. put the ieee-fraud-detection.zip inside the data folder
4. upzip ieee-fraud-detection.zip

### 3. Build the image
```
make build
```

### 4. Start services
Start both two service
```
docker-compose up
```
or only start the app
```
docker-compose up app
```
## Documents
[How to set up the working environment for this project](docs/dev_mode.md)

[Api example](docs/api_example.md)

## TODO
### Database
Put the data inside the data to make it more like real world problem.

### Ray tune
Currently the ray tune will cost huge memory usage if setup **num_samples** greater than one. Might need to save the binary file instead of csv.

## Api 
Might need to run api in mutiple machine and put load balancer for big amount of  request.