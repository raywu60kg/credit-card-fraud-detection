# lightgbm-project-demo
![image](pictures/api-ui.png)
For this project, I used the datasets from the  kaggle competition called [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data). The competition aims to improve fraud prevention system by building fraud detection models based on [Vesta Corporation's](https://trustvesta.com/) real-world e-commerce transactional data, which contains information from device type to product features.  My personal goal for this project is to not only explore the data and build models, but to also build an API server with retrainable model.  To achieve this goal, I used **fastapi**, **lightgbm** and **ray tune**. 

Also, I decided to develop this project to be the same as how data-related projects are developed in real-world scenarios, wherein the end goal of development is a project that is feasible for production. Therefore, I have put efforts on creating:

1. Exploratory Data Analysis (EDA) in the `notebooks/` folder;
2. An API Server inside the `api/` folder;
3. Files for deployment such as Dockerfile and docker-compose.yml;
4. Documentations in the `docs/` folder; and
5. Some necessary scripts in `scripts/` folder.

## How to run this demo
I have two services: `app` and `script`. The `app` service is a machine learning API that is open on port 8000. I used fastapi for the API server, so you can check it on **http://localhost:8000/docs** after you run the `app` service. The `script` sends the request for the predictions on new sets of data, such as the kaggle testing data. After the `script` get all the responses, files will be written on **/tmp/submission.csv** (on host and container), but this part can take a lot of time. It is suggested to use **docker logs -f lightgbm-project-demo_script_1** to check the progress of the process.

![image](pictures/docker-logs.png)

### 1. Install the requirements
- docker
- docker-compose
- make 

### 2. Download the datasets
#### Option 1. 
Setup kaggle API and use 
```
make init-data
```

#### Option 2.
1. Create a data folder: i.e. 
```
mkdir data
```
2. Download the data from kaggle [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data).

3. Put the ieee-fraud-detection.zip inside the `data/` folder.
4. Unzip ieee-fraud-detection.zip.

### 3. Build the image
```
make build
```

### 4. Start the services
Start both the two services
```
docker-compose up
```
or only start the `app` service using
```
docker-compose up app
```

![image](pictures/docker-compose-up-app.png)
## Here are some documentations
[How to set up the working environment for this project](docs/dev_mode.md)

[API example](docs/api_example.md)

## TODO
### Database
Put the data inside a database to make it more similar to real world cases.

### Ray tune
Currently the whenever fastapi trigger ray tune will costs a lot of memory usage. If the **num_samples** is set to greater than one, it will crush my personal computer which has 16G mermery. Might need to use `Celery` and save the data to binary files instead of CSV.

## API 
Might need to run API in mutiple machines and put load balancer for big amount of  request.