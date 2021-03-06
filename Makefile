init-data:
	mkdir data
	kaggle competitions download -c ieee-fraud-detection
	mv ieee-fraud-detection.zip data/
	unzip data/ieee-fraud-detection.zip -d data
	echo "Init data sucessfully"

run-app:
	uvicorn api.app:app --host 0.0.0.0 --port 8000
	
run-app-dev:
	uvicorn api.app:app --reload
	
kaggle-predict:
	python scripts/get_kaggle_predictions.py

build:
	docker build -f ./docker/dockerfile -t=lightgbm-project-demo:0.0.1 .
	
