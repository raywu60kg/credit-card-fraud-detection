init-data:
	mkdir data
	kaggle competitions download -c ieee-fraud-detection
	mv ieee-fraud-detection.zip data/
	unzip data/ieee-fraud-detection.zip -d data
	echo "Init data sucessfully"
	