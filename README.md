# DeepLearning_Anomalies_Detection_with_Bluetooth_Sensor_Data
Final Year Project. Constructing models to create offline anomalies detection using Travel Time Data collected from Bluetooth sensors along the route.

The Bluetooth sensor Data is collected from Coronation Drive and Milton Road Corridors located in Bribane in 2017 and early 2018, which shown in the link below:
https://www.google.com/maps/@-27.483276,152.9775122,14z/data=!3m1!4b1!4m2!6m1!1s1UYOOSCqlAaE-H2yjm_OhaoaDLF1qKgpo

The dataset details are as follow:

Column 0: id

Column 1: mac_id

Column 2: node_id

Column 3: date 

Column 4: time

Column 5: duration

Models used in this project are LSTM Autoencoder and 1D-Convolution NN Autoencoder.

As the dataset has no label about when and where the anomalies occured, outliers at different time intervals labelled as anomalies to test out the performance of the model.
The purpose of this project is to test out the capability of Neural Network to capture the anomalies of route by bluetooth sensor data and further looks for a optimal model approach to do so.
Further development may be real-time anomalies detection.
