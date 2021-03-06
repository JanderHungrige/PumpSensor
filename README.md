# PumpSensor

![alt](images/rsz_1wasserhaltung_pumpe_1.jpg?raw=true)

<br><br>
**Summary**

This Repo will analyze the sensor data of several pumps. You can find the data here: https://ga-data-cases.s3.eu-central-1.amazonaws.com/pump_sensor.zip

After a first analysis, we will use a simple LSTM to predict 10min into the future to detect any upcoming failures.
<br><br>
**Files:**

**Sensor_analysis.py** : Here we will analyze and manipulate the sensor data. This incorporates several plot functions to visualize the data.  
**Sensor_learning.py** : Here we create the time series for prediction, set up and train the LSTM. The model is saved into a */model* folder. Result plots are generated  
**Sensor_learning_OOP.py** :This file has the same functionallity as the *Sensor_learning.py* file, however, it is coded object oriented. This is meant for better implementation, but also for people learning OOP to compare both files and see the differneces. The *Sensor_learning.py* might be easier to read/understand for some.  
**printing_functions.py** : This file is used by the *Sensor_learning_OOP.py* file
<br><br>

**Quick Peak:**

We have this amount of data:

![alt](images/overview.png?raw=true)

The Data is composed like this:

![alt](images/keys.png?raw=true)

We use the following simple LSTM to predict the classes:

![alt](images/model42.png?raw=true)

With this simple model, we achieved a prediction that should be sufficient to warn the teams 10min in advance about a pump failure. 

![alt](images/Prediction_class_fapi_10.png?raw=true)
