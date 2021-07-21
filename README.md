# PumpSensor

![alt](images/rsz_1wasserhaltung_pumpe_1.jpg?raw=true)

<br><br>

**This Repo will analyze the data of pumps you can find here:** ![Data](https://ga-data-cases.s3.eu-central-1.amazonaws.com/pump_sensor.zip)

**After the first analysis, we will use a simple LSTM to predict 10min into the future to detect any upcomming failures.**

<br><br>
We have this amount of data:

![alt](images/overview.png?raw=true)

The Data is composed like this:

![alt](images/keys.png?raw=true)

We use folling simple LSTM to predict the classes:

![alt](images/model.png?raw=true)

With this simple model we achieved a prediction that should be sufficient to warn the teams 10min in advance about a pump failure. 

![alt](images/Prediction_class_fapi_10.png?raw=true)



