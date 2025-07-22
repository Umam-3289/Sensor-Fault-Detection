from  pymongo import MongoClient 
import pandas as pd
import json

url="mongodb+srv://Umam:12345@cluster0.9ajxfcx.mongodb.net/?retryWrites=true&w=majority"

client=MongoClient(url)

DATABASE_NAME="SensorData"
COLLECTION_NAME="waferfault"

df=pd.read_csv("E:\DATA ANALYTICS\Projects\Sensor Fault Detection\notebooks\wafer_23012020_041211.csv")
df=df.drop("Unnammed: 0",axis=1)

json_record=list(json.loads(df.T.to_csv()).values())
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)


