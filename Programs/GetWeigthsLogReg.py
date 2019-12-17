import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import json
import os
from io import BytesIO
from tensorflow.python.lib.io import file_io


GCP_BUCKET_NAME = "bucket-bigdata-proj"
DIRECTORY_PREFIX = "Lab/Data"

import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.sql import functions as F


conf = pyspark.SparkConf().setAll([('spark.nodemanager.vmem-check-enabled', 'false')])

sc = SparkSession \
    .builder \
    .appName("ML SQL session") \
    .config(conf=conf) \
    .getOrCreate()

sqlContext = SQLContext(sparkContext=sc.sparkContext, sparkSession=sc)


# filenames = ["All Beauty.parquet", "All Electronics.parquet", "Amazon Fashion.parquet", "Appliances.parquet", "Arts.parquet", "Automotive.parquet", "Baby Products.parquet", "Baby.parquet", "Camera & Photo.parquet", "Car Electronics.parquet", "Cell Phones & Accessories.parquet", "Clothing.parquet", "Computers.parquet", "Electronics.parquet", "GPS & Navigation.parquet", "Health & Personal Care.parquet", "Home & Kitchen.parquet", "Home Improvement.parquet", "Industrial & Scientific.parquet", "Kitchen & Dining.parquet", "Luxury Beauty.parquet", "Luxury Beauty.parquet", "Microsoft.parquet", "Musical Instruments.parquet", "Office & School Supplies.parquet", "Office Products.parquet", "Patio.parquet", "Software.parquet", "Sports & Outdoors.parquet", "Tools & Home Improvement.parquet", "Toys & Games.parquet", "Video Games.parquet"]
# filenames = ["All Beauty.parquet", "All Electronics.parquet", "Amazon Fashion.parquet"]

filenames = ["Appliances.parquet", "Arts.parquet", "Automotive.parquet", "Baby Products.parquet", "Baby.parquet", "Camera & Photo.parquet", "Car Electronics.parquet", "Cell Phones & Accessories.parquet", "Clothing.parquet", "Computers.parquet", "Electronics.parquet", "GPS & Navigation.parquet", "Health & Personal Care.parquet", "Home & Kitchen.parquet", "Home Improvement.parquet", "Industrial & Scientific.parquet", "Kitchen & Dining.parquet", "Luxury Beauty.parquet", "Luxury Beauty.parquet", "Microsoft.parquet", "Musical Instruments.parquet", "Office & School Supplies.parquet", "Office Products.parquet", "Patio.parquet", "Software.parquet", "Sports & Outdoors.parquet", "Tools & Home Improvement.parquet", "Toys & Games.parquet", "Video Games.parquet"]


def create_vectors(x):
    s = pd.Series(data=[0]*len(aspects)+[x['scores'].values[0]], index=list(aspects)+['scores'])
    for i in range (len(x)):
        s[x['key_aspect']]=x['predicted_score']
    return s

def save_weigths(weigths_, category_name):
    json_ = json.dumps(weigths_)
    with file_io.FileIO('weights_g.json', mode='wb') as f:
        f.write(json_)
    with file_io.FileIO('weights_g.json', mode='rb') as input_f:
        with file_io.FileIO("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/weigths/weights_"+category_name+"_g.json", mode='wb+') as output_f:
            output_f.write(input_f.read())
            print("Saved weights_"+category_name+"_g.json to GCS")


for filename in filenames :
    print ("working on "+filename)
    df_input = sqlContext.read.parquet("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/sentiments_aspects/Sentiments "+filename)
    df_input_pd = df_input.toPandas()
    aspects = df_input_pd['key_aspect'].unique()
    df_opinions = df_input_pd.groupby('idx').apply(create_vectors)

    y = df_opinions.iloc[:,-1].values
    x = df_opinions.iloc[:,:-1].values

    lr = LinearRegression()

    lr.fit(x,y)
    weights = {aspects[i] : lr.coef_[i] for i in range(len(aspects))}

    save_weigths(weights, filename)




