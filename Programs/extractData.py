# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:12:23 2019

Here we convert our data from Json to Pyspark Dataframe and save as a parquet file 

@author: Austin Bell
"""

import pyspark
import pyspark.sql.functions as F
from pyspark import SparkConf, SparkContext

conf = SparkConf()
conf.set('spark.local.dir', '/remote/data/match/spark')
conf.set('spark.sql.shuffle.partitions', '2100')
conf.set("spark.sql.broadcastTimeout",  3000)
SparkContext.setSystemProperty('spark.executor.memory', '10g')
SparkContext.setSystemProperty('spark.driver.memory', '10g')
sc = SparkContext(appName='mm_exp', conf=conf)
sqlContext = pyspark.SQLContext(sc)

# load in our reviews dataset 
print("extracting Reviews")
df = sqlContext.read.json("gs://core_bucket_abell/aspect_ranking/item_dedup.json")
df = df.select("asin", "overall", "reviewText")\
    .withColumnRenamed("overall", "scores")\
    .withColumnRenamed("reviewText", "reviews")

# load in metadata dataset
print("Extracting and Fixing Metadata")
metadata = sqlContext.read.json("gs://core_bucket_abell/aspect_ranking/metadata.json")

# metadata dataset is pretty corrupted so we have to fix this
nulls = metadata.where(F.col("asin").isNull() | F.col("categories").isNull()) # get nulls

# fix our corrupted
fixed = nulls.select("_corrupt_record")\
    .withColumn("_corrupt_record", F.regexp_replace("_corrupt_record", "[\'|\\\']", "\""))\
    .withColumn("categories", F.regexp_extract(F.col("_corrupt_record"), "\"categories\": (\[\[.*\]\])",1))\
    .withColumn("asin", F.regexp_extract(F.col("_corrupt_record"), "\"asin\": \"(.*?[0-9|A-Z])\"",1))\
    .where(F.col("categories").isNotNull() & F.col("asin").isNotNull())\
    .select("categories", "asin")

# combine
metadata = metadata.select("categories", "asin")\
    .where(F.col("categories").isNotNull() & F.col("asin").isNotNull())\
    .withColumn("categories", F.concat_ws(", ", F.flatten(F.col("categories"))))\
    .union(fixed)
    
    
# merge categories to dataframe 
print("Joining the Reviews and Metadata")
final_df = df.join(metadata, "asin", how = "left")\
    .where(F.col("categories").isNotNull())

print("Exporting")
final_df.write.parquet("gs://core_bucket_abell/aspect_ranking/Amazon Product Reviews Raw Total.parquet")