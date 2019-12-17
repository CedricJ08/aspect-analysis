

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




filenames = ["All Beauty.parquet", "All Electronics.parquet", "Amazon Fashion.parquet", "Appliances.parquet", "Arts.parquet", "Automotive.parquet", "Baby Products.parquet", "Baby.parquet", "Camera & Photo.parquet", "Car Electronics.parquet", "Cell Phones & Accessories.parquet", "Clothing.parquet", "Computers.parquet", "Electronics.parquet", "GPS & Navigation.parquet", "Health & Personal Care.parquet", "Home & Kitchen.parquet", "Home Improvement.parquet", "Industrial & Scientific.parquet", "Kitchen & Dining.parquet", "Luxury Beauty.parquet", "Musical Instruments.parquet", "Office & School Supplies.parquet", "Office Products.parquet", "Patio.parquet", "Software.parquet", "Sports & Outdoors.parquet", "Tools & Home Improvement.parquet", "Toys & Games.parquet", "Video Games.parquet"]


for filename in filenames :
    print('filename : '+str(filename))

    df = sqlContext.read.parquet("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/sentiments_aspects/Sentiments "+filename).unpersist()

    df_clean = df.where(F.col("predicted_score").isNotNull()).repartition(200)

    df_clean.write.mode('append').parquet("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/sentiments_final/Sentiments "+filename)


