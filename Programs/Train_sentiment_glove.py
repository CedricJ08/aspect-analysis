#################################################
# Config
#################################################


# ! python -m spacy download en_core_web_sm
# ! conda update wrapt
# ! pip install tensorflow==1.14

GCP_BUCKET_NAME = "bucket-bigdata-proj"
DIRECTORY_PREFIX = "Lab/Data" # "aspect_ranking" for you !

extracted_data_filename = "Extracted Aspects.parquet"

#################################################
# Imports
#################################################
print("################### Importation #########################")

import os
from io import BytesIO
from tensorflow.python.lib.io import file_io
import numpy as np

# pyspark
import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import abs, rand
from pyspark.sql.types import StringType
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType, StructType, StructField, IntegerType

# sparknlp
import sparknlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from sparknlp.embeddings import *


import tensorflow
import tensorflow.python
# keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM, Bidirectional



#################################################
# Initialize
#################################################
print("################### Initialize Spark Session #########################")
# read in data

conf = pyspark.SparkConf().setAll([('spark.nodemanager.vmem-check-enabled', 'false'), ('spark.executor.memoryOverhead', '2500') ])

sc = SparkSession \
    .builder \
    .appName("ML SQL session") \
    .config(conf=conf) \
    .getOrCreate()


sqlContext = SQLContext(sparkContext=sc.sparkContext, sparkSession=sc)


# print("################## load DATA #################################")
#################################################
# Load Training data
#################################################

filenames = ["All Beauty.parquet", "All Electronics.parquet", "Amazon Fashion.parquet", "Appliances.parquet", "Arts.parquet", "Automotive.parquet", "Baby Products.parquet", "Baby.parquet", "Camera & Photo.parquet", "Car Electronics.parquet", "Cell Phones & Accessories.parquet", "Clothing.parquet", "Computers.parquet", "Electronics.parquet", "GPS & Navigation.parquet", "Health & Personal Care.parquet", "Home & Kitchen.parquet", "Home Improvement.parquet", "Industrial & Scientific.parquet", "Kitchen & Dining.parquet", "Luxury Beauty.parquet", "Luxury Beauty.parquet", "Microsoft.parquet", "Musical Instruments.parquet", "Office & School Supplies.parquet", "Office Products.parquet", "Patio.parquet", "Software.parquet", "Sports & Outdoors.parquet", "Tools & Home Improvement.parquet", "Toys & Games.parquet", "Video Games.parquet"]

# Create a dataframe from the reviews where no aspecte was detected
# Class defined as 0 if score = 1 or 2, 1 if score = 4 or 5
# Undersampling to rebalance the classses
# Output simple dataframe with 3 columns : idx, text, label


###" SAVE TRAINING DF #############

# for filename in filenames :
#   print (filename+"                    Loading...")
#   df_temp = sqlContext.read.parquet("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/extracted_aspects/"+"Extracted "+filename)\
#                        .filter(F.col("scores").isin(1, 2, 4, 5))\
#                        .withColumn("label", (1+(F.col("scores")-3)/abs(F.col("scores")-3))/2)\
#                        .unpersist()
#   df_temp = df_temp.filter(F.col("label")==0).union(df_temp.orderBy(rand()).limit(df_temp.filter(F.col("label")==0).count()))\
#                        .withColumn("text", F.col("reviews"))\
#                        .orderBy(rand())\
#                        .limit(5000)\
#                        .select("idx", "text", "label")\
#                        .repartition(10)\
#                        .unpersist()
#   df_temp.write.mode('append').parquet("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/Training_set.parquet")


# df_train = sqlContext.read.parquet("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/Training_set.parquet")\
#                            .orderBy(rand())\
#                            .repartition(200)\
#                            .unpersist()





#################################################
# Vectorize 
#################################################


################# Vectorizer Pipeline ###########################

# Built ad fit the vectorizer Pipeline

def get_nlp_model(df_train_):
    print("build vectorizer")
    document_assembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

    sentence_detector = SentenceDetector() \
      .setInputCols(["document"]) \
      .setOutputCol("sentence") \
      .setUseAbbreviations(True)

    tokenizer = Tokenizer().setInputCols(["sentence"])\
      .setOutputCol("token")

    word_embeddings = WordEmbeddingsModel.pretrained()\
      .setInputCols(["document", "token"])\
      .setOutputCol("embeddings")


    nlp_pipeline = Pipeline().setStages([document_assembler, sentence_detector, tokenizer, word_embeddings])

    nlp_model_ = nlp_pipeline.fit(df_train_)

    return nlp_model_

# print("################### Get Features #########################")
# nlp_model = get_nlp_model(df_train)
# df_features = nlp_model.transform(df_train)


################# Create Feature Matrix ###########################
# We create from df_featurize x_train and y_train that would feed the tensorflow model

def get_features(row_):
    result_feat = []
    for tk in row_:
        result_feat.append(tk['embeddings'])
    return np.array(result_feat)

def build_data(df_input, nb_chunks=10):
    x_train_builder = []
    y_train_builder = []

    row_count = df_input.count()
    i = 0
    
    chunks = df_input.randomSplit(weights=[1/nb_chunks] * nb_chunks)

    for chunk in chunks:
        rows = chunk.collect()
        for row in rows:
            if i % 1000 == 0:
                print('row {} / {} ({:.1f} %)'.format(i, row_count, 100 * i / row_count))
            embeddings_ = get_features(row['embeddings'])
            label_ = row['label']
            x_train_builder.append(embeddings_)
            y_train_builder.append(label_)
            i += 1

    x_train_builder = np.array(x_train_builder)
    y_train_builder = np.array(y_train_builder)
    return x_train_builder, y_train_builder

# print("################### create Matrix #########################")
# x_train, y_train = build_data(df_features)


# print("################### Save Matrix #########################")
# np.save(file_io.FileIO("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/x_train", 'w'), x_train)
# np.save(file_io.FileIO("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/y_train", 'w'), y_train)


# f = BytesIO(file_io.read_file_to_string("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/x_train", binary_mode=True))
# x_train = np.load(f, allow_pickle=True)
# f = BytesIO(file_io.read_file_to_string("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/y_train", binary_mode=True))
# y_train = np.load(f, allow_pickle=True)

def save_train(x_train, y_train):
  step = 10000
  deb = 0
  fin = step
  len_ = len(x_train)
  counter = 0
  while fin < len_ :
    print(" Batche n°{} Saved".format(str(counter)))
    np.save(file_io.FileIO("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/x/x_train_"+str(counter), 'w'), x_train[deb:fin])
    np.save(file_io.FileIO("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/y/y_train_"+str(counter), 'w'), y_train[deb:fin])
    deb = deb + step
    fin = fin + step
    counter += 1
  np.save(file_io.FileIO("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/x/x_train_"+str(counter), 'w'), x_train[deb:fin])
  np.save(file_io.FileIO("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/y/y_train_"+str(counter), 'w'), y_train[deb:fin])
  print("X, y Saved")
  print ("{} batchs saved".format(str(counter)))
  return None

# save_train(x_train, y_train)


print("################### Load Matrix #########################")
def load_train():
  x_train_load = np.array([])
  y_train_load = np.array([])
  for i in range(14):
    f = BytesIO(file_io.read_file_to_string("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/x/x_train_"+str(i), binary_mode=True))
    x_train_tmp = np.load(f, allow_pickle=True)
    x_train_load = np.concatenate([x_train_load, x_train_tmp])
    f = BytesIO(file_io.read_file_to_string("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/y/y_train_"+str(i), binary_mode=True))
    y_train_tmp = np.load(f, allow_pickle=True)
    y_train_load = np.concatenate([y_train_load, y_train_tmp])
  return x_train_load, y_train_load

x_train, y_train = load_train()


#################################################
# Train CNN_Bilstm 
#################################################

print("################### Train Model #########################")

maxlen = 200
batch_size = 64 
filters = 128 
kernel_size = 3 
bilstm_dim = 100 
dropout = 0.2
epochs = 10

def train_Sentiment_Anlyzer(x_train_, y_train_):
    x_train_ = sequence.pad_sequences(x_train_, maxlen=maxlen, dtype=float, padding='post')
    model_ = Sequential()
    model_.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu'))
    model_.add(MaxPooling1D(pool_size=kernel_size))
    model_.add(Bidirectional(LSTM(bilstm_dim)))
    model_.add(Dropout(dropout))
    model_.add(Dense(1, activation='sigmoid'))
    model_.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_.fit(x_train_, y_train_,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)

    return model_

model = train_Sentiment_Anlyzer(x_train, y_train)

#################################################
# Save model
#################################################

print("################### Save Model #########################")

def save_model(model_saved):
    model_saved.save('sent_analyzer_g.h5')
    with file_io.FileIO('sent_analyzer_g.h5', mode='rb') as input_f:
        with file_io.FileIO("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/sent_analyzer_g.h5", mode='wb+') as output_f:
            output_f.write(input_f.read())
            print("Saved sent_analyzer_g.h5 to GCS")


save_model(model)
