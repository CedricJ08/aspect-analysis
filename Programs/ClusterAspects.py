# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 23:47:08 2019

Cluster the aspect and select the most representative word 

@author: Austin Bell
"""

#################################################
# Imports
#################################################

import re, string, time, datetime
import numpy as np
from tqdm import tqdm
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# pyspark
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark import SparkConf, SparkContext
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, StringType, Row, DoubleType, IntegerType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.window import Window
#from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
import pyspark.ml.feature as feat

# sparknlp
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.embeddings import *



#################################################
# Initialize
#################################################
# read in data

#conf = SparkConf()
#conf.set('spark.local.dir', '/remote/data/match/spark')
#conf.set('spark.sql.shuffle.partitions', '100')
#conf.set("spark.default.parallelism", "100")
#conf.set("spark.sql.broadcastTimeout",  3000)
#conf.set("spark.executor.instances", "25")
#conf.set("spark.executor.cores",  '4')
#SparkContext.setSystemProperty('spark.executor.memory', '10g')
#SparkContext.setSystemProperty('spark.driver.memory', '10g')
#sc = SparkContext(appName='cluster', conf=conf)
#sqlContext = pyspark.SQLContext(sc)

spark = SparkSession \
    .builder \
    .appName("aspect extraction") \
    .getOrCreate()

spark.conf.set("spark.sql.broadcastTimeout",  1000)

# remove stop words 
def rmStop(aspect):
    split_aspect = aspect.split()
    aspect = [s for s in split_aspect if s not in stop_words and s not in string.punctuation]
    return ' '.join([s for s in aspect])

rmStop_udf = F.udf(rmStop, StringType())


def prepData(category):
    
    df = spark.read.parquet("gs://core_bucket_abell/aspect_ranking/extracted_aspects/Extracted " + category + ".parquet")

    #explode aspects
    spread_df = df.select("*", F.posexplode("indiv_aspects").alias("pos", "split_aspect"))

    # get unique by category and split aspect - remove blank aspects
    unique_cat_aspect = spread_df.groupBy(*('asin', "reviews", "main_cat", "categories", "scores", "split_aspect", "idx")).count()\
        .withColumn("split_aspect", F.lower(F.col("split_aspect")))\
        .withColumn("split_aspect", rmStop_udf(F.col("split_aspect")))\
        .filter("split_aspect != ''")
    
    return unique_cat_aspect

##################################################
# Generate BERT Embeddings
##################################################
# Using Spark NLP package from Jon Snow Labs
document_assembler = DocumentAssembler()\
    .setInputCol("split_aspect")\
    .setOutputCol("aspect_text")

tokenizer = Tokenizer().setInputCols(["aspect_text"])\
    .setOutputCol("token")

#BertEmbeddings.pretrained('bert_base_uncased', 'en')
word_embeddings = WordEmbeddingsModel.pretrained()\
  .setInputCols(["aspect_text", "token"])\
  .setOutputCol("embeddings")

vec_pipeline = Pipeline().setStages(
  [
    document_assembler,
    tokenizer,
    word_embeddings
  ]
)


# this gets word embeddings for each word, now we average
def avg_vectors(bert_vectors):
    length = len(bert_vectors[0]["embeddings"])
    avg_vec = [0] * length
    for vec in bert_vectors:
        for i, x in enumerate(vec["embeddings"]):
            avg_vec[i] += x
    avg_vec[i] = avg_vec[i] / length
    return avg_vec


# conver to pyspark dense vectors
def dense_vector(vec):
    return Vectors.dense(vec)
dense_vector_udf = F.udf(dense_vector, VectorUDT())
   

# Remove zero vectors
def vector_sum(arr): 
    return int(np.sum(arr))

vector_sum_udf = F.udf(vector_sum, IntegerType())

#########################################################
# Functions for Clustering, identifying closest points
# and selecting the most representative words / ngrams
########################################################
# train kmeans algorithm 
def idKMeans(training2, init, dist):
    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    # run various kmeans and see which has the max silhouette score
    # need to test cosine distance as well 
    score = -1
    optim_k = None
    optim_model = None
    for k in range(75, min(training2.count(),105), 5):
        print(k)
        kmeans = KMeans(k=k, maxIter=10, seed=1, initMode=init, distanceMeasure=dist) 
        model = kmeans.fit(training2)

        silhouette = evaluator.evaluate(model.transform(training2.select('features')))
        print(silhouette)
        
        if silhouette > score:
            score = silhouette
            optim_model = model
            optim_k = k
        
    print("Selecting the Optimal K: " + str(optim_k))
    return optim_model

# compute distances to clusters 
def calcDist(centers, feature, prediction):
    # get relevant centroid
    c = centers[prediction]
    return float(feature.squared_distance(c))

def calcDist_udf(centers):
     return F.udf(lambda features, preds: calcDist(centers, features, preds), FloatType())

def idDistances(clustered_df, model):
    centers = model.clusterCenters()
    centers = [c.tolist() for c in centers]
    clustered_df = clustered_df.withColumn('prediction',F.col('prediction').cast(IntegerType()))

    clustered_dist = clustered_df.withColumn("dist", calcDist_udf(centers)(F.col("features"), F.col("prediction")))
    
    return clustered_dist


# identify closest n by group
def closestN(clustered_df, n):
    window = Window.partitionBy(clustered_df['prediction']).orderBy(clustered_df['dist'])

    closest = clustered_df.select(F.col('*'), F.row_number().over(window).alias('row_number')) \
        .where(F.col('row_number') <= n)

    return closest

# select most frequent aspect by cluster 
# Will also test by highest word count, but this is easier for now 
def freqNGram(closest):
    window = Window.partitionBy(closest['prediction']).orderBy(closest['count'].desc())
    
    key_aspect = closest.select(F.col("*"), F.row_number().over(window).alias("freq_rank")) \
        .where(F.col("freq_rank") == 1) \
        .select(['prediction', 'split_aspect']) \
        .withColumnRenamed("split_aspect", "key_aspect")

    return key_aspect

# compute word frequency by group
def freqWord(closest, num_words):
    word_freq = closest.withColumn('word', F.explode(F.split(F.col('split_aspect'), ' ')))\
        .groupBy('prediction','word')\
        .count()\
        .sort('count', ascending=False)\
        .filter(~F.lower(F.col("word")).isin(stop_words))\
        .filter(F.col("count") >= 2)

    # filter out stop words and collapse top n words
    window = Window.partitionBy(word_freq['prediction']).orderBy(word_freq['word'].desc())

    key_aspect = word_freq.select(F.col("*"), F.row_number().over(window).alias("rank"))\
        .where(F.col("rank") <= num_words)\
        .groupBy("prediction")\
        .agg(F.collect_list("word").alias("word"))\
        .withColumn("word", F.concat_ws("|", "word"))\
        .withColumnRenamed("word", "key_aspect"+str(num_words))
        
                
    return key_aspect

# Compute TF-IDF score and return the top 5 tf-idf words 
def getTFIDF(closest):
    grouped_clusters = closest.groupBy("prediction")\
        .agg(F.collect_list("split_aspect").alias("text"))\
        .withColumn("text", F.concat_ws(" ", "text"))

    tokenizer = feat.Tokenizer(inputCol="text", outputCol="words")
    wordsData = tokenizer.transform(grouped_clusters)

    # get term freqs (using count vectorizer because it does hash the words and we can revert back to words from idx)
    cv = feat.CountVectorizer(inputCol="words", outputCol="rawFeatures").fit(wordsData)
    featurizedData = cv.transform(wordsData)

    # save vocab object
    vocab = cv.vocabulary

    # compute idf
    idf = feat.IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    tfidf = idfModel.transform(featurizedData)

    return tfidf, vocab


# extract the top n words from each sparse vector 
def topNvec(vocab, vector, n):
    # get idx and values in a list and sort
    idx = vector.indices.tolist()
    values = vector.values.tolist()
    idx_values = list(zip(idx, values))
    sorted(idx_values, key = lambda x: x[1], reverse = True)
    
    # extract top n 
    top_idx = [i for (i, val) in idx_values][:n]
    
    # use vocabulary to convert back to string
    top_words = [vocab[i] for i in top_idx]
    return top_words

def topNvec_udf(vocab):
     return F.udf(lambda vector, n: topNvec(vocab, vector, n), ArrayType(StringType()))

# bring all the tf functions together and collapse string lists
def topTFIDF(closest, num_words):
    tfidf, vocab = getTFIDF(closest)
    
    tfidf = tfidf.withColumn("key_aspect_tfidf", topNvec_udf(vocab)(F.col("features"), F.lit(num_words)))\
        .withColumn('key_aspect', F.concat_ws("|", F.col('key_aspect_tfidf')))\
        .select("prediction", "key_aspect")
    
    return tfidf

#############################################################
# Run Everything from Top to Bottom
############################################################
# Now we bring it all together
def main(inp_df, dist_str):
    
    # k means
    model = idKMeans(inp_df, "random", dist_str) # 'k-means||'
    clustered_df = model.transform(inp_df)

    clustered_df = clustered_df.select(['asin', 'idx', 'main_cat', 'categories', "reviews", "scores",
                                        'split_aspect', 'count', 'features', 'prediction'])

    # compute distances and select closest set of words
    clustered_dist = idDistances(clustered_df, model)
    closest = closestN(clustered_dist, 100)

    # select most representative words 
    print("Selecting the Most Representative Words")
    #key_aspect1 = freqNGram(closest)
    #key_aspect2 = freqWord(closest, 3)
    #key_aspect_freq = freqWord(closest, 5)
    key_aspect_tfidf = topTFIDF(closest, 5)

    # merge into single dataset 
    #matched_aspects_tmp = clustered_dist.join(key_aspect_freq, "prediction", how = "left")
    matched_aspects_tmp = clustered_dist.join(key_aspect_tfidf, "prediction", how = "left")
     
    #dfs.append(matched_aspects_tmp)
    
    return matched_aspects_tmp

#############################################################
# Loop Through each category and run main 
#############################################################
#categories = training.select("main_cat").distinct().rdd.map(lambda r: r[0]).collect()
#distances = ["euclidean", "cosine"]

#for dist in distances:
#dfs = []

categories = ['All Electronics', 'Electronics','Home & Kitchen', 'Home Improvement', 
 'Industrial & Scientific', 'Health & Personal Care',  'Camera & Photo', 'Software',
 'Cell Phones & Accessories','Arts', 'Computers', 'Office & School Supplies', 
 'GPS & Navigation','Car Electronics', 'Toys & Games', 'Kitchen & Dining','Tools & Home Improvement',
 'Baby', 'All Beauty', 'Baby Products', 'Microsoft', 'Office Products','Appliances',
'Amazon Fashion','Patio', 'Sports & Outdoors', 'Video Games',
 'Clothing', 'Musical Instruments', 'MP3 Players & Accessories', 'Automotive', 'Luxury Beauty']

for cat in tqdm(categories[26:]):
    print("Starting: ", cat)
    unique_aspects = prepData(cat)
    
    if unique_aspects.count() < 500:
        continue
    
    # get embeddings    
    print("generated Embeddings")
    glove_vecs = vec_pipeline.fit(unique_aspects).transform(unique_aspects)
    
    # average the embeddings
    avg_vectors_udf = F.udf(avg_vectors, ArrayType(DoubleType()))
    df_doc_vec = glove_vecs.withColumn("vector", avg_vectors_udf(F.col("embeddings")))
    
    # convert to dense format and remove zero vectors 
    training = df_doc_vec.withColumn("features", dense_vector_udf(F.col("vector")))\
        .withColumn('sum',vector_sum_udf(F.array('features'))).filter(F.col("sum") != 0)
    
    
    print(datetime.datetime.now())# for now limiting to 1 mil
    matched_aspects = main(training, "euclidean")
        
    # concatenate them 
    #matched_aspects = dfs[0]
    #for aspect_next in dfs[1:]:
    #    matched_aspects = matched_aspects.union(aspect_next)
    print("Exporting")
    matched_aspects.select("idx", "asin", "prediction", "categories", "main_cat", "reviews", 
                           "scores", "split_aspect", "count", "key_aspect")\
        .write\
        .parquet("gs://core_bucket_abell/aspect_ranking/clustered_aspects/Clustered " + cat + ".parquet")
        