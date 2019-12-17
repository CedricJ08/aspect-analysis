#################################################
# Config
#################################################
import os
os.system("python -m spacy download en_core_web_sm")


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
import re

# spacy
import spacy
nlp = spacy.load('en_core_web_sm', disable=["ner", "tagger"])

# nltk
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
untok = TreebankWordDetokenizer()

# pyspark
import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.sql.functions import abs, rand, udf
from pyspark.sql.types import StringType, DoubleType, StructType, StructField
from pyspark.sql import functions as F

# sparknlp
import sparknlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from sparknlp.embeddings import *

# keras
from keras.models import load_model
from keras.preprocessing import sequence



#################################################
# Initialize
#################################################

print("################### Initialize Spark Session #########################")
# read in data

conf = pyspark.SparkConf().setAll([('spark.nodemanager.vmem-check-enabled', 'false')])

sc = SparkSession \
    .builder \
    .appName("ML SQL session") \
    .config(conf=conf) \
    .getOrCreate()

sqlContext = SQLContext(sparkContext=sc.sparkContext, sparkSession=sc)

#################################################
# Load Sentiment Analyzer
#################################################

################ Fit vecetorizer #############################

df_train = sqlContext.read.parquet("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/Training_set.parquet")\
                           .orderBy(rand())\
                           .repartition(200)\
                           .unpersist()

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
  .setOutputCol("embed")

finisher = Finisher() \
    .setInputCols(["embed"]) \
    .setOutputCols(["embeddings"]) \
    .setCleanAnnotations(True)

nlp_pipeline = Pipeline().setStages([document_assembler, sentence_detector, tokenizer, word_embeddings, finisher])

featurizer = nlp_pipeline.fit(df_train)



################ import Keras model #############################
model_file = file_io.FileIO("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/sent_analyzer_g.h5", mode='rb')
temp_model_location = './temp_model_g.h5'
temp_model_file = open(temp_model_location, 'wb')
temp_model_file.write(model_file.read())
temp_model_file.close()
model_file.close()

model = load_model(temp_model_location)



#################################################
# Predict function
#################################################


def predict(embed_rows):
    # feature = sequence.pad_sequences(np.array([[embed_row['embeddings'] for embed_row in embed_rows]]), maxlen=200, dtype=float, padding='post')

    # test with finisher
    feature = sequence.pad_sequences(np.array([[x.split(' ') for x in embed_rows]], dtype =float), maxlen=200, dtype=float, padding='post')
     
    return 2*(model.predict(feature).item()-0.5)

predict_udf = udf(predict, DoubleType())



#################################################
# Split functions
#################################################

def split_nsubj(multi_clause_sentence_nlp):
    
    subjs = [token for token in multi_clause_sentence_nlp if token.dep_ in ['nsubj', 'nsubjpass', 'csubj', 'csubjpass']]
    if len(subjs) == 0 :
        return [multi_clause_sentence_nlp.text]
    else :
        simple_clause_sentences = []
        for subj in subjs:
            head_subj = subj.head
            sub_tree = get_sub_tree([head_subj], [head_subj])+[subj]
            sub_toks = list(np.array([t.text for t in sub_tree])[np.argsort([t.i for t in sub_tree])])
            sub_sent = untok.detokenize(sub_toks)
            simple_clause_sentences.append(sub_sent)
        return simple_clause_sentences




def get_sub_tree(state, sub_tree):
    state_children = list(np.concatenate([[child for child in head.children if child.dep_ not in ['nsubj', 'nsubjpass', 'csubj', 'csubjpass']] for head in state]))
    if len(state_children) == 0 :
        return sub_tree
    else :
        sub_tree += state_children
        return get_sub_tree(state_children, sub_tree)



def splitSentences(multi_clause_sentences_nlp):
    return list(np.concatenate([split_nsubj(sent_nlp) for sent_nlp in multi_clause_sentences_nlp.sents]))


def find_sentence(candidate_sentences, target_aspect):
    # scores_sentences = list(map(lambda potential_sent : sum(map(lambda word_ : len(potential_sent)+10000 if word_ in potential_sent.lower() else 0, target_aspect.split(' '))), candidate_sentences))
    scores_sentences = [sum([len(candidate_sentence)+10000 if word_ in candidate_sentence.lower() else 0 for word_ in target_aspect.split(' ')]) for candidate_sentence in candidate_sentences]
    idx_best_sentence = np.argmax(scores_sentences)
    best_sentence = candidate_sentences[idx_best_sentence]
    return best_sentence

#################################################
# Loop over categories
#################################################

def get_aspect_sent(chunk_row):
    rev = chunk_row[0]
    asp = chunk_row[1]
    rev_nlp = nlp(rev)
    complexe_sentence = find_sentence([s.text for s in rev_nlp.sents], asp) # hate this line !!!
    complexe_sentence_nlp = nlp(complexe_sentence)
    simple_sentence = find_sentence(splitSentences(complexe_sentence_nlp), asp)

    return simple_sentence.replace("[^a-zA-Z ]", "")



# filenames = ["All Beauty.parquet", "All Electronics.parquet", "Amazon Fashion.parquet", "Appliances.parquet", "Arts.parquet", "Automotive.parquet", "Baby Products.parquet", "Baby.parquet", "Camera & Photo.parquet", "Car Electronics.parquet", "Cell Phones & Accessories.parquet", "Clothing.parquet", "Computers.parquet", "Electronics.parquet", "GPS & Navigation.parquet", "Health & Personal Care.parquet", "Home & Kitchen.parquet", "Home Improvement.parquet", "Industrial & Scientific.parquet", "Kitchen & Dining.parquet", "Luxury Beauty.parquet", "Luxury Beauty.parquet", "Microsoft.parquet", "Musical Instruments.parquet", "Office & School Supplies.parquet", "Office Products.parquet", "Patio.parquet", "Software.parquet", "Sports & Outdoors.parquet", "Tools & Home Improvement.parquet", "Toys & Games.parquet", "Video Games.parquet"]
 

# filenames = ["All Beauty.parquet", "All Electronics.parquet", "Amazon Fashion.parquet"]

# filenames = ["Appliances.parquet", "Arts.parquet", "Automotive.parquet", "Baby Products.parquet", "Baby.parquet", "Camera & Photo.parquet", "Car Electronics.parquet", "Cell Phones & Accessories.parquet", "Clothing.parquet", "Computers.parquet", "Electronics.parquet", "GPS & Navigation.parquet", "Health & Personal Care.parquet", "Home & Kitchen.parquet", "Home Improvement.parquet", "Industrial & Scientific.parquet", "Kitchen & Dining.parquet", "Luxury Beauty.parquet", "Luxury Beauty.parquet", "Microsoft.parquet", "Musical Instruments.parquet", "Office & School Supplies.parquet", "Office Products.parquet", "Patio.parquet", "Software.parquet", "Sports & Outdoors.parquet", "Tools & Home Improvement.parquet", "Toys & Games.parquet", "Video Games.parquet"]

# failed on "Electronics.parquet" !!! chunk 9/9 memory error
# failed on "Microsoft.parquet" !!! not Clustered Microsoft.parquet in bucket !
# failed on "Tools & Home Improvement.parquet" Typo but continued, i don't really know the impact of this error...

filenames = ["Video Games.parquet"]


chunks_size = 200000
for filename in filenames :
    print('filename : '+str(filename))

    df_cat = sqlContext.read.parquet("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/clustered_aspects/Clustered "+filename)\
                            .withColumn("idx_", F.monotonically_increasing_id())\
                            .unpersist()
                            
    len_ = df_cat.count()
    nb_chunks = int(len_/chunks_size)+1
    
    chunks = df_cat.randomSplit(weights=[1/nb_chunks] * nb_chunks)
    count_iter = 1
    for chunk in chunks :
        print ('chunk : '+str(count_iter)+' over '+str(nb_chunks))
        row = chunk.rdd.map(list).map(lambda r: (int(r[-1]), (r[5], r[7])))
        row = row.mapValues(lambda r : get_aspect_sent(r))

        df_sentence_split = row.toDF(StructType([StructField('idx_', StringType(), True), StructField('text', StringType(), True)])).repartition(100).unpersist()


        df_predict = featurizer.transform(df_sentence_split)\
                                .withColumn( "predicted_score", predict_udf("embeddings"))\
                                .select("idx_", "text", "predicted_score")\
                                .unpersist()

        df_output = df_cat.join(df_predict, on=['idx_'], how='left_outer').unpersist()

        df_output.write.mode('append').parquet("gs://"+GCP_BUCKET_NAME+"/"+DIRECTORY_PREFIX+"/sentiments_aspects/Sentiments "+filename)
        
        count_iter+=1
