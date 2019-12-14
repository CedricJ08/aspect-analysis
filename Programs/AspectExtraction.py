# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:42:23 2019

Leverage Dependency Trees to extract aspects

@author: Austin Bell
"""

#################################################
# Import packages and tools
#################################################


import numpy as np
import pandas as pd
import string, re

#!python -m spacy download en_core_web_sm

import spacy
nlp = spacy.load('en_core_web_sm', disable = ['ner'])

from nltk.stem import PorterStemmer
stem = PorterStemmer()

# init spark
from pyspark import SparkConf, SparkContext
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import Row
print("success")
base_path = "gs://core_bucket_abell/aspect_ranking"

# read in data
# the load method will need to change - I will not be able to save and load as csv. Will need to save as json
spark = SparkSession \
    .builder \
    .appName("aspect extraction") \
    .getOrCreate()
    
#sc.stop()

#conf = SparkConf()
#conf.set('spark.local.dir', '/remote/data/match/spark')
#conf.set('spark.sql.shuffle.partitions', '100')
#conf.set("spark.default.parallelism", "100")
#conf.set("spark.sql.broadcastTimeout",  3000)
#conf.set("spark.executor.instances", "25")
#conf.set("spark.executor.cores",  '4')
#SparkContext.setSystemProperty('spark.executor.memory', '10g')
#SparkContext.setSystemProperty('spark.driver.memory', '10g')
#sc = SparkContext(appName='mm_exp', conf=conf)
#sqlContext = pyspark.SQLContext(sc)

# linking verbs 
link_verbs = ['is', '\'s', 'am', 'are', 'wa', 'were', 'look', 'sound', 'smell', 'tast','feel','appear',
              'remain','seem','ha','had', 'have']

search_verbs = ['is', '\'s', 'am', 'are', 'was', 'were', 'look', 'sound', 'smell', 'tast','feel','appear',
              'remain','seem','has','had', 'have']
search_verbs = " " + "| ".join([v for v in search_verbs])


# personal pronouns - pretty much all pronouns except "it", "this", and "that"
pronouns = ["i", "you", "he", "she"," they", "me", "you", "him", "her", "my", "mine", "your", "yours",
            "his", "her", "hers", "who", "whom", "whose", "what", "which",  "another", "each", 
            "everything", "nobody", "either", "someone", "who", "whom", "whose", "which", 
           "myself", "yourself", "himself", "herself", "itself"]

clean = False
if clean:
    df = sqlContext.read.parquet(base_path + "/Amazon Product Reviews Raw Total.parquet")\
        .withColumn('cats', F.regexp_replace("categories", "\"", "")) \
        .withColumn('cats', F.regexp_replace("cats", ", ", "|")) \
        .withColumn("cats", F.concat(F.col("cats"), F.lit("|")))\
        .withColumn("main_cat", F.regexp_extract(F.col("cats"), "(.*?)\|",1))\
        .withColumn("main_cat", F.regexp_replace(F.col("main_cat"), "\[|\||\]\]", ""))\
        .where(F.col("reviews").isNotNull() & F.col("reviews").rlike(search_verbs))\
        .withColumn("idx", monotonically_increasing_id())
    
    # export 
    df.write.parquet(base_path + "/Amazon Product Reviews Raw Cleaned.parquet")
##########################################################
# Core Aspect Extraction Functions
##########################################################
# extracts the children for a verb (removes punctuation and verifies its not part of original chunk)
# utility function
def extractChildren(verb, chunk):
    return [child for child in verb.children 
            if child.text not in string.punctuation 
            and child.text not in [word.text for word in chunk]]


# identify individual aspects
# returns all potential adjectives excluding proper and pronouns
def extractIndividualAspects(noun_chunk, feature, indiv_aspects):
   
    if any([word for word in noun_chunk if word.pos_ in ['PRON', 'PROPN']]):
        indiv_aspects.append(feature)

    else:
        indiv_aspects += [noun_chunk.text, feature]
        
    return indiv_aspects

# Identify whether a noun chunk links to adjective
def link2adj(child, chunk, aspects, indiv_aspects, verb):
    # if adjective then we know its part of a feature
    if child.pos_ == "ADJ":

        # logic that captures "to VERB" after adjective - not very robust
        child_text = None
        for c in child.children:
            try:
                if c.pos_ == "VERB" and list(c.children)[0].text == "to":
                    child_text = ' '.join([word for word in [child.text, "to", c.text]])
            except:
                pass
        if child_text == None:
            child_text = child.text

        # append
        aspects.append(' '.join([word for word in [chunk.text, verb.text, child_text]]))

        # get the individual aspects too
        indiv_aspects = extractIndividualAspects(chunk, child_text, indiv_aspects)

    return aspects, indiv_aspects

# noun chunk links to noun then adjective
def link2noun(child, chunk, aspects, indiv_aspects, verb, skips, sent):
    if child.pos_ == "NOUN":
        # collect the entire chunk 
        feat_chunk = [chunk.text for chunk in list(sent.noun_chunks)
                      if re.search(re.escape(child.text), chunk.text)]
        
        # error handling
        if feat_chunk is None:
            feat_chunk = child
        elif len(feat_chunk) > 0:
            feat_chunk = feat_chunk[0]
        else:
            return aspects, indiv_aspects, skips

        # add to aspects and skips
        aspects.append(' '.join([word for word in [chunk.text, verb.text, feat_chunk]]))
        skips.append(feat_chunk)
        
        # get the individual aspects too
        indiv_aspects = extractIndividualAspects(chunk, feat_chunk, indiv_aspects)


    return aspects, indiv_aspects, skips

# recursive algorithm to extract conjugations 
def extractCONJ(child, chunk, aspects, indiv_aspects, verb, skips, sent):

    # see if we need to go further down the tree
    # if there are further children (multiple conjugations then we continue traversing to the bottom)
    child_children = extractChildren(child, chunk)
    if len(child_children) > 0:
        for c in child_children:
            aspects, indiv_aspects = extractCONJ(c, chunk, aspects, indiv_aspects, verb, skips, sent)

    # otherwise extract the aspects
    # once at the bottom, we check to see if our criteria are met then add aspects
    # then we will come back up the tree
    # MAYBE I ONLY EXTRACT ADDITIONAL NOUN CHUNKS IN THE RECURSIVE FUNCTION INSTEAD?
    if child.dep_ in ['acomp',  'conj']:
        aspects, indiv_aspects = link2adj(child, chunk, aspects, indiv_aspects, verb)
        aspects, indiv_aspects, _ = link2noun(child, chunk, aspects, indiv_aspects, verb, skips, sent)
            
    return aspects, indiv_aspects


# Main function that runs through a sentence, traverses the tree, and extracts the aspects
def extractAspects(sent):
    aspects = [] # aspect clauses
    indiv_aspects = [] # individual aspects
    skips = [] # noun chunks already reviewed
    for chunk in sent.noun_chunks:
        # skip if noun chunk was already reviewed or if the noun chunk is a pronoun
        if chunk.text in skips or any([word for word in chunk if word.text.lower() in pronouns]):
            continue
        if (chunk.root.dep_ == "nsubj") and (stem.stem(chunk.root.head.text) in link_verbs):
            verb = chunk.root.head
          
            # get children for each verb, removing core noun chunk and any punctuation
            children = extractChildren(verb, chunk)

            for child in children:
                if child.dep_ in ['acomp', 'dobj', "xcomp"]: # unsure about including dobj, but can leave for now
                    # create aspect if links to adjective
                    aspects, indiv_aspects = link2adj(child, chunk, aspects, indiv_aspects, verb)
                    
                    # create aspect if links to other noun chunk
                    aspects, indiv_aspects, skips = link2noun(child, chunk, aspects, indiv_aspects, verb, skips, sent)
                    
                    # if criteria are met, we check to see if there are any conjugations
                    # recursively call functions to extract conjugations
                    # since it is conjugations, we do not need to worry about skips, since it will have been captured already
                    child_children = extractChildren(child, chunk)
                    for c in child_children:
                        aspects, indiv_aspects = extractCONJ(c, chunk, aspects, indiv_aspects, verb, skips, sent)
      
    return aspects, set(indiv_aspects)

##########################################################
# Run the extraction 
##########################################################
# for each set of sentences, extract the aspects
# create key value pair with asin and reviews
# I cannot save just the sentence when it is part of a doc, so I need to parse sentence, convert to string, and re parse
def splitSentences(review):
    review = nlp(review)
    return [str(sent) for sent in review.sents]

def mapExtraction(sentences):
    aspects = []
    indiv_aspects = set()
    
    sentences = nlp(sentences)
    
    for sent in sentences.sents:
        extracted_aspects, extracted_indiv_aspects = extractAspects(sent)

        # add to running list
        aspects += extracted_aspects
        indiv_aspects = indiv_aspects.union(extracted_indiv_aspects)

    return (aspects, list(indiv_aspects))


# run the process
def main(subset_df, category):
          
    reviews = subset_df.rdd.map(list).map(lambda x: ([int(x[6]), x[0], x[1], x[4], x[5], x[2]],  x[2]))
    reviews2 = reviews.mapValues(lambda x: mapExtraction(x))\
        .filter(lambda x: len(x[1][0]) > 0)
    
    
    ##########################################################
    # Export
    ##########################################################
    print("prepping export")
    aspect_df = reviews2.map(lambda x: Row(idx = x[0][0], asin = x[0][1], reviews = x[0][5], 
                                          scores = x[0][2], categories = x[0][3], main_cat =x[0][4],
                                          indiv_aspects =x[1][1])).toDF()
    
    
    # I need to avoid joins at all costs - will need to revisit appending the column 
    #to_export = df.join(aspect_df, 'idx', how = "inner")
    print("exporting") 
    aspect_df.write.parquet(base_path + "/extracted_aspects/Extracted " + category + ".parquet")


categories = ['All Electronics', 'Electronics','Home & Kitchen', 'Home Improvement', 'Books',
 'Industrial & Scientific', 'Health & Personal Care',  'Camera & Photo', 'Software',
 'Cell Phones & Accessories','Arts', 'Computers', 'Office & School Supplies',
 'GPS & Navigation','Car Electronics', 'Toys & Games', 'Kitchen & Dining','Tools & Home Improvement',
 'Baby', 'All Beauty', 'Baby Products', 'Microsoft', 'Office Products','Appliances',
'Amazon Fashion','Patio', 'Sports & Outdoors', 'Video Games',
 'Clothing', 'Musical Instruments', 'MP3 Players & Accessories', 'Automotive', 'Luxury Beauty']

import time, datetime

df = spark.read.parquet(base_path + "/Amazon Product Reviews Raw Cleaned.parquet")

for i, c in enumerate(categories[:1]):
    print("starting ", c)
    print("extracting Aspects")
    print(df.where(F.col("main_cat") == c).count())
    
    start = time.time()
    print(datetime.datetime.now())# for now limiting to 1 mil
    df2 = df.where(F.col("main_cat") == c).limit(1000000)\
        .repartition(100)
    main(df2, c)
    print(time.time() - start)

    