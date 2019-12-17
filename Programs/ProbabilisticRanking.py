# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:22:51 2019

Run the Probabilistic Ranking Algorithm across all categories

@author: Austin Bell
"""

###############################################################
# Import Packages and Initialize Spark
###############################################################

# additional packages
import numpy as np
import pandas as pd

from numpy.linalg import inv, pinv
from scipy.linalg import sqrtm
from sklearn.datasets import make_spd_matrix
import math
import gcsfs

# pyspark
from pyspark import SparkConf, SparkContext
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# init spark
spark = SparkSession \
    .builder \
    .appName("ranking") \
    .getOrCreate()
    
    
###############################################################
# Data Prep Functions
###############################################################
# for each review, create an opinion vec
def create_vec(r) :
    vec = [0]*len(map_aspects)
    asps = r[0].split('<split>')
    ops = [float(x) for x in str(r[1]).split('<split>')]
    for i in range(len(asps)):
        vec[map_aspects[asps[i]]] = ops[i]
    return vec, r[2]

# convert to rdd, then collapse aspects by review
def genOpinionVecs(df):
    row = df.rdd.map(list).map(lambda r: (int(r[1]), (r[10], r[12], r[7],r[8],  r[6])))
    row = row.reduceByKey(lambda a, b: (a[0]+'<split>'+b[0],
                                        str(a[1])+'<split>'+str(b[1]), 
                                        a[2],
                                        str(a[3])+'<split>'+str(b[3]), a[4]))

    # gen opinion vectors
    row = row.mapValues(lambda r : create_vec(r))
    opinion_vecs = row.collect()
    
    return opinion_vecs

# create a dataframe of aspect frequencies in the same order as the opinion vecs
def computeFrequencies(df):
    aspect_freqs = df.groupby(F.col("key_aspect")).count().toPandas()
    total_aspect_counts = aspect_freqs['count'].sum()
    aspect_freqs['frequencies'] = aspect_freqs['count'] / total_aspect_counts
    return aspect_freqs




###############################################################
# Probabilistic Ranking Functions
###############################################################
def compute_w(opinion_vec, mu, sigma_sq, inv_cov, score):
    # compute the first segment of w
    opinion_var = np.dot(opinion_vec, opinion_vec.T) / sigma_sq
  
    try:
        first_segment = inv(opinion_var + inv_cov)
    except:
        first_segment = pinv(opinion_var + inv_cov)

    # compute the second segment of w
    score_var = np.dot(score, opinion_vec) / sigma_sq
    inv_cov_mu = np.matmul(inv_cov, mu)
    second_segment = score_var + inv_cov_mu

    # return w 
    W = np.matmul(first_segment, second_segment)
    return W

# compute our new mu
def compute_mu(W, mu, cov, phi, identity, total_reviews):
    # first step 
    freq_cov = total_reviews * pinv(cov)
    phi_identity = phi*identity
    first_segment  = pinv(freq_cov + phi_identity)

    # second step 
    cov_w = np.matmul(pinv(cov),np.sum(W, axis = 0))
    phi_mu = phi*mu
    second_segment = cov_w + phi_mu

    # compute new_mu
    new_mu = np.matmul(first_segment, second_segment) 

    return new_mu

# compute our new covariance matrix
# input is the new_mu
def compute_cov(W, mu, phi, identity, total_reviews):
    # in covariance calculations - the authors assume that we know to do it by column and so it is not explicitly written in formulas 
    w_mu_var = np.dot((W-mu).T, (W-mu).conj())/phi
    r_phi_identity = ((total_reviews - phi)/(2*phi))**2 *identity
    first_segment = sqrtm(w_mu_var + r_phi_identity).real

    second_segment = ((total_reviews - phi)/(2*phi)) *identity

    new_cov = first_segment - second_segment

    return new_cov

def compute_sigma(W, opinion_vectors, total_reviews):
    var = 0
    for i in range(len(W)):
        review = opinion_vectors[i][1]
        score = review[1]
        opinion_vec = review[0]
        weighted_opinions = np.dot(W[i].T, opinion_vec)
        var += (score - weighted_opinions)**2
    
    new_sigma_sq = var/total_reviews
    return new_sigma_sq



def EMRanking(opinion_vectors, aspect_freqs, phi=100, tol=.01):

    # initialize parameters
    mu = np.array(aspect_freqs.frequencies)
    sigma_sq = np.random.random()
    cov = make_spd_matrix(len(aspect_freqs))
    identity = np.identity(len(cov))
    total_reviews = len(opinion_vectors)


    err = math.inf
    j=0
    while err > tol:
        # Expectation Step

        # init W 
        W = np.zeros((total_reviews, len(aspects)))

        inv_cov = pinv(cov)
        for i, r in enumerate(opinion_vectors):
            review = r[1]
            opinion_vec = np.array(review[0])
            score = review[1]
            w = compute_w(opinion_vec, mu, sigma_sq, inv_cov, score)
            W[i] = w
            if i % 50000 == 0:
                print(i)

        # now maximization step 

        # mu 
        new_mu = compute_mu(W, mu, cov, phi, identity, total_reviews)

        # covariance 
        new_cov = compute_cov(W, new_mu, phi, identity, total_reviews)

        # sigma squared
        new_sigma_sq = compute_sigma(W, opinion_vectors, total_reviews)

        # change from old
        print("Current change in Mu: ", str(abs(np.sum(mu-new_mu))))
        if j != 0:
            err = abs(np.sum(W-old_W))
            print("Current change in W: ", str(err))

        # rename vars
        mu = new_mu
        cov = new_cov
        sigma_sq = new_sigma_sq
        old_W = W
        j+=1
    print("Successfully Converged")
    
    return W

if __name__ == "__main__":

    # full run through 
    # Initialize parameters
    phi = 100
    tol = 1
    
    categories = ['All Electronics', 'Electronics','Home & Kitchen', 'Home Improvement', 
                 'Industrial & Scientific', 'Health & Personal Care',  'Camera & Photo', 'Software',
                 'Cell Phones & Accessories','Arts', 'Computers', 'Office & School Supplies',
                 'GPS & Navigation','Car Electronics', 'Toys & Games', 'Kitchen & Dining','Tools & Home Improvement',
                 'Baby', 'All Beauty', 'Baby Products', 'Office Products','Appliances',
                'Amazon Fashion','Patio', 'Sports & Outdoors', 'Video Games',
                 'Clothing', 'Musical Instruments', 'Automotive', 'Luxury Beauty']
    
    
    base_path = "gs://core_bucket_abell/aspect_ranking"
    for k, cat in enumerate(categories[28:]):
        print("Starting: " + cat)
        print("Number: " + str(k) + " out of " + str(len(categories)-28))
        
        # update with path to cedric's drive
        df = spark.read.parquet(base_path + "/sentiment_analysis/Sentiments " + cat + ".parquet")
    
    
        # create list and then dictionary of unique aspects within the dataset
        aspects = [r['key_aspect'] for r in df.select("key_aspect").distinct().collect()]
        map_aspects = {aspects[i] : i for i in range (len(aspects)) }
        
        print("Generating Opinion Vectors")
        # generate an rdd with each opinion vector
        opinion_vectors = genOpinionVecs(df)
        
        print("Computing Aspect Frequencies")
        # create a dataset of aspect frequencies
        aspect_freqs = computeFrequencies(df)
        
        print("Initializing Ranking")
        # run the ranking algorithm
        W = EMRanking(opinion_vectors, aspect_freqs, phi=phi, tol=tol)
        
        print("Finalizing and Exporting")
        # generate output
        aspect_freqs['aspect_value'] = W[0]
        aspect_freqs['freq_weighted_value'] = aspect_freqs['aspect_value'] * aspect_freqs['frequencies'] * 100
        aspect_freqs.to_csv(base_path + "/ranking/Weights " + cat + ".csv", index = False)
          
    
    
