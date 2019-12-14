# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 22:18:26 2019

Subset data so I can work with it locally before sending to cloud

@author: Austin Bell
"""

import pandas as pd
import json
import re

base_path = "C:/Users/Austin Bell/Documents/Columbia/Big Data/project"
reviews = base_path + "/Data/item_dedup.json"
metadata = base_path + "./Data/metadata.json"

n_reviews = 18000000

# create small review dataset
with open(reviews, "r") as f:
    asin = []
    review_text = []
    scores = []
    
    for i, line in enumerate(f):
        if i < 15000000:
            continue
        if i == n_reviews:
            break
    
        line = json.loads(line)
        
        asin.append(line['asin'])
        review_text.append(line['reviewText'])
        scores.append(line['overall'])
        

    df = pd.DataFrame()
    df['asin'] = asin
    df['reviews'] = review_text
    df['scores'] = scores


# init empty category variable
df['cats'] = ''

unique_asins = df.asin.unique()
asin_dict = {}
for a in unique_asins:
    asin_dict[a] = ""

# get the topics for each of these reviews
with open(metadata, "r", encoding = "utf-8") as f:
    
    for i, line in enumerate(f):

        # json does not appear to be working
        line = re.sub("[\'|\\\']", "\"", line)
        asin = str(re.search("\"asin\": \"(.*?[0-9|A-Z])\"", line).group(1))
        
        tru_cat = re.search("\"categories\": (\[\[.*\]\])", line)
        if tru_cat:
            cat = str(tru_cat.group(1))
             
            try:
                asin_dict[asin] = cat
            except:
                pass
            #if asin in unique_asins:    
            #    df.loc[df['asin'] == asin, "cats"] = cat
                
        
        if i % 100000 == 0:
            print(i)
            

new_dict = {}
for i, (key, value) in enumerate(asin_dict.items()):
    if value != "":
        new_dict[key] = value



df['cats'] = df['asin']
df["cats"] = df.cats.map(new_dict)


check = df[~df.cats.isnull()]
#check.cats.unique()

text = check[check.cats.str.contains("Electronics|Games|Software|Phones|Video|Sports")]
len(text.reviews)

text2 = pd.concat([text, saved_text], axis = 0)

saved_text = text
saved_again = text


df.to_csv(base_path + "/Data/Amazon_products_ElectronicsAndToys.csv", index = False)
