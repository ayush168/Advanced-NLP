# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 19:52:37 2019

@author: Nikhil B. Mankame
@author: Ayush
"""

import  pandas as pd
import en_core_web_sm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.svm import LinearSVC

# Load data from the Input File 
load_data = pd.read_table('combined.txt', encoding = "ISO-8859-1")

# Convert the data loaded from the Input File to a list
data_list = [load_data]

# If the length of the data list is greater than 0 then append the column names "Review" : For consumer reviews and "Label" : 1 - Positive Review, 0 - Negative Review
if len(data_list) > 0:
    data_list[0].columns = ["Review","Label"]

def my_tokenizer(sentence):
    doc = nlp(sentence)
    mytokens = [token.orth_ for token in doc if not token.is_punct]
    mytokens = [item.lower() for item in mytokens]
    filtered_sentence = []
    for word in mytokens:
        text = nlp.vocab[word]
        if text.is_stop == False:
            filtered_sentence.append(word)
    return filtered_sentence

# Load the pre-trained statiscal English model
nlp = en_core_web_sm.load()

# Perform Vectorization
vectorizer = CountVectorizer(tokenizer = my_tokenizer, ngram_range=(1,1)) 

# Perform Tfid based Vectorization
#tfvectorizer = TfidfVectorizer(tokenizer = my_tokenizer)

# Start the Data Splitting Process
X = load_data["Review"]
ylabels = load_data["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)

class Predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        transformed_data = []
        for word in X:
            transformed_data.append(word.strip().lower())
        return transformed_data
    def fit(self, X, y, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

classifier = LinearSVC()

# Create the pipeline to clean, tokenize, vectorize, and classify the data using "Count Vectorizor"
pipe = Pipeline([("cleaner", Predictors()), ('vectorizer', vectorizer), ('classifier', classifier)])
    
# Fitting the Data
pipe.fit(X_train,y_train)

print("Accuracy: ",pipe.score(X_test,y_test))

# Load the Test Data
load_tweets_list = pd.read_csv('TrumpTweetsArchive.csv', encoding = "ISO-8859-1" )

# Remove the Columns that are not required
load_tweets_list = load_tweets_list.drop(['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6'], axis = 1)
# Convert the Test Data to a List format
tweets_list = load_tweets_list.values.tolist()

tweet_flat = []
for sublist in tweets_list:
    for item in sublist:
        tweet_flat.append(item)

predictions = pipe.predict(tweet_flat)

# Storing the obtained output along with the original tweets in a dataframe
dataframe = pd.DataFrame({'tweet': tweet_flat[:],
                        'prediction':predictions[:]})

# Exporting the dataframe to a csv file
csv_export = dataframe.to_csv ('result.csv', index = None, header=True)