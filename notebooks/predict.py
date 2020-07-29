import argparse
import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import string

def fileArgparser():
	parser = argparse.ArgumentParser(description='file argument for sentiment prediction')
	parser.add_argument("--f", default=1, required=True, type=str, help="full path location to the csv file with tweets")
	args = parser.parse_args()
	a = args.f
	return a

def preprocessing(text):
	text = str(text).lower()
	text = re.sub(r'(?<=@)\w+', '', text)
	text = re.sub('\[.*?\]', '', text)
	text = re.sub('https?://\S+|www\.\S+', '', text)
	text = re.sub('<.*?>+', '', text)
	text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
	text = re.sub('\n', '', text)
	text = re.sub('\w*\d\w*', '', text)
	return text

def tokenization(text):
    text = re.split('\W+', text)
    return text


def remove_stopwords(text):
	nltk.download('stopwords')
	stopword = nltk.corpus.stopwords.words('english')
	text = [word for word in text if word not in stopword]
	return text


def stemming(text):
	ps = nltk.PorterStemmer()
	text = [ps.stem(word) for word in text]
	return text

def wordToCountVec(df):
	vectorizer = pickle.load(open("data/model/vector.pickel", 'rb'))
	features = vectorizer.transform(
	    df
	)
	features_nd = features.toarray() # for easy usage
	return features_nd


def main():
	filePath = fileArgparser()
	df = pd.read_csv(filePath, sep='\t')
	df['clean_text'] = df['full_text'].apply(lambda x:preprocessing(x))
	df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
	df['tokenized'] = df['clean_text'].apply(lambda x: tokenization(x.lower()))
	df['stopwords_removed'] = df['tokenized'].apply(lambda x: remove_stopwords(x))
	df['stemmed'] = df['stopwords_removed'].apply(lambda x: stemming(x))
	df["stemmed_joined"]= df["stemmed"].str.join(" ") 


	loaded_model = pickle.load(open("data/model/finalized_model.sav", 'rb'))

	features_nd = wordToCountVec(df['stemmed_joined'])
	y_pred = loaded_model.predict(features_nd)

	df['prediced_sentiment'] = y_pred

	df = df[['full_text', 'prediced_sentiment' ]]
	df.to_csv('data/output/output.csv')

	#TODO: Add word cloud output as well
	print("Prediction is done - word cloud and predicted output is present now")

    

if __name__ == "__main__":
    main()