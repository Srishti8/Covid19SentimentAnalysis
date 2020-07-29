import argparse
import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
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
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
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
    vectorizer = pickle.load(open("../data/model/vector.pickel", 'rb'))
    features = vectorizer.transform(df)
    features_nd = features.toarray() # for easy usage
    return features_nd


def find_frequencies(tokens: [str]):
    frequencies = {}
    for token in tokens:
        frequency = frequencies.get(token, 0) + 1
        frequencies[token] = frequency
    return frequencies


def main():
    filePath = fileArgparser()
    df = pd.read_csv(filePath, sep='\t')
    df['clean_text'] = df['full_text'].apply(lambda x : preprocessing(x))
    df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
    df['tokenized'] = df['clean_text'].apply(lambda x: tokenization(x.lower()))
    df['stopwords_removed'] = df['tokenized'].apply(lambda x: remove_stopwords(x))
    df['stemmed'] = df['stopwords_removed'].apply(lambda x: stemming(x))
    df["stemmed_joined"] = df["stemmed"].str.join(" ")

    loaded_model = pickle.load(open("../data/model/finalized_model.sav", 'rb'))

    features_nd = wordToCountVec(df['stemmed_joined'])
    y_pred = loaded_model.predict(features_nd)
    df['prediced_sentiment'] = y_pred

    cloud_df = df.groupby(['prediced_sentiment'], as_index=False).agg({'stopwords_removed': sum})
    cloud_df['frequencies'] = cloud_df['stopwords_removed'].apply(lambda x: find_frequencies(x))
    for _, row in cloud_df.iterrows():
        sentiment = row['prediced_sentiment']
        frequencies = row['frequencies']
        wc = WordCloud().generate_from_frequencies(frequencies)
        plt.figure()
        plt.imshow(wc)
        plt.axis('off')
        plt.savefig(f"../data/output/{sentiment}-wordcloud.png", format="png")
    
    df = df[['full_text', 'prediced_sentiment' ]]
    df.to_csv('../data/output/output.csv')
    

if __name__ == "__main__":
    main()