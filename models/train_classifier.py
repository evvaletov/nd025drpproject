import sys
import re
import pickle
import os.path
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    '''
    Loads the SQLite database with messages and categories and returns the X and Y for machine learning model training.
    
        Parameters:
            database_filename: Filename of the SQLite database

        Returns:
            X, Y: The X and Y Pandas dataframes for machine learning model training
            Y.columns: The column names of the Y dataframe
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("messages", con=engine)
    df = df.reset_index()
    X = df['message']
    Y = df.drop(['level_0','index','id','message','original','genre'], axis=1)
    return X, Y, Y.columns


def tokenize(text):
    '''
    Tokenizes the input text.
    
        Parameters:
            text: The text to be tokenized

        Return:
            clean_tokens: Lemmatized and stripped tokens in the lower case
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    Builds a machine learning model for message classification.
    
        Parameters:
            (nothing)

        Returns:
            df: The built machine learning model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__criterion' : ['gini', 'entropy'],
        'tfidf__norm' : ['l1', 'l2']
    }
    return GridSearchCV(pipeline, parameters)


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Prints the classification report for each message category using the trained machine learning model.
    
        Parameters:
            model: The machine learning model
            X_test, Y_test: The test X, Y dataframes for the classification report
            category names: The message category names

        Returns:
            (nothing)
    '''
    Y_pred = model.predict(X_test)
    for j, column in enumerate(category_names):
        print("Classification report for category \""+column+"\"")
        print(classification_report(Y_test[column], Y_pred[:,j])) 
    

def save_model(model, model_filepath):
    '''
    Saves the trained machine learning model into a picke file.
    
        Parameters:
            model: The machine learning model
            model_filepath: The path for the saved machine learning model

        Return:
            (nothing)
    '''
    pickle.dump(model, open(model_filepath,"wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
