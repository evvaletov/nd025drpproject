import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads messages and categories data and returned the merged data in a Pandas dataframe.
    
        Parameters:
            messages_filepath: path to the messages CSV file
            categories_filepath: path to the categories CSV file

        Returns:
            df: Pandas dataframe with loaded messages and categories
    '''
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    messages.drop_duplicates(subset='id', inplace=True)
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    categories.drop_duplicates(subset='id', inplace=True)
    # Merge datasets
    df = pd.merge(messages, categories, on="id", how='inner')
    # create a dataframe of the 36 individual category columns
    categories = categories.categories.str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(row.str.split("-", expand=True)[0])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # convert category values to binary
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-",expand=True)[1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int).ge(0.5).astype(int)
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df.reset_index(inplace=True)
    categories.reset_index(inplace=True)
    df = pd.concat([df,categories], axis=1)
    return df


def clean_data(df):
    '''
    Cleans the data by dropping duplicate rows in the supplied Pandas dataframe.
    
        Parameters:
            df: Pandas dataframe to be cleaned

        Returns:
            df: Cleaned Pandas dataframe
    '''
    df.drop_duplicates(inplace=True)
    assert sum(df.duplicated().astype(int))==0
    return df


def save_data(df, database_filename):
    '''
    Saves the Pandas dataframe with messages and categories to a SQLite database.
    
        Parameters:
            df: Pandas dataframe to be saved
            database_filename: Filename of the output SQLite database

        Returns:
            (nothing)
    '''
    engine = create_engine("sqlite:///"+database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
