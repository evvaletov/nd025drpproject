# Disaster Response Pipeline Project
# nd025drpproject

### Project summary:
The project consists of three parts: an ETL pipeline that loads disaster-related messages and categories and stores the cleaned data in a database, an ML pipeline that trains and saves a message classifier, and an app that gives several summary plots for the messsages in the database and can classify user-given messages.

### Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Main Files:

data/disaster_messages.csv: a CSV file with disaster-related messages
data/disaster_categories.csv: a CSV file with categorisation of the messages
data/process_data.py: a Python program to run the ETL pipeline that cleans the data and stores it in the database
data/DisasterResponse.db: a SQLite database with the combined and cleaned message and category data
models/train_classifier.py: a Python program that trains and saves a ML classifier for message classification
models/classifier.pkl: the saved ML classifier produced by train_classifier.py
app/app.py: a Flask app that shows several summary plots for the messages in the SQLite database and can classify user-giver messages
