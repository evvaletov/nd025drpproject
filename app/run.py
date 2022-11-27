import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar, Pie, Scatter
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    '''
    Tokenizes the input text.

        Parameters:
            text: The text to be tokenized

        Return:
            clean_tokens: Lemmatized and stripped tokens in the lower case
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # Data for the first plot
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    # Data for the second plot
    df['message_type'] = df.apply(
        lambda row: "aid" if row.aid_related == 1 else (
            "infrastructure" if row.infrastructure_related == 1 else (
                "weather" if row.weather_related == 1 else "other")), axis=1)
    df['message_type'] = df.apply(
        lambda row: "mixed" if row.aid_related +
        row.infrastructure_related +
        row.weather_related > 1 else row.message_type,
        axis=1)
    message_type_counts = df.groupby('message_type').count()['message']
    message_type_names = list(message_type_counts.index)
    # Data for the third plot
    aid_related_offer_counts = df[df.offer == 1][['medical_help',
                                                  'medical_products',
                                                  'search_and_rescue',
                                                  'security',
                                                  'military',
                                                  'child_alone',
                                                  'water',
                                                  'food',
                                                  'shelter',
                                                  'clothing',
                                                  'money']].astype(bool).sum(axis=0)
    aid_related_offer_names = list(aid_related_offer_counts.index)
    aid_related_request_counts = df[df.request == 1][['medical_help',
                                                      'medical_products',
                                                      'search_and_rescue',
                                                      'security',
                                                      'military',
                                                      'child_alone',
                                                      'water',
                                                      'food',
                                                      'shelter',
                                                      'clothing',
                                                      'money']].astype(bool).sum(axis=0)
    aid_related_request_names = list(aid_related_request_counts.index)

    # Third plot specification

    # create visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=message_type_names,
                    y=message_type_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Broad Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Type"
                }
            }
        },
        {
            'data': [
                Scatter(
                    x=aid_related_request_counts,
                    y=aid_related_request_names,
                    name="Requests",
                    mode="markers",
                    marker=dict(
                        color="rgba(156, 165, 196, 0.95)",
                        line=dict(
                            color="rgba(156, 165, 196, 1.0)",
                            width=1,
                        ),
                        symbol="circle",
                        size=16
                    )
                ),
                Scatter(
                    x=aid_related_offer_counts,
                    y=aid_related_offer_names,
                    name="Offers",
                    mode="markers",
                    marker=dict(
                        color="rgba(204, 204, 204, 0.95)",
                        line=dict(
                            color="rgba(217, 217, 217, 1.0)",
                            width=1,
                        ),
                        symbol="circle",
                        size=16
                    )
                )
            ],

            'layout': {
                'title': 'Aid related offers and requests',
                'yaxis': {
                    'title': "Type"
                },
                'xaxis': {
                    'title': "Count"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
