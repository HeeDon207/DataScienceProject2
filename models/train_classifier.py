import sys
import nltk
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    database_filepath: file path of the database

    First, create engine, then read sql table from database
    return : X, y, category_name
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisaterResponse', con=engine)
    X = df.message
    y = df.drop(columns=['id', 'message', 'original', 'genre'])
    # print(y.columns)
    category_names = y.columns.tolist()

    return X, y, category_names


def tokenize(text):
    """
    text: text (string)
    return: token cleaned

    First, we tokenize text.
    Then using WordNetLemmatizer to lemmatize token from text
    Finally return the token cleaned
    """
    tokens = word_tokenize(text)
    lemma = WordNetLemmatizer()
    clean_tokens = [lemma.lemmatize(i).lower().strip() for i in tokens]
    return clean_tokens


def build_model():
    """
    Builds classifier model by using GridSearchCV.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [4, 6, 8],
        'clf__estimator__max_depth': [4, 6, 8],
    }

    model = GridSearchCV(pipeline, parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    model: trained model
    X_test: test data
    Y_test: test data
    category_names

    First, using trained model to predict the X_test data.
    Then evaluate between Y_test, and y_predict.
    print the classification report
    """
    y_pred = model.predict(X_test)
    for index, column in enumerate(category_names):
        print(column, classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """ 
    model: trained model
    model_filepath: path to save model
    finally dump model to model_filepath
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
