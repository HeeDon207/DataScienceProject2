import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    messages_filepath: file path of message data
    categories_filepath: file path of categories

    First we load data from disater_messager.csv and disater_categories.csv
    Then we merge two data.
    Finally return dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    df: dataframe
    rename column
    remove duplicated
    convert related to value 0 and 1

    """
    categories = df['categories'].str.split(';', expand=True)

    # select first row
    row = categories.iloc[0]
    # print(row)

    # extract new columns names
    category_colnames = [r.split('-')[0] for r in row]
    # rename
    categories.columns = category_colnames

    for col in categories:
        categories[col] = categories[col].str[-1]
        categories[col] = categories[col].astype(int)

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    df: dataframe
    database_filename: name of database
    save data to sqlite database
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisaterResponse', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
