# Disaster Response Pipeline Project

### Instructions:

In this project, I've learned and built on my data engineering skills to expand my opportunities and potential as a data scientist. In this project, I'll apply these skills to analyze disaster data from Appen to build a model for an API that classifies disaster messages.

You created a machine learning pipeline to categorize these events so that I can send the messages to an appropriate disaster relief agency.

### File in the repository

```
app
| - template
| |- master.html
| |- go.html
|- run.py
data
|- disaster_categories.csv
|- disaster_messages.csv
|- process_data.py
|- DisasterResponse.db
models
|- train_classifier.py
|- classifier.pkl
README.md
```

1. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3000/
