#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ml_algorithms.py handles the parsing and uploading of files to the server
"""
__author__      = "Erik Eklund"
__copyright__   = "Copyright 2016, Planet Earth"
__license__     = "MIT"
__version__     = "0.1"
__status__      = "Prototype"

import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug import secure_filename
import numpy as np
import pandas as pd  
import json
from modules import ml_algorithms as mla


app = Flask(__name__)

# Maximum allowed fileSize 2x10 MB
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

# Set to False when in production
app.debug = False

# Path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# Extension that can be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'xlsx', 'csv'])

# Visiting the endpoint /wdc will trigger the function wdc_root
@app.route('/wdc')
def wdc_root():
    return app.send_static_file('wdc_ml_index.html')
    
   
# Return if the file is allowed or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


# Function for reading file into a pandas dataframe
def parse_file(input_filename):
    filename, file_extension = os.path.splitext(input_filename)
    if file_extension == '.txt' or file_extension == '.csv':
        df = pd.read_csv(input_filename, sep=';', decimal='.', encoding = "ISO-8859-1", header=0)
    elif file_extension == '.xlsx':
        df = pd.read_excel(input_filename)
    else:
        df = None

    return df


def get_uploaded_file(filename):
    filename_fnuttless = filename.replace('"', '')
    filename = 'uploads/' + filename_fnuttless
    return (filename)


@app.route('/uploadFileClassification', methods=['POST'])
def upload_file_classification():
    # Get the name of the uploaded training file
    file1 = request.files['trainfile']
    # Check if the file is one of the allowed types/extensions
    if file1 and allowed_file(file1.filename):
        # Make the filename safe by removing unsupported chars
        trainingfilename = secure_filename(file1.filename)
        # Move the file to the upload folder we set above
        file1.save(os.path.join(app.config['UPLOAD_FOLDER'], trainingfilename))

    # Do the same for the score file
    file2 = request.files['scorefile']

    if file2 and allowed_file(file2.filename):
        scorefilename = secure_filename(file2.filename)
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], scorefilename))

    # Set path for the uploaded files
    trainingfilename = 'uploads/' + trainingfilename
    scorefilename = 'uploads/' + scorefilename

    # Read files into a pandas dataframe
    my_train_df = parse_file(trainingfilename)
    my_score_df = parse_file(scorefilename)

    # Create list to hold all column names in dataframe
    my_score_colnames = list(my_score_df.columns.values)
    # Add column names for model validation metrics
    my_score_colnames.extend(['yProbaTrue', 'accuracy', 'recall', 'precision',
                             'f1', 'auc'])
    # Create empty list to hold the datatypes from the dataframe
    my_score_datatypes = []

    # Get all datatypes of the score file dataframe
    list_dataypes = list(my_score_df.dtypes)

    # loop the datatypes and set (this is info tableau.connectionData needs)
    for col in list_dataypes:
        if col == 'object':
            my_score_datatypes.append('string')
        else:
            my_score_datatypes.append('float')

    # Append columns for the five validation metrics and the yProbaTrue to my_score_datatypes
    my_score_datatypes.extend(['float', 'float', 'float', 'float',
                               'float', 'float'])
    # Check if yPred is null
    ypredcol = [col for col in my_train_df.columns.values
                if 'YPRED' in col.upper()]

    # Validate. For improved  validation put the following code in a function or a class
    if not ypredcol:
        return (json.dumps(['error', 'There is no target (Y) variable in your data. In your training file type "ypred" at the start of you target (response) variable (Y)']))
    elif len(np.unique(my_train_df[ypredcol[0]])) != 2:
        return (json.dumps(['error', 'The target (Y) variable has more than two levels. Multiclass classification is not yet supported.']))
    else:
        # Return lists with column names and column datatypes
        return (json.dumps([my_score_colnames, my_score_datatypes]))


@app.route('/uploadFileOLS', methods=['POST'])
def upload_file_ols():
    # Get the name of the uploaded file
    file1 = request.files['trainfile']
    # Check if the file is one of the allowed types/extensions
    if file1 and allowed_file(file1.filename):
        trainingfilename = secure_filename(file1.filename)
        file1.save(os.path.join(app.config['UPLOAD_FOLDER'], trainingfilename))


    file2 = request.files['scorefile']
    if file2 and allowed_file(file2.filename):
        scorefilename = secure_filename(file2.filename)
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], scorefilename))

    trainingfilename = 'uploads/' + trainingfilename
    scorefilename = 'uploads/' + scorefilename

    my_train_df = parse_file(trainingfilename)
    my_score_df = parse_file(scorefilename)

    my_score_colnames = list(my_score_df.columns.values)
    my_score_colnames.extend(['yPrediction', 'RSquare', 'mse'])

    my_score_datatypes = []
    list_dataypes = list(my_score_df.dtypes)

    for col in list_dataypes:
        if col == 'object':
            my_score_datatypes.append('string')
        else:
            my_score_datatypes.append('float')

    my_score_datatypes.extend(['float', 'float', 'float'])

    ypredcol = [col for col in my_train_df.columns.values if 'YPRED' in col.upper()]

    if not ypredcol:
        return (json.dumps(['error', 'There is no target (Y) variable in your data. In your training file type "ypred" at the start of your target variable.']))
    else:
        return (json.dumps([my_score_colnames, my_score_datatypes]))


@app.route('/uploadFileKmeans', methods=['POST'])
def upload_file_kmeans():

    file1 = request.files['trainfile']
    if file1 and allowed_file(file1.filename):
        trainingfilename = secure_filename(file1.filename)
        file1.save(os.path.join(app.config['UPLOAD_FOLDER'], trainingfilename))

    trainingfilename = 'uploads/' + trainingfilename
    my_cluster_df = parse_file(trainingfilename)

    my_cluster_colnames = list(my_cluster_df.columns.values)
    my_cluster_colnames.extend(['cluster'])
    my_cluster_datatypes = []

    list_dataypes = list(my_cluster_df.dtypes)
    for col in list_dataypes:
        if col == 'object':
            my_cluster_datatypes.append('string')
        else:
            my_cluster_datatypes.append('float')
    my_cluster_datatypes.extend(['float'])

    return (json.dumps([my_cluster_colnames, my_cluster_datatypes]))


@app.route('/callMLalgorithm', methods=['POST'])
# Function when file has been uploaded and myConnector.getTableData is called by clicking the 'Process File' button
def call_ml_algorithm():
    try:
        trainfilen = json.dumps(request.json['TrainfilNamn'])
        scorefilen = json.dumps(request.json['ScorefilNamn'])
        algo_chosen = json.dumps(request.json['algoChosen'])
        algo_chosen = algo_chosen.replace('"', '')
        trainfiledata = get_uploaded_file(trainfilen)
        scorefiledata = get_uploaded_file(scorefilen)

        if algo_chosen == 'Clustering(K-means)':
            ClusterDF = parse_file(trainfiledata)
            result_df = mla.runClustering(ClusterDF)

        elif algo_chosen == 'Logistic Regression':
            my_train_df = parse_file(trainfiledata)
            my_score_df = parse_file(scorefiledata)
            reg_param1 = json.dumps(request.json['regularisationParam1'])
            reg_param2 = json.dumps(request.json['regularisationParam2'])
            reg_param1 = reg_param1.replace('"', '')
            reg_param2 = reg_param2.replace('"', '')
            result_df = mla.runLogisticRegression(my_train_df, my_score_df, reg_param1, reg_param2)

        elif algo_chosen == 'OLS Linear Regression':
            my_train_df = parse_file(trainfiledata)
            my_score_df = parse_file(scorefiledata)
            result_df = mla.runOLSRegression(my_train_df, my_score_df,)

        elif algo_chosen == 'Random Forest':
            my_train_df = parse_file(trainfiledata)
            my_score_df = parse_file(scorefiledata)
            numbtrees_param = json.dumps(request.json['numbTrees'])
            splitcriteria_param = json.dumps(request.json['splitCriteria'])
            numbtrees_param = numbtrees_param.replace('"', '')
            splitcriteria_param = splitcriteria_param.replace('"', '')
            result_df = mla.runRandomForest(my_train_df, my_score_df, numbtrees_param, splitcriteria_param)

        else:
            pass

        # Remove uploaded files
        if trainfiledata == scorefiledata:
            os.remove(trainfiledata)
        else:
            os.remove(trainfiledata)
            os.remove(scorefiledata)

        # Create json
        my_json = result_df.to_json(orient='records')
        # return json with results to the Ajax call
        return (my_json)

    except:
        print('An error occcured - A version 0.2 should include better error handling on the server side')

if __name__ == "__main__":
    app.run()
