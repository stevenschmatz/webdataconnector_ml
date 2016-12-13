#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ml_algorithms.py contains the scikit-learn machine learning classes
"""
__author__      = "Erik Eklund"
__copyright__   = "Copyright 2016, Planet Earth"
__license__     = "MIT"
__version__     = "0.1"
__status__      = "Prototype"

import numpy as np
import pandas as pd
import ctypes  

def Mbox(title, text, style):
    ctypes.windll.user32.MessageBoxA(0, text, title, style)


def runLogisticRegression(train_df, score_df, reg_param1, reg_param2):

    # PREPARE TRAINING DATA
    ypredcol = [col for col in train_df.columns.values if 'YPRED' in col.upper()]
    Xcols = [col for col in train_df.columns if col not in ypredcol and 'NOTMODEL' not in col.upper()]

    # Set y to the first occurrence in list
    y = train_df[ypredcol[0]]
    # Convert character columns to dummy variables
    X = train_df[Xcols]
    cols = X.columns
    num_cols = X._get_numeric_data().columns
    char_cols = list(set(cols) - set(num_cols))

    for col in char_cols:
        if len(X[col].unique()) <= 20:
            dummy = pd.get_dummies(X[col], prefix='dm'+col)
            column_name = X.columns.values.tolist()
            column_name.remove(col)
            X = X[column_name].join(dummy)
        else:
            if col in X.columns:    # If more than 20 distinct values then delete
                del X[col]

    from scipy.stats import zscore
    # Standardize (Z-score normalize) all continuous variables
    for col in X:
        if len(X[col].unique()) > 2:  # standardize only non-dummy variables
            col_zscore = 'z_' + col
            # X[col_zscore] = (X[col] - X[col].mean())/X[col].std()
            X[col_zscore] = zscore(X[col])
            del X[col]

    # Fill missing values with 0 = the mean in the z-normalize data
    # Obviously missing values can be handled in many different ways
    X.fillna(0, inplace=True)

    # PREPARE DATA TO SCORE
    y_score_predcol = [col for col in score_df.columns.values if 'YPRED' in col.upper()]
    X_score_cols = [col for col in score_df.columns if col not in y_score_predcol and 'NOTMODEL' not in col.upper()]

    XScore = score_df[X_score_cols]
    score_cols = XScore.columns
    score_num_cols = XScore._get_numeric_data().columns
    score_char_cols = list(set(score_cols) - set(score_num_cols))

    # Convert character columns to dummy variables
    for col in score_char_cols:
        if len(XScore[col].unique()) <= 20:
            score_dummy = pd.get_dummies(XScore[col], prefix='dm'+col)
            score_column_name = XScore.columns.values.tolist()
            score_column_name.remove(col)
            XScore = XScore[score_column_name].join(score_dummy)
        else:
            if col in XScore.columns:       # If more than 20 distinct values then delete
                del XScore[col]

    # Standardize (Z-score normalize) all continuous variables
    for col in XScore:
        if len(XScore[col].unique()) > 2:     # standardize only non-dummy variables
            col_zscore = 'z_' + col
            # X[col_zscore] = (X[col] - X[col].mean())/X[col].std()
            XScore[col_zscore] = zscore(XScore[col])
            del XScore[col]

    # Fill missing values with 0 = the mean in the z-normalize data
    # Obviously missing values can be handled in many different ways
    XScore.fillna(0, inplace=True)

    # GET ALL MATCHING COLUMNS IN X and XScore
    matching_cols = []
    for X_score_col in XScore.columns.values:   # for loop XScoreCols
        match_col = ""
        match_col = [col for col in X.columns.values if X_score_col == col]  # Check if X Train Col exists in X Score Col
        matching_cols.extend(match_col)

    # Set dataframes to include only matching columns
    X = X[matching_cols].sort_index()
    XScore = XScore[matching_cols].sort_index()

    # INSTANTIATE LOGISTIC REGRESSION MODEL
    from sklearn.linear_model import LogisticRegression
    if reg_param2 == 'Y':   # User has chosen Grid Search
        from sklearn.grid_search import GridSearchCV
        # from sklearn.grid_search import RandomizedSearchCV
        param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
        # param_grid = {'C': [1.00000000e-04,  1.00000000e+04]}
        gs = GridSearchCV(estimator=LogisticRegression(penalty=reg_param1), param_grid=param_grid, n_jobs=1, refit=True, scoring='accuracy', cv=5)
        gs.fit(X, y)
        logit_model = gs.best_estimator_

    else:  # User has NOT chosen Grid Search
        logit_model = LogisticRegression(penalty=reg_param1)

    # FIT THE MODEL
    logit_model.fit(X, y)

    # CROSS VALIDATION
    # Evaluate the model using 5-fold cross-validation
    from sklearn.cross_validation import cross_val_score
    accuracyScore = cross_val_score(logit_model, X, y, scoring='accuracy', cv=5)
    recallScore = cross_val_score(logit_model, X, y, scoring='recall', cv=5)
    precisionScore = cross_val_score(logit_model, X, y, scoring='precision', cv=5)
    f1Score = cross_val_score(logit_model, X, y, scoring='f1', cv=5)
    aucScore = cross_val_score(logit_model, X, y, scoring='roc_auc', cv=5)

    # SCORE THE DATAFRAME  score_df
    # Create new fields for the predicte value and validation metrics
    score_df['yProbaTrue'] = np.round(logit_model.predict_proba(XScore)[:, 1], decimals=3)
    score_df['accuracy'] = np.round(accuracyScore.mean(), decimals=3)
    score_df['recall'] = np.round(recallScore.mean(), decimals=3)
    score_df['precision'] = np.round(precisionScore.mean(), decimals=3)
    score_df['f1'] = np.round(f1Score.mean(), decimals=3)
    score_df['auc'] = np.round(aucScore.mean(), decimals=3)
    return score_df


def runRandomForest(train_df, score_df, numbtrees_param, splitcriteria_param):
    Mbox('running', 'RandomForest', 1)

    # PREPARE TRAINING DATA
    ypredcol = [col for col in train_df.columns.values if 'YPRED' in col.upper()]
    Xcols = [col for col in train_df.columns if col not in ypredcol and 'NOTMODEL' not in col.upper()]

    # Set y to the first occurrence in list
    y = train_df[ypredcol[0]]

    # Convert character columns to dummy variables
    X = train_df[Xcols]
    cols = X.columns
    num_cols = X._get_numeric_data().columns
    char_cols = list(set(cols) - set(num_cols))

    # Convert character columns to dummy variables
    for col in char_cols:
        if len(X[col].unique()) <= 20:
            dummy = pd.get_dummies(X[col], prefix='dm'+col)
            column_name = X.columns.values.tolist()
            column_name.remove(col)
            X = X[column_name].join(dummy)
        else:
            if col in X.columns:        # If more than 20 distinct values then delete
                del X[col]

    from scipy.stats import zscore
    # Standardize (Z-score normalize) all continuous variables
    for col in X:
        if len(X[col].unique()) > 2:  # standardize only non-dummy variables
            col_zscore = 'z_' + col
            X[col_zscore] = zscore(X[col])
            del X[col]

    # Fill missing values with 0 = the mean in the z-normalize data
    # Obviously missing values can be handled in many different ways
    X.fillna(0, inplace=True)

    # PREPARE DATA TO SCORE
    y_score_predcol = [col for col in score_df.columns.values if 'YPRED' in col.upper()]
    X_score_cols = [col for col in score_df.columns if col not in y_score_predcol and 'NOTMODEL' not in col.upper()]

    # Convert char col to dummys
    XScore = score_df[X_score_cols]
    score_cols = XScore.columns
    score_num_cols = XScore._get_numeric_data().columns
    score_char_cols = list(set(score_cols) - set(score_num_cols))

    for col in score_char_cols:
        if len(XScore[col].unique()) <= 20:
            score_dummy = pd.get_dummies(XScore[col], prefix='dm'+col)
            score_column_name = XScore.columns.values.tolist()
            score_column_name.remove(col)
            # print ('Removing  %s' % col)
            XScore = XScore[score_column_name].join(score_dummy)
            # print ('Adding %s' %list(dummy))
        else:
            if col in XScore.columns:   # check if column exists in df
                del XScore[col]
                # print ('Deleting %s' % col)

    # Standardize (Z-score normalize) all continuous variables
    for col in XScore:
        if len(XScore[col].unique()) > 2:  # standardize only non-dummy variables
            col_zscore = 'z_' + col

            XScore[col_zscore] = zscore(XScore[col])
            del XScore[col]

    # Fill missing values with 0 = the mean in the z-normalize data
    # Obviously missing values can be handled in many different ways
    XScore.fillna(0, inplace=True)

    # GET ALL MATCHING COLUMNS IN X and XScore
    matching_cols = []
    for X_score_col in XScore.columns.values:  # for loop XScoreCols
        match_col = ""
        match_col = [col for col in X.columns.values if X_score_col == col]  # Check if X Train Col exists in X Score Col
        matching_cols.extend(match_col)

    # Set dataframes to include only matching columns
    X = X[matching_cols].sort_index()
    XScore = XScore[matching_cols].sort_index()

    # RANDOM FOREST
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(criterion=splitcriteria_param, n_estimators=int(numbtrees_param), n_jobs=1)
    # FIT THE RANDOM FOREST
    forest.fit(X, y)

    # CROSS VALIDATION
    # Evaluate the model using 5-fold cross-validation
    from sklearn.cross_validation import cross_val_score
    # from sklearn.metrics import roc_auc_score
    accuracyScore = cross_val_score(forest, X, y, scoring='accuracy', cv=5)
    recallScore = cross_val_score(forest, X, y, scoring='recall', cv=5)
    precisionScore = cross_val_score(forest, X, y, scoring='precision', cv=5)
    f1Score = cross_val_score(forest, X, y, scoring='f1', cv=5)
    aucScore = cross_val_score(forest, X, y, scoring='roc_auc', cv=5)

    # SCORE THE DATAFRAME  score_df
    score_df['yProbaTrue'] = np.round(forest.predict_proba(XScore)[:, 1], decimals=3)
    score_df['accuracy'] = np.round(accuracyScore.mean(), decimals=3)
    score_df['recall'] = np.round(recallScore.mean(), decimals=3)
    score_df['precision'] = np.round(precisionScore.mean(), decimals=3)
    score_df['f1'] = np.round(f1Score.mean(), decimals=3)
    score_df['auc'] = np.round(aucScore.mean(), decimals=3)

    return score_df


def runOLSRegression(train_df, score_df):

    # PREPARE TRAINING DATA
    ypredcol = [col for col in train_df.columns.values if 'YPRED' in col.upper()]
    Xcols = [col for col in train_df.columns if col not in ypredcol and 'NOTMODEL' not in col.upper()]

     # Set y to the first occurrence in list
    y = train_df[ypredcol[0]]
    # Convert character columns to dummy variables
    X = train_df[Xcols]
    cols = X.columns
    num_cols = X._get_numeric_data().columns
    char_cols = list(set(cols) - set(num_cols))

    for col in char_cols:
        if len(X[col].unique()) <= 20:
            dummy = pd.get_dummies(X[col], prefix='dm'+col)
            column_name = X.columns.values.tolist()
            column_name.remove(col)
            X = X[column_name].join(dummy)
        else:
            if col in X.columns:    # If more than 20 distinct values then delete
                del X[col]

    # Standardize (Z-score normalize) all continuous variables
    from scipy.stats import zscore
    for col in X:
        if len(X[col].unique()) > 2:  # Standardize non-dummy variables
            col_zscore = 'z_' + col
            X[col_zscore] = zscore(X[col])
            del X[col]

    # Fill missing values with 0 = the mean in the z-normalize data
    # Obviously missing values can be handled in many different ways
    X.fillna(0, inplace=True)

    # PREPARE DATA TO SCORE
    y_score_predcol = [col for col in score_df.columns.values if 'YPRED' in col.upper()]
    X_score_cols = [col for col in score_df.columns if col not in y_score_predcol and 'NOTMODEL' not in col.upper()]

    # Convert character columns to dummy variables
    XScore = score_df[X_score_cols]
    score_cols = XScore.columns
    score_num_cols = XScore._get_numeric_data().columns
    score_char_cols = list(set(score_cols) - set(score_num_cols))

    for col in score_char_cols:
        if len(XScore[col].unique()) <= 20:
            score_dummy = pd.get_dummies(XScore[col], prefix='dm'+col)
            score_column_name = XScore.columns.values.tolist()
            score_column_name.remove(col)
            XScore = XScore[score_column_name].join(score_dummy)
        else:
            if col in XScore.columns:        # If more than 20 distinct values then delete
                del XScore[col]

    # Standardize (Z-score normalize) all continuous variables
    for col in XScore:
        if len(XScore[col].unique()) > 2:  # Standardize non-dummy variables
            col_zscore = 'z_' + col
            XScore[col_zscore] = zscore(XScore[col])
            del XScore[col]

    # Fill missing values with 0 = the mean in the z-normalize data
    # Obviously missing values can be handled in many different ways
    XScore.fillna(0, inplace=True)

    # GET MATCHING COLUMNS IN X and XScore
    matching_cols = []
    for X_score_col in XScore.columns.values:  # for loop XScoreCols
        match_col = ""
        match_col = [col for col in X.columns.values if X_score_col == col]  # Check if X Train Col exists in X Score Col
        matching_cols.extend(match_col)

    # Set dataframes to include only matching columns
    X = X[matching_cols].sort_index()
    XScore = XScore[matching_cols].sort_index()

    # OLS REGRESSION
    from sklearn.linear_model import LinearRegression
    from sklearn.cross_validation import cross_val_score
    ols = LinearRegression()

    # FIT THE MODEL
    ols.fit(X, y)

    # CROSS-VALIDATION Get R2 score and mse through 5- fold cross validation
    Rsquared = cross_val_score(ols, X, y, scoring='r2', cv=5)
    mse = cross_val_score(ols, X, y, scoring='mean_squared_error', cv=5)

    # SCORE THE DATAFRAME  score_df
    score_df['yPrediction'] = np.round(ols.predict(XScore), decimals=3)  # predicted values
    score_df['RSquare'] = np.round(Rsquared.mean(), decimals=3)  # average scores of cross validation
    score_df['mse'] = np.round(mse.mean(), decimals=3)   # average scores of cross validation
    return score_df


def runClustering(cluster_df):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score as silhouette_score

    Xcols = [col for col in cluster_df.columns if 'NOTMODEL' not in col.upper()]

    # Convert character columns to dummy variables
    X = cluster_df[Xcols]
    cols = X.columns
    num_cols = X._get_numeric_data().columns
    char_cols = list(set(cols) - set(num_cols))
    for col in char_cols:
        if len(X[col].unique()) <= 20:
            dummy = pd.get_dummies(X[col], prefix='dm' + col)
            column_name = X.columns.values.tolist()
            column_name.remove(col)
            X = X[column_name].join(dummy)
        else:
            if col in X.columns:    # If more than 20 distinct values then delete
                del X[col]

    # Standardize (Z-score normalize) all continuous variables
    from scipy.stats import zscore
    for col in X:
        if len(X[col].unique()) > 2:    # Standardize non-dummy variables
            col_zscore = 'z_' + col
            X[col_zscore] = zscore(X[col])
            del X[col]

    # Fill missing values with 0 = the mean in the z-normalize data
    # Obviously missing values can be handled in many different ways
    X.fillna(0, inplace=True)

    # convert to matrix/numpy array to use in KMeans clustering class
    data_for_clustering_matrix = X.as_matrix()

    number_of_Clusters = []
    silhouette_value = []
    # Loop through 2 and 20 clusters and identify which has the highest silhouette score
    k = range(2, 21)
    for i in k:
        clustering_method = KMeans(n_clusters=i)
        clustering_method.fit(data_for_clustering_matrix)
        labels = clustering_method.predict(data_for_clustering_matrix)
        silhouette_average = silhouette_score(data_for_clustering_matrix, labels)
        silhouette_value.append(silhouette_average)
        number_of_Clusters.append(int(i))

        # maxind = np.argmax(silhouette_value)
        max_value = max(silhouette_value)
        indexMaxValue = silhouette_value.index(max_value)

        # FIT KMEANS CLUSTER MODEL WITH NUMBER OF CLUSTERS WITH HIGHEST SILHOUETTE SCORE
        clustering_method = KMeans(n_clusters=number_of_Clusters[indexMaxValue])
        clustering_method.fit(data_for_clustering_matrix)
        labels = clustering_method.predict(data_for_clustering_matrix)

        # SCORE THE DATAFRAME  score_df
        cluster_df['cluster'] = labels
        return cluster_df
