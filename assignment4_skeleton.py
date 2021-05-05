# Skeleton for Assignment 4.
# Ling471 Spring 2021.

import sys
import re
import string
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


'''
The below function should be called on a file name.
It opens the file, reads its contents, stores it in a variable.
Then it removes punctuation marks, and returns the "cleaned" text.
'''


def cleanFileContents(f):
    with open(f, 'r') as f:
        text = f.read()
    clean_text = text.translate(str.maketrans('', '', string.punctuation))
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text


'''The main function is the entry point of the program.
When debugging, if you want to start from the very beginning,
start here. NB: Put the breakpoint not on the "def" line but below it.'''


# This function is already fully written.
# Do not modify it; only modify the createDataFrames function.
def main(argv):
    df_train, df_test = createDataFrames(argv)

    # TODO: Set the below 4 variables to contain:
    # X_train: the training data; y_train: the training data labels;
    # X_test: the test data; y_test: the test data labels.
    # Access the data frames by column names.
    # X_train = 
    # y_train = 
    # X_test = 
    # y_test = 

    # TODO: Look up what the astype() method is doing and add a comment, explaining in your own words,
    # what the next two lines are doing.
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    # TODO: Look up what TfidfVectorizer is and what its methods "fit_transform" and "transform" are doing,
    #  and add a comment, explaining in your own words,
    # what the next three lines are doing.
    tf_idf_vect = TfidfVectorizer(ngram_range=(1, 2))
    tf_idf_train = tf_idf_vect.fit_transform(X_train.values)
    tf_idf_test = tf_idf_vect.transform(X_test.values)

    # TODO: Look up what "alpha" is in the MultinomialNB sklearn class, and add a comment explaining in your own words,
    # what this parameter is for. The value "6" was picked by Olga through a cross-validation procedure. Explain in a comment,
    # what that means. Try to be brief. You don't need to be formal or fully correct.
    clf = MultinomialNB(alpha=6)
    # TODO: Add a comment explaining in your own words what the "fit" method is doing.
    clf.fit(tf_idf_train, y_train)

    # TODO: Add a comment explaining in your own words what the "fit" method is doing in the next two lines.
    y_pred_train = clf.predict(tf_idf_train)
    y_pred_test = clf.predict(tf_idf_test)

    acc = accuracy_score(y_test, y_pred_test, normalize=True) * float(100)
    acc_train = accuracy_score(
        y_train, y_pred_train, normalize=True) * float(100)
    print(acc_train)
    print(acc)


'''
Write a function which accepts a list of 4 directories:
train/pos, train/neg, test/pos, and test/neg.

It returns two pandas dataframes: one for the training data and another for the test data.
The columns are labeled "label" and "text".

'''


def createDataFrames(argv):
    train_pos = list(Path(argv[1]).glob("*.txt"))
    train_neg = list(Path(argv[2]).glob("*.txt"))
    test_pos = list(Path(argv[3]).glob("*.txt"))
    test_neg = list(Path(argv[3]).glob("*.txt"))

    train_data = []
    test_data = []

    # TODO: Populate train_data and test_data.
    # Each entry should itself be a list of the form: [label, text],
    # where label is 1 if the review is positive and 0 if the review is negative,
    # and text is the cleaned review text.

    # Your code here...
 
    # Column names to use in the dataframes
    column_names = ["label", "text"]
    # TODO: Create two pandas dataframes (pd.DataFrame() constructor), 
    # one for training data and another for test data,
    #  using the column names above.
    
    # df_train = 
    # df_test = 
    
    return(df_train, df_test)


if __name__ == "__main__":
    main(sys.argv)
