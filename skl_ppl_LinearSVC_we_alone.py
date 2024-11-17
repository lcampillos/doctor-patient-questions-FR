import sklearn
import numpy as np
import re
# module for Python codecs (encoders and decoders)
import codecs

# Import pandas to import data from CSV
import pandas as pd

# Import module for cross validation 
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

# Modules to define costume transformers
from sklearn.base import BaseEstimator, TransformerMixin

# Import classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

# Import pipeline and feature union features
from sklearn.pipeline import Pipeline, FeatureUnion

# To print results
from sklearn import metrics

# Auxiliary functions/tools
# Word embedding transformer
# Code from: http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """
    Word embedding transformer.
    Uses a dictionary mapping words to vectors to build features.
    The simplest way to do that is by averaging word vectors for all words in a text.
    """
    def __init__(self, we):
        """
        Input:
        we: word embedding dictionary. Must implement 'we[token]' and 'token in we'.
        """
        self.we = we
        # dimensionality of vector/we
        self.dim = next(iter(we.values())).shape

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
                # average of word vectors in sentence
                # if a text is empty we should return a vector of zeros
                # with the same dimensionality as all the other vectors
                np.mean([self.we[w] for w in words if w in self.we]
                        or [np.zeros(self.dim)], axis=0)
                for words in X
                ])

# use word embeddings for each word in the sentence
# Use the .vec format. Pretrained word embeddings format is compatible with those of Word2Vec and FastText
def read_embeddings(file):
    with codecs.open(file, "r", "utf-8") as lines:
        w2v = {line.split()[0]: np.array([x for x in map(float, line.split()[1:])]) for line in lines if len(line.split()) > 2}
    return w2v

# Last column needs to be the labels
dataset = pd.read_csv('./data/questions_to_classif_root_sem_nwords_all_with_lbls.csv', header=0, sep=';', encoding="iso-8859-1")
X, y = dataset.iloc[:,:-1], dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y) # use "stratify" parameter to get better results

# List of all models to test
# Contains lists where: the 1st element is the name of features used, the 2nd is the training set, and the 3rd is the test set
Models=[]

# Pretrained word embeddings
# First model tested: we = read_embeddings("data/model_EMEA_ws=6_dim=50_3chgr=3_3gr=0_neg=10.vec")
we = read_embeddings("data/model_EMEA_ws=10_dim=100_3chgr=0_3gr=0_neg=10.vec")
print("Loaded word embeddings")


model_we = [ 'we', X_train['str'], X_test['str'] ]
Models.append(model_we)


#######################
# Build the pipelines #
#######################


pipeline = Pipeline([
              # To test we alone
              ('we', MeanEmbeddingVectorizer(we)),
              ('clf', LinearSVC())
      ])


'''# Simple pipeline with different dictionaries (each is a model with a different combination of features) '''
for feature_model in Models:
    print("Model:",feature_model[0])
    X_train = feature_model[1]
    X_test = feature_model[2]
    # train the classifier
    # This gets cross-validation on the training set (split in 10 subsets)
    # Accuracy:
    # scores = cross_val_score(pipeline, X_train, y_train, cv=10)
    # print("Accuracy of training set (10-fold cross-validation): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    # F1:
    scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='f1_weighted')
    print("F1-weighted of training set (10-fold cross-validation): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    #########################
    # Predictions using a test set
    model = pipeline.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print("Test set:", metrics.classification_report(y_test, predicted, digits=3))
    #########################