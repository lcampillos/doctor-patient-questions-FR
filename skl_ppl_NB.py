import sklearn
import numpy as np
import re

# Import pandas to import data from CSV
import pandas as pd

# Import module for cross validation 
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

# Import transformers/vectorizers
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer

# Import classifiers
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC

# Import pipeline and feature union features
from sklearn.pipeline import Pipeline, FeatureUnion
# Import grid search
from sklearn.model_selection import GridSearchCV

# To print results
from sklearn import metrics

# Auxiliary functions/tools
# FreqDist: given a list, returns a dictionary of frequencies of each list item: {'eme': 2, 'nt ': 2, ...})
from nltk.probability import FreqDist 

# Tokenizing function
# Use parameters to indicate language preferences ('fr' for French, 'sp' for Spanish...)
def tokenize(line,language):
    line = re.sub("’","'",line)
    # lowercase (think if do it or not)
    line = line.lower()    
    if (language=='fr'):
        line = re.sub("(\-t\-)"," t ",line)
        line = re.sub(r"(\-)(je|y|vous|nous|moi|toi|soi|lui|il|ils|elle|elles|ce|là|le|les|on|en)(\s|$)", r" \2 ", line)
        line = re.sub(r"(^|\s)(je|y|vous|nous|moi|toi|soi|lui|il|ils|elle|elles|ce|là|le|les|on|en)(\-)", r" \2 ", line)
        line = re.sub(" -ce "," ce ",line)
    elif (language=='sp'):
        # Beginning typographic characters
        line = re.sub("([¿¡])","\\1 ",line)
    elif (language=='en'):
        # don't -> do n't
        line = re.sub("(n\'t)"," \\1",line)
    #line = re.sub("(\S)([\?\!])","\\1 \\2",line)
    # Remove punctuation marks
    line = re.sub("([\?\!]+)","",line)
    line = re.sub(",","",line)
    line = re.sub("[\(\)]","",line)
    line = re.sub("\'","\' ",line)
    line = re.sub(" +"," ",line)
    Tokens = []
    for token in line.split():
        Tokens.append(token)
    # Remove empty elements
    Tokens = [x for x in Tokens if x != '']
    return Tokens

# Get 3-grams (input is a sentence)
def get_3grams(str):
    v = CountVectorizer(analyzer='word', ngram_range=(3, 3), min_df=1,tokenizer=lambda text: tokenize(str,'fr'))
    analyze = v.build_analyzer()
    List = analyze(str)
    return List

# Get 3-character-grams (input is a sentence)
def get_3chargrams(str):
    v_char = CountVectorizer(analyzer='char', ngram_range=(3, 3), min_df=1,tokenizer=lambda text: tokenize(str,'fr'))
    analyze_char = v_char.build_analyzer()
    List = analyze_char(str)
    return List

# Open list of system vocabulary
File = open("./data/dict_corr_ort_freq.txt","r",encoding="iso-8859-1")
wordsDict={}

for line in File:
    line.strip()
    entry = re.search("(.+?)\|",line)
    if entry:
        item = entry.group(1)
        wordsDict[item]=True
File.close()

# Open lists of stop words
File = open("./data/french_stop_words.txt","r",encoding="iso-8859-1")
for line in File:
    line=line.strip()
    wordsDict[line]=True
File.close()

# Open lists of adverbes
File = open("./data/advs-prep-conj-fr.txt","r",encoding="iso-8859-1")
for line in File:
    line=line.strip()
    wordsDict[line]=True
File.close()

# Feature: word in system vocabulary
# Given a string, test if each token is included in vocabulary list
# Return a dictionary only with out-of-vocabulary tokens: e.g. 'not=emettre': True
def test_word_in_dict(str,wordsDict):
    Lex=[]
    # tokenize
    for token in tokenize(str,'fr'):
        # Do not use numbers
        number = re.search("[0-9]",token)
        if not number and token not in wordsDict:
            data = "not=" + token
            Lex.append(data)
    return Lex

# average word length in sentence
def get_averg_wrd_len(str):
    max_wrd_len = 1
    min_wrd_len = 20
    for word in tokenize(str, 'fr'):
        # minimum word length
        if len(word) < min_wrd_len:
            min_wrd_len = len(word)
        # maximum word length
        if len(word) > min_wrd_len:
            max_wrd_len = len(word)
    # average word length
    averg_wrd_len = np.mean([len(word) for word in tokenize(str, 'fr')])
    Averg = []
    averg_wrd_len = np.mean([len(word) for word in tokenize(str,'fr')])
    Averg.append(averg_wrd_len)
    Averg.append(min_wrd_len)
    Averg.append(max_wrd_len)
    return Averg


''' Importing data with pandas + cross-validation
# Last column needs to be the labels '''
dataset = pd.read_csv('./data/questions_to_classif_root_sem_nwords_all_with_lbls.csv', header=0, sep=';', encoding="iso-8859-1")
X, y = dataset.iloc[:,:-1], dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y) # use "stratify" parameter to get better results

# List of all models to test
# Contains lists where: the 1st element is the name of features used, the 2nd is the training set, and the 3rd is the test set
Models=[]

# Given a list of features and a corpus (in the pandas dataframe format), it creates the model
# (output is a list of feature lists for each sentence)
# Possible features are:
#    'root': question root (3 first words)
#    'str': token + freq in sentence
#    'n_wds': number of words in sentence
#    '3grams': 3-grams + freq
#    '3chargrams': 3-character-grams + freq
#    'sem_tag': semantic label + freq of semantic annotation
#     'wrd_len': minimum, maximum and average word length in sentence
#    'in_lexic': the word is not in system vocabulary (known words are not included)
def create_model_list(feature_list,corpus):
    SentData=[]
    for i,_ in enumerate(corpus['str'].tolist()):
        emptyList=[]
        SentData.append(emptyList)
    for feature in feature_list:
        # Feature: question root
        if (feature=="root"):
            questRoots = corpus['root'].tolist()
            for i,_ in enumerate(questRoots):
                SentData[i].append(questRoots[i])
        # Token in sentence
        elif (feature=="str"):
            Tokens = [tokenize(row,'fr') for row in corpus['str']] #print("------",row)
            for i,_ in enumerate(SentData):
                for token in Tokens[i]:
                    SentData[i].append(token)
        # Number of words in sentence
        elif (feature=="n_wds"):
            ListNWords = corpus['n_wds'].tolist()
            for i,_ in enumerate(ListNWords):
                # Convert integer to string to avoid errors mixing different data types
                SentData[i].append(str(ListNWords[i]))
        # Feature: 3-grams
        elif (feature=="3grams"):
            List = [get_3grams(row) for row in corpus['str']]
            for i,_ in enumerate(SentData):
                for trigr in List[i]:
                    SentData[i].append(trigr)
        # Feature: 3-char-grams
        elif (feature=="3chargrams"):
            List = [get_3chargrams(row) for row in corpus['str']]
            for i,_ in enumerate(SentData):
                SentData[i] = SentData[i] + List[i]
        # Feature: semantic label
        elif (feature=="sem_tag"):
            List=[tokenize(row,'fr') for row in corpus['sem_tag']]
            for i,_ in enumerate(SentData):
                SentData[i]=SentData[i] + List[i]
        # Feature: average word length in sentence
        elif (feature=="wrd_len"):
            AvgWrdLen = [get_averg_wrd_len(row) for row in corpus['str']]
            for i,_ in enumerate(SentData):
                SentData[i].append(str(AvgWrdLen[i]))
        # Feature: the word is not in system vocabulary
        elif (feature=="in_lexic"):
            LexData = [test_word_in_dict(row,wordsDict) for row in corpus['str']]
            for i,_ in enumerate(SentData):
                SentData[i]=SentData[i] + LexData[i]
    return SentData


''' Models for NB '''
model_00 = [ 'root', create_model_list(['root'],X_train), create_model_list(['root'],X_test) ]
Models.append(model_00)

model_01 = [ 'str', create_model_list(['str'],X_train), create_model_list(['str'],X_test) ]
Models.append(model_01)

model_02 = [ 'n_wds', create_model_list(['n_wds'],X_train), create_model_list(['n_wds'],X_test) ]
Models.append(model_02)

model_03 = [ 'sem_tag', create_model_list(['sem_tag'],X_train), create_model_list(['sem_tag'],X_test) ]
Models.append(model_03)

model_05 = [ '3grams', create_model_list(['3grams'],X_train), create_model_list(['3grams'],X_test) ]
Models.append(model_05)

model_06 = [ 'wrd_len', create_model_list(['wrd_len'],X_train), create_model_list(['wrd_len'],X_test) ]
Models.append(model_06)

model_07 = [ '3chargrams', create_model_list(['3chargrams'],X_train), create_model_list(['3chargrams'],X_test) ]
Models.append(model_07)

model_1 = [ 'str_+_n_wds', create_model_list(['str','n_wds'],X_train), create_model_list(['str','n_wds'],X_test) ]
Models.append(model_1)

model_2 = [ 'str_+_sem_tag', create_model_list(['str','sem_tag'],X_train), create_model_list(['str','sem_tag'],X_test)]
Models.append(model_2)

model_3 = [ 'str_+_sem_tag_+_n_wds', create_model_list(['str','n_wds','sem_tag'],X_train), create_model_list(['str','n_wds','sem_tag'],X_test)]
Models.append(model_3)

model_4 = [ 'str_+_sem_tag_+_in_lexic', create_model_list(['str','sem_tag', 'in_lexic'],X_train), create_model_list(['str','sem_tag', 'in_lexic'],X_test)]
Models.append(model_4)

model_5 = [ 'str_+_sem_tag_+_root', create_model_list(['str','sem_tag','root'],X_train), create_model_list(['str','sem_tag','root'],X_test)]
Models.append(model_5)

model_6 = [ 'str_+_sem_tag_+_wd_len', create_model_list(['str','sem_tag','wd_len'],X_train), create_model_list(['str','sem_tag','wd_len'],X_test)]
Models.append(model_6)

model_7 = [ 'str_+_sem_tag_+_3grams', create_model_list(['str','sem_tag','3grams'],X_train), create_model_list(['str','sem_tag','3grams'],X_test)]
Models.append(model_7)

model_7bis = [ 'str_+_3grams', create_model_list(['str','3grams'],X_train), create_model_list(['str','3grams'],X_test)]
Models.append(model_7bis)

model_8 = [ 'str_+_sem_tag_+_3chargrams', create_model_list(['str','sem_tag','3chargrams'],X_train), create_model_list(['str','sem_tag','3chargrams'],X_test)]
Models.append(model_8)

model_8bis = [ 'str_+_3chargrams', create_model_list(['str','3chargrams'],X_train), create_model_list(['str','3chargrams'],X_test)]
Models.append(model_8bis)

model_9 = [ 'root_+_str_+_n_wds_+_sem_tag', create_model_list(['root','str','n_wds','sem_tag'],X_train), create_model_list(['root','str','n_wds','sem_tag'],X_test)]
Models.append(model_9)

model_9bis = [ 'str_+_sem_tag_+_3grams_+_3chargrams', create_model_list(['str','sem_tag','3grams','3chargrams'],X_train), create_model_list(['str','sem_tag','3grams','3chargrams'],X_test)]
Models.append(model_9bis)

model_9c = [ 'str_+_sem_tag_+_3grams_+_n_wds', create_model_list(['str','sem_tag','3grams','n_wds'],X_train), create_model_list(['str','sem_tag','3grams','n_wds'],X_test)]
Models.append(model_9c)

model_9d = [ 'str_+_sem_tag_+_3grams_+_root', create_model_list(['str','sem_tag','3grams','root'],X_train), create_model_list(['str','sem_tag','3grams','root'],X_test)]
Models.append(model_9d)

model_9e = [ 'str_+_sem_tag_+_3grams_+_wd_len', create_model_list(['str','sem_tag','3grams','wd_len'],X_train), create_model_list(['str','sem_tag','3grams','wd_len'],X_test)]
Models.append(model_9e)

model_9f = [ 'str_+_sem_tag_+_3grams_+_in_lexic', create_model_list(['str','sem_tag','3grams','in_lexic'],X_train), create_model_list(['str','sem_tag','3grams','in_lexic'],X_test)]
Models.append(model_9f)

model_10 = [ 'root_+_str_+_n_wds_+_sem_tag_+_wd_len', create_model_list(['root','str','n_wds','sem_tag', 'wrd_len'],X_train), create_model_list(['root','str','n_wds','sem_tag','wrd_len'],X_test)]
Models.append(model_10)

model_10bis = [ 'str_+_n_wds_+_sem_tag_+_3grams_+_in_lexic', create_model_list(['str','n_wds','sem_tag','3grams','in_lexic'],X_train), create_model_list(['str','n_wds','sem_tag','3grams','in_lexic'],X_test)]
Models.append(model_10bis)

model_11 = [ 'root_+_str_+_n_wds_+_sem_tag_+_wd_len_+_in_lexic', create_model_list(['root','str','n_wds','sem_tag', 'wrd_len', 'in_lexic'],X_train), create_model_list(['root','str','n_wds','sem_tag','wrd_len','in_lexic'],X_test)]
Models.append(model_11)

model_12 = [ 'root_+_str_+_n_wds_+_sem_tag_+_wd_len_+_3grams_+_in_lexic', create_model_list(['root','str','n_wds','sem_tag', '3grams', 'wrd_len', 'in_lexic'],X_train), create_model_list(['root','str','n_wds','sem_tag', '3grams', 'wrd_len','in_lexic'],X_test)]
Models.append(model_12)

model_13 = [ 'root_+_str_+_n_wds_+_wd_len_+_3grams_+_in_lexic', create_model_list(['root','str','n_wds','3grams', 'wrd_len', 'in_lexic'],X_train), create_model_list(['root','str','n_wds', '3grams', 'wrd_len','in_lexic'],X_test)]
Models.append(model_13)

model_14 = [ 'root_+_str_+_n_wds_+_sem_tag_+_wd_len_+_3chargrams_+_in_lexic', create_model_list(['root','str','n_wds','sem_tag', '3chargrams', 'wrd_len', 'in_lexic'],X_train), create_model_list(['root','str','n_wds','sem_tag', '3chargrams', 'wrd_len','in_lexic'],X_test)]
Models.append(model_14)

model_15 = [ 'root_+_str_+_n_wds_+_3grams_+_3chargrams_+_sem_tag_+_wrd_len_+_in_lexic', create_model_list(['root','str', 'n_wds', '3grams', '3chargrams', 'sem_tag', 'wrd_len', 'in_lexic'], X_train), create_model_list(['root','str', 'n_wds', '3grams', '3chargrams', 'sem_tag', 'wrd_len', 'in_lexic'], X_test) ]
Models.append(model_15)

model_16 = [ 'root_+_str_+_n_wds_+_wd_len_+_in_lexic', create_model_list(['root','str','n_wds','wrd_len','in_lexic'],X_train), create_model_list(['root','str','n_wds','wrd_len','in_lexic'],X_test)]
Models.append(model_16)


#######################
# Build the pipelines #
#######################


pipeline = Pipeline([
    ('clf',GaussianNB()) # Best result: F1 = 0.75
    #('clf', MultinomialNB())
      ])


'''# Simple pipeline with different dictionaries (each is a model with a different combination of features) '''
for feature_model in Models:
    print("Model:",feature_model[0])
    X_train = feature_model[1]
    X_test = feature_model[2]
    # For GaussianNB, use the following data conversion for modelling:
    # Use parameter "tokenizer=lambda doc: doc" when vectorizing a list of lists (each for a sentence)
    v = CountVectorizer(min_df=1, tokenizer=lambda doc: doc, lowercase=False)
    X_train = v.fit_transform(X_train)
    X_test = v.transform(X_test)
    #########################
    # This gets cross-validation on the training set (split in 10 subsets)
    # Accuracy
    # scores = cross_val_score(pipeline, X_train.toarray(), np.asarray(y_train), cv=10)
    # print("Accuracy of training set (10-fold cross-validation): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(pipeline, X_train.toarray(), np.asarray(y_train), cv=10, scoring = 'f1_weighted')
    print("F1-weighted of training set (10-fold cross-validation): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    #########################
    # Predictions using a test set
    model = pipeline.fit(X_train.toarray(), np.asarray(y_train))
    # test the classifier
    predicted = model.predict(X_test.toarray())
    print(metrics.classification_report(np.asarray(y_test), predicted, digits=3))
    print("-------------------------------")

'''
# Do grid search to test hyperparameters
# For SVCLinear()
parameters = {'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'clf__loss': ['hinge', 'squared_hinge'] # This causes errors
              }
# For GaussianNB, do not use parameters

grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)

# Get the best hyperparameters by fitting the training data to the training values/labels
grid.fit(X_train,y_train) 

y_predictions = grid.predict(X_test)
report = sklearn.metrics.classification_report( y_test, y_predictions )
print(metrics.classification_report(y_test, y_predictions))
print("Best parameters set:")
print(grid.best_params_)
print("Grid scores:")
print(grid.best_score_)
'''
