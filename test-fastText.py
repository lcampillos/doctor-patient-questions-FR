#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# test-fastText.py
#
# https://github.com/facebookresearch/fastText
#
# ./test-fastText.py <data/input_file> (python3 interpreter)
#
# Or: python3 test-fastText.py data/questions_to_classif_u8.tst
#
###########################

import re
import sys
import os
import sklearn
from sklearn.grid_search import ParameterGrid

#### subroutines
# remove punctuation marks, separate with white space apostrophe
def normalize(sentence):
    # lowercase
    sentence = sentence.lower()
    # remove punctuation marks
    sentence = re.sub("[?!.,;\(\)]","", sentence)
    sentence = re.sub(",","",sentence)
    sentence = re.sub("'","' ", sentence)
    sentence = re.sub("(\-t\-)"," t ",sentence)
    sentence =re.sub(r"(\-)(je|y|vous|nous|moi|toi|soi|lui|il|ils|elle|elles|ce|là|le|les|on|en)(\s|$)",r" \2 ",sentence)
    sentence =re.sub(r"(^|\s)(je|y|vous|nous|moi|toi|soi|lui|il|ils|elle|elles|ce|là|le|les|on|en)(\-)",r" \2 ",sentence)
    sentence = re.sub(" -ce "," ce ",sentence)
    sentence = re.sub("^ ","",sentence)
    sentence = re.sub(" +", " ", sentence)
    #print(sentence)
    return sentence


corpusTrnFile = open(sys.argv[1],'r',encoding='utf-8')
auxFileTrnName = os.path.splitext(os.path.basename(str(corpusTrnFile)))[0] + "_trn.aux"
auxFileTrnNamePath = os.path.join("data/",auxFileTrnName)
auxFileTrn = open(auxFileTrnNamePath,'w',encoding='utf-8')

for line in corpusTrnFile:
    line=line.strip()
    line = normalize(line)
    print(line,file=auxFileTrn)
corpusTrnFile.close()

corpusTstFile = open('data/questions_to_classif_u8.tst', 'r', encoding='utf-8')
auxFileTstName = os.path.splitext(os.path.basename(str(corpusTstFile)))[0] + "_tst.aux"
auxFileTstNamePath = os.path.join("data/", auxFileTstName)
auxFileTst = open(auxFileTstNamePath,'w', encoding='utf-8')

for line in corpusTstFile:
    line = line.strip()
    line = normalize(line)
    print(line, file=auxFileTst)
corpusTstFile.close()


parameters = {
    # window size (5 by default)
    'ws': [2, 4, 6, 8, 10],
    # vector dimension
    'dim': [50, 100, 300],
    # max length of char ngram
    'maxn': [0, 3],#0,
    # length of word ngrams
    'wordNgrams': [0, 3],
    # number of negative samples (5 by default)
    'neg': [10, 20],
    # learning rate (0.1 by default)
    'lr': [0.1, 0.05],
    # sampling threshold (0.0001 by default)
    't': [0.001, 0.0001]
}

parameter_grid = list(ParameterGrid(parameters))

for parameter_set in parameter_grid:
    model_name = "questions_model_ws={}_dim={}_3chgr={}_3gr={}_neg={}_lr={}_t={}".format(parameter_set['ws'], parameter_set['dim'], parameter_set['maxn'], parameter_set['wordNgrams'], parameter_set['neg'], parameter_set['lr'], parameter_set['t'])
    # Pre-train word vectors
    # Note that -minCount 1 by default; no need to indicate it if this value is needed
    pretrained_model = "model_EMEA_ws={}_dim={}_3chgr={}_3gr={}_neg={}_lr={}_t={}".format(parameter_set['ws'], parameter_set['dim'], parameter_set['maxn'], parameter_set['wordNgrams'], parameter_set['neg'], parameter_set['lr'], parameter_set['t'])
    os.system('./fasttext skipgram -verbose 0 -dim ' + str(parameter_set['dim']) + ' -ws ' + str(
        parameter_set['ws']) + ' -neg ' + str(parameter_set['neg']) + ' -maxn ' + str(parameter_set['maxn']) + ' -wordNgrams ' + str(parameter_set['wordNgrams']) + ' -lr ' + str(parameter_set['lr']) + ' -t ' + str(parameter_set['t']) +' -input data/EMEA.es-fr.fr.tok -output data/' + str(pretrained_model))
    # Train supervised text classifiers
    os.system('./fasttext supervised -verbose 0 -pretrainedVectors data/' + str(pretrained_model) + ".vec" + ' -dim ' + str(parameter_set['dim']) + ' -ws ' + str(parameter_set['ws']) + ' -maxn ' + str(parameter_set['maxn']) + ' -wordNgrams ' + str(parameter_set['wordNgrams']) + ' -neg ' + str(parameter_set['neg']) + ' -lr ' + str(parameter_set['lr']) + ' -t ' + str(parameter_set['t']) + ' -input ' + str(auxFileTrnNamePath) + ' -output data/model_' + str(model_name))
    # Test model
    os.system("echo RESULTS FOR " + str(model_name) + " ; " + "./fasttext test data/model_" + str(model_name) + ".bin " + str(auxFileTstNamePath))
    os.system("echo -----------------")
    # Remove files to empty space
    os.system("rm data/model_" + str(model_name) + ".bin")
    os.system("rm data/" + str(pretrained_model) + ".vec")
    os.system("rm data/" + str(pretrained_model) + ".bin")

auxFileTrn.close()
auxFileTst.close()