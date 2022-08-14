#%load_ext autoreload
#%autoreload 2

import os
os.getcwd()

import spacy
from spacy.pipeline import EntityRuler
from spacy.tokens import Token, Doc, Span
from spacy.matcher import PhraseMatcher
from spacy.matcher import Matcher
from spacy.tokens import DocBin

import os
import re
import pandas as pd

import numpy as np
import pickle
import json
import sys
from pathlib import Path

df_filepaths = pd.read_csv("VIP_FilePaths.csv")
Path2NLPLibrary = df_filepaths.at[0,'PathActual']
Path2NLPInputFiles = df_filepaths.at[1,'PathActual']
Path2Data = df_filepaths.at[2,'PathActual']
Path2OutputEval = df_filepaths.at[3,'PathActual']

sys.path.insert(1, Path2NLPLibrary)#inserts it into the workflow

from lib.keywds import *

from lib.keywds import *
from lib.EncapLabelingFun import *
from lib.helper_fun import make_analytic_df

StudyIndex = 'Study_N'#name of columns with the study number of the patient
FlowInfoPrefix = 'Flow_Bullet'#column with clinical report used for NLP
PredFlowPrefix = 'PredNormAb'#predicted flow normal/abnormal column
csvfilename = '17segment_2008to2019 + Deidentified_mod_RC.csv'
interp_colum = 'Interp_Bin'#true flow normal/abnormal column
inputs_Dir = 'inputs'#name of folder to store input generated
data_Dir = 'data'#name of folder where data is stored

json_dir = os.path.join(Path2NLPInputFiles, inputs_Dir)

root2csv = os.path.join(Path2Data, data_Dir)


#%%time

###read in input parameters, cleaned csv and sentence filter, labels for nlp objects
print('reading in files and parameters')
###################################################INPUT PARAMETERS#########################
outputDir_folder = 'snorkel_evaluation'
inputdir = json_dir
metrics_out = 'metrics_IschemiaVessel'
outputDir = os.path.join(Path2OutputEval, outputDir_folder)

#This df contains the cleaned strings as well as the true columns for precision and recall calculations
df = pd.read_csv(root2csv + '\\' + csvfilename)

#This filter should be the same as the one used in the nlp_pipe streamed to create the nlp objects
filt ='ischemi(a|c)|scar|lcx_disease|transmural|nontransmural|infarction|infarct'

#subset of columns to include in output dataframe
orig_columns = [StudyIndex, 'Ischemia_Ischemia', FlowInfoPrefix,
       'Ischemia_LAD', 'Ischemia_LCX', 'Ischemia_RCA']

#one of the keys included in the input arguments json to pull the retokenization list from from
example_key = 'Ischemia_RCA'

#input parameters for snorkel pipeline created from input_arguments script
with open(f'{inputdir}/IschemiaVessel_input_parameters.json') as f:
    inputDatas = json.load(f)
    
   #retokenization is the only step in the spacy pipeline that occurs in this script and not the nlp_pipe script
   # since the retokenization may be specific to the search patterns/rules
    # make sure the key value here matches the 
retokenization_list = inputDatas[example_key]['retokenization_list']

#These are the prefixes from the pickle files created in the nlp_pipe script
labs = [FlowInfoPrefix, FlowInfoPrefix + '_Lemmas', FlowInfoPrefix + '_sents', FlowInfoPrefix + '_Lemmas_sents']
#the suffix for the study ids json to rebuild the the doc, sentence structure
json_suffix = FlowInfoPrefix

##################################load nlp model#########################################
# Load English tokenizer, tagger, parser, NER and word vectors
#only needs to be done once per session.

print('loading nlp model')
nlp = load_negex_model(model = "en_core_sci_lg", language = 'en_clinical', 
                          chunks = ["no"], pseudo_negations=["with no improvement at rest",
                                                             "with no improvement on rest",
                                                             "not improve at rest", 
                                                             "with no reversibility at rest",
                                                             "with no improvement on the rest study",
                                                             "with no significant improvement on rest imaging",
                                                             "with no reversibility on rest"],
                      preceding_negations =["rather than"])


###########################################################################
#Load in the nlp objects from pickle and run retokenization. Then reassociate the sentences 
#with the docs to build the analytic dataframe to feed into snorkel

print('processing nlp objects and nlp object keys')
#read in nlp objects
nlp_objects = {}
for index,lab in enumerate(labs):
    with open(f"{inputdir}/{lab}_nlp_objects.pickle", "rb") as handle:
        bytes_data = pickle.load(handle)
        doc_bin = DocBin().from_bytes(bytes_data)
        docs = list(doc_bin.get_docs(nlp.vocab))

        print(f'reading in {lab} with {len(docs)} documents')
    nlp_objects.update({index:docs})

print(nlp_objects.keys())        

# read in keys to rebuild sentences to docs
with open(f'{inputdir}/NLPpipeKeys_{json_suffix}.json') as f:
    Keys = json.load(f)
print(Keys.keys())

print('creating analytic df')
#make analytic df with nlp objects and merge back with original columns needed for evaluation

forPipeDF = make_analytic_df(df, eval_columns = orig_columns, nlp_objects =nlp_objects, nlp_Keys = Keys, filt = filt, retokenization_list = retokenization_list, StudyKey=StudyIndex)

#Run the snorkel pipeline combining the nlp objects, the 'truth' labels and the snorkel function arguments from the input arguments script
#todo: both this function and the input arguments need additional feature to work when there are no 'truth labels'
print('running snorkel pipeline')
metrics_df = pd.DataFrame(columns=['labels','label','precision', 'recall', 'f1'])

for key in inputDatas:
    print(f'running {key} rules')
    column = inputDatas[key]['true_col']
    out_col =inputDatas[key]['predicted_column']
    inputs = inputDatas[key]['inputArgs']


    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        
    coverage_overlap, preds_train, forPipeDF = snorkel_pipeline(inputs, forPipeDF, out_col, column)
    prf_dict  = precision_recall(forPipeDF, out_col, column, preds_train)
    metrics_df = metrics_df.append(prf_dict, ignore_index=True)
    coverage_overlap.to_csv(f'{outputDir}/{column}_overlap.csv')

forPipeDF.to_csv(f'{outputDir}/{metrics_out}_data.csv')
metrics_df.to_csv(f'{outputDir}/{metrics_out}.csv')


#%%time

###read in input parameters, cleaned csv and sentence filter, labels for nlp objects
print('reading in files and parameters')
###################################################INPUT PARAMETERS#########################
metrics_out = 'metrics_ScarVessel'

df = pd.read_csv(root2csv + '\\' + csvfilename)
filt ='ischemi(a|c)|scar|lcx_disease|transmural|nontransmural|infarction|infarct'

orig_columns = [StudyIndex, 'Scar_Scar', FlowInfoPrefix,
       'Scar_LAD', 'Scar_LCX', 'Scar_RCA']

example_key = 'Scar_RCA'

#input parameters for snorkel pipeline
with open(f'{inputdir}/ScarVessel_input_parameters.json') as f:
    inputDatas = json.load(f)

    
retokenization_list = inputDatas[example_key]['retokenization_list']

labs = [FlowInfoPrefix, FlowInfoPrefix + '_Lemmas', FlowInfoPrefix + '_sents', FlowInfoPrefix + '_Lemmas_sents']
json_suffix = FlowInfoPrefix
##################################load nlp model#########################################
# Load English tokenizer, tagger, parser, NER and word vectors
print('loading nlp model')
# nlp = load_negex_model(model = "en_core_sci_lg", language = 'en_clinical', 
#                           chunks = ["no"], pseudo_negations=["with no improvement at rest",
#                                                              "with no improvement on rest",
#                                                              "not improve at rest", 
#                                                              "with no reversibility at rest",
#                                                              "with no improvement on the rest study",
#                                                              "with no significant improvement on rest imaging",
#                                                              "with no reversibility on rest"],
#                       preceding_negations =["rather than"])
# #set custom extensions
# Span.set_extension("negex", default=False, force=True)
# Token.set_extension("negex", default=False, force=True)


###########################################################################
print('processing nlp objects and nlp object keys')
#read in nlp objects
nlp_objects = {}
for index,lab in enumerate(labs):
    with open(f"{inputdir}/{lab}_nlp_objects.pickle", "rb") as handle:
        bytes_data = pickle.load(handle)
        doc_bin = DocBin().from_bytes(bytes_data)
        docs = list(doc_bin.get_docs(nlp.vocab))
        print(f'reading in {lab} with {len(docs)} documents')
    nlp_objects.update({index:docs})

print(nlp_objects.keys())        

# read in keys to rebuild sentences to docs
with open(f'{inputdir}/NLPpipeKeys_{json_suffix}.json') as f:
    Keys = json.load(f)
print(Keys.keys())

print('creating analytic df')
#make analytic df with nlp objects and merge back with original columns needed for evaluation

forPipeDF = make_analytic_df(df, eval_columns = orig_columns, nlp_objects =nlp_objects, nlp_Keys = Keys, filt = filt, retokenization_list = retokenization_list, StudyKey=StudyIndex)

print('running snorkel pipeline')
metrics_df = pd.DataFrame(columns=['labels','label','precision', 'recall', 'f1'])

for key in inputDatas:
    print(f'running {key} rules')
#     pathologies = inputDatas[key]['pathology']
#     regions = inputDatas[key]['regions']
    column = inputDatas[key]['true_col']
    out_col =inputDatas[key]['predicted_column']
    inputs = inputDatas[key]['inputArgs']


    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        
    coverage_overlap, preds_train, forPipeDF = snorkel_pipeline(inputs, forPipeDF, out_col, column)
    prf_dict  = precision_recall(forPipeDF, out_col, column, preds_train)
    metrics_df = metrics_df.append(prf_dict, ignore_index=True)
    coverage_overlap.to_csv(f'{outputDir}/{column}_overlap.csv')

forPipeDF.to_csv(f'{outputDir}/{metrics_out}_data.csv')
metrics_df.to_csv(f'{outputDir}/{metrics_out}.csv')











#%%time

###read in input parameters, cleaned csv and sentence filter, labels for nlp objects
print('reading in files and parameters')
###################################################INPUT PARAMETERS#########################
metrics_out = 'metrics_overall'

df = pd.read_csv(root2csv + '\\' + csvfilename)
filt ='normal|'

orig_columns = [StudyIndex, interp_colum, PredFlowPrefix]


#input parameters for snorkel pipeline
with open(f'{inputdir}/NormAbnorm_input_parameters.json') as f:
    inputDatas = json.load(f)
    

retokenization_list = []

labs = [PredFlowPrefix, PredFlowPrefix + '_Lemmas', PredFlowPrefix + '_sents', PredFlowPrefix + '_Lemmas_sents']
json_suffix = PredFlowPrefix
##################################load nlp model#########################################
# # Load English tokenizer, tagger, parser, NER and word vectors
print('loading nlp model')
nlp = load_negex_model(model = "en_core_sci_lg", language = 'en_clinical', 
                          chunks = ["no"], pseudo_negations=["with no improvement at rest",
                                                             "with no improvement on rest",
                                                             "not improve at rest", 
                                                             "with no reversibility at rest",
                                                             "with no improvement on the rest study",
                                                             "with no significant improvement on rest imaging",
                                                             "with no reversibility on rest"],
                      preceding_negations =["rather than"])
#set custom extensions
Span.set_extension("negex", default=False, force=True)
Token.set_extension("negex", default=False, force=True)


###########################################################################
print('processing nlp objects and nlp object keys')
#read in nlp objects
nlp_objects = {}
for index,lab in enumerate(labs):
    with open(f"{inputdir}/{lab}_nlp_objects.pickle", "rb") as handle:
        bytes_data = pickle.load(handle)
        doc_bin = DocBin().from_bytes(bytes_data)
        docs = list(doc_bin.get_docs(nlp.vocab))
        print(f'reading in {lab} with {len(docs)} documents')
    nlp_objects.update({index:docs})

print(nlp_objects.keys())        

 

# read in keys to rebuild sentences to docs
with open(f'{inputdir}/NLPpipeKeys_{json_suffix}.json') as f:
    Keys = json.load(f)
print(Keys.keys())

print('creating analytic df')
#make analytic df with nlp objects and merge back with original columns needed for evaluation

forPipeDF = make_analytic_df(df, eval_columns = orig_columns, 
                             nlp_objects = nlp_objects, 
                             nlp_Keys = Keys, 
                             filt = filt, 
                             retokenization_list = retokenization_list, 
                             StudyKey=StudyIndex)
forPipeDF[PredFlowPrefix] = forPipeDF[PredFlowPrefix].astype('str')

print('running snorkel pipeline')
metrics_df = pd.DataFrame(columns=['labels','label','precision', 'recall', 'f1'])

for key in inputDatas:
    print(f'running {key} rules')

  
    column = inputDatas[key]['true_col']
    out_col =inputDatas[key]['predicted_column']
    inputs = inputDatas[key]['inputArgs']


    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        
    coverage_overlap, preds_train, forPipeDF = snorkel_pipeline(inputs, forPipeDF, out_col, column)
    for average in ['micro', 'weighted', None]:
        print(average)
        prf_dict  = precision_recall(forPipeDF, out_col, column, preds_train,labels = [0,1,-1], avg = average)
        metrics_df = metrics_df.append(prf_dict, ignore_index=True)
        
    coverage_overlap.to_csv(f'{outputDir}/{column}_overlap.csv')
y_true = forPipeDF[column].to_numpy()
y_pred = forPipeDF[out_col].to_numpy()


# for lab in [[0,1], [0,1,-1]]:
#     labels = lab
#     for avg in ['micro', 'weighted', None]:
#         precision = metrics.recall_score(y_true, y_pred, labels = labels, average = avg)
#         recall=metrics.precision_score(y_true, y_pred, labels = labels, average = avg)
#         f1=metrics.f1_score(y_true, y_pred, labels = labels, average = avg)

#         new_row = {'labels': labels, 'label':f'{avg}_{column}', 'precision': precision, 'recall' :recall, 'f1':f1}
# #append row to the dataframe
#         metrics_df = metrics_df.append(new_row, ignore_index=True)

forPipeDF.to_csv(f'{outputDir}/{metrics_out}_data.csv')
metrics_df.to_csv(f'{outputDir}/{metrics_out}.csv')















#%%time
#scar and ischemia

###read in input parameters, cleaned csv and sentence filter, labels for nlp objects
print('reading in files and parameters')
###################################################INPUT PARAMETERS#########################
metrics_out = 'metrics_IschemiaScar'

df = pd.read_csv(root2csv + '\\' + csvfilename)
filt ='ischemi(a|c)|scar|lcx_disease|transmural|nontransmural|infarction|infarct'

orig_columns = [StudyIndex, 'Scar_Scar','Ischemia_Ischemia', FlowInfoPrefix]

example_key = 'Scar_Scar'

#input parameters for snorkel pipeline
with open(f'{inputdir}/IschemiaScar_input_parameters.json') as f:
    inputDatas = json.load(f)
    
retokenization_list = inputDatas[example_key]['retokenization_list']

labs = [FlowInfoPrefix, FlowInfoPrefix + '_Lemmas', FlowInfoPrefix + '_sents', FlowInfoPrefix + '_Lemmas_sents']
json_suffix = FlowInfoPrefix
##################################load nlp model#########################################
# Load English tokenizer, tagger, parser, NER and word vectors
print('loading nlp model')
# nlp = load_negex_model(model = "en_core_sci_lg", language = 'en_clinical', 
#                           chunks = ["no"], pseudo_negations=["with no improvement at rest",
#                                                              "with no improvement on rest",
#                                                              "not improve at rest", 
#                                                              "with no reversibility at rest",
#                                                              "with no improvement on the rest study",
#                                                              "with no significant improvement on rest imaging",
#                                                              "with no reversibility on rest"],
#                       preceding_negations =["rather than"])
# #set custom extensions
# Span.set_extension("negex", default=False, force=True)
# Token.set_extension("negex", default=False, force=True)


###########################################################################
print('processing nlp objects and nlp object keys')
#read in nlp objects
nlp_objects = {}
for index,lab in enumerate(labs):
    with open(f"{inputdir}/{lab}_nlp_objects.pickle", "rb") as handle:
        bytes_data = pickle.load(handle)
        doc_bin = DocBin().from_bytes(bytes_data)
        docs = list(doc_bin.get_docs(nlp.vocab))
        print(f'reading in {lab} with {len(docs)} documents')
    nlp_objects.update({index:docs})

print(nlp_objects.keys())        

# read in keys to rebuild sentences to docs
with open(f'{inputdir}/NLPpipeKeys_{json_suffix}.json') as f:
    Keys = json.load(f)
print(Keys.keys())

print('creating analytic df')
#make analytic df with nlp objects and merge back with original columns needed for evaluation

forPipeDF = make_analytic_df(df, eval_columns = orig_columns, nlp_objects =nlp_objects, nlp_Keys = Keys, filt = filt, retokenization_list = retokenization_list, StudyKey=StudyIndex)

print('running snorkel pipeline')
metrics_df = pd.DataFrame(columns=['labels','label','precision', 'recall', 'f1'])

for key in inputDatas:
    print(f'running {key} rules')
    pathologies = inputDatas[key]['pathology']
    column = inputDatas[key]['true_col']
    out_col =inputDatas[key]['predicted_column']
    inputs = inputDatas[key]['inputArgs']


    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        
    coverage_overlap, preds_train, forPipeDF = snorkel_pipeline(inputs, forPipeDF, out_col, column)
    prf_dict  = precision_recall(forPipeDF, out_col, column, preds_train)
    metrics_df = metrics_df.append(prf_dict, ignore_index=True)
    coverage_overlap.to_csv(f'{outputDir}/{column}_overlap.csv')

forPipeDF.to_csv(f'{outputDir}/{metrics_out}_data.csv')
metrics_df.to_csv(f'{outputDir}/{metrics_out}.csv')

