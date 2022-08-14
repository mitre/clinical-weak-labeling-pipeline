#%load_ext autoreload
#%autoreload 2





import logging
import sys
import os
import spacy
import re
import pandas as pd
import sys
from pathlib import Path

df_filepaths = pd.read_csv("VIP_FilePaths.csv")
Path2NLPLibrary = df_filepaths.at[0,'PathActual']
Path2NLPInputFiles = df_filepaths.at[1,'PathActual']
Path2Data = df_filepaths.at[2,'PathActual']
Path2OutputData = df_filepaths.at[3,'PathActual']

sys.path.insert(1, Path2NLPLibrary)#inserts it into the workflow

from lib.helper_fun import *


StudyIndex = 'Study_N'#name of columns with the study number of the patient
FlowInfoPrefix = 'Flow_Bullet'#column with clinical report used for NLP
PredFlowPrefix = 'PredNormAb'#predicted flow normal/abnormal column
csvfilename = '17segment_2008to2019 + Deidentified_mod_RC.csv'
interp_colum = 'Interp_Bin'#true flow normal/abnormal column
inputs_Dir = 'inputs'#name of folder to store input generated
data_Dir = 'data'#name of folder where data is stored


import os

script_folder = os.path.dirname(os.path.realpath(__file__))
print(script_folder)

# Load English tokenizer, tagger, parser, NER and word vectors to create nlp objects

nlp = load_negex_model(model = "en_core_sci_lg", 
                       language = 'en_clinical', 
                       chunks = ["no"], 
                       pseudo_negations=["with no improvement at rest",
                                         "with no improvement on rest",
                                         "not improve at rest", 
                                         "with no reversibility at rest",
                                         "with no improvement on the rest study",
                                         "with no significant improvement on rest imaging",
                                         "with no reversibility on rest"],
                      preceding_negations=["rather than"])






#%%time

#files
#load a dataframe containing the 'clean' strings for processing through the nlp pipeline
outputDir = os.path.join(Path2NLPInputFiles, inputs_Dir)
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

root2csv = os.path.join(Path2Data, data_Dir)
df = pd.read_csv(root2csv + '\\' + csvfilename)

#This is the sentence filter. Only sentences in a given document will be included in the processing if they match the below regex pattern
#For more information on sentences inside a doc see: https://spacy.io/api/doc
filt ='ischemi(a|c)|scar|lcx_disease|transmural|nontransmural|infarction|infarct'


        
#batch sizes for doc and targets column for nlp processing to pickle. Varying b and b2 will change the processing speed. 
#they are set to values that give a reasonable processing time.  
# For more info on optimizing stream processing see: https://spacy.io/api/language or
# https://prrao87.github.io/blog/spacy/nlp/performance/2020/05/02/spacy-multiprocess.html

b = 100
b2 = 500
texts = df[FlowInfoPrefix].astype('str').tolist()
studyid = df[StudyIndex].tolist()
labs = [FlowInfoPrefix, FlowInfoPrefix + '_Lemmas', FlowInfoPrefix + '_sents', FlowInfoPrefix + '_Lemmas_sents'] #output prefixes
json_suffix = FlowInfoPrefix



#This processes the strings input to spacy docs and then individually processes the sentences from a given document. Performance of negation
#and concept association improves when spacy evaluates smaller text segments.
# returns [text_docs,lemma_docs,sentence_docs, lemma_sent_docs] and {'StudyId': studyid, 'text_sentence_keys': text_keys,'lemma_sentence_keys': lemma_keys}         
nlpobjects, StudyIdData = stream_nlp_process(texts, nlp, studyid, filt, b, b2)

#This creates a 1) a list of bytes objects that have been serialized and 2) a pickle object containing the serialized objects
#There is one pickle for the sentences and one for the documents for both the nlp run on the string and the lemmatized string.
#All four objects are needed to build the analytic dataframe that feeds the snorkel ruleset
byter = []
jasons = {}
for lab,docs in zip(labs, nlpobjects):
    bytes_data = pickle_nlp_objects(outputDir,lab,docs, nlp)
    for doc in docs:
        byter.append(bytes_data)


        
            
nlp_keys_to_json(outputDir, StudyIdData, json_suffix)





# Deserialize later, e.g. in a new process and verify process
#this block of code lets you check the serialization and deserialization process worked as expected.
#take care to check any custom extensions
nlp2 = spacy.blank("en")
doc_bin = DocBin().from_bytes(byter[0])
docs = list(doc_bin.get_docs(nlp.vocab))


for e in docs[22].doc:
    print(e, e._.negex)
    
    
    
    
#%%time


#This runs the nlp processing pipeline on the Bullet that contains mentions of normal or abnormal
#files
df = pd.read_csv(root2csv + '\\' + csvfilename)
#bullet 1 filter 
filt =''

        
#batch sizes for doc and targets column for nlp processing to pickle
b = 100
b2 = 500
texts = df[PredFlowPrefix].astype('str').tolist()
studyid = df[StudyIndex].tolist()
labs = [PredFlowPrefix, PredFlowPrefix + '_Lemmas', PredFlowPrefix + '_sents', PredFlowPrefix + '_Lemmas_sents'] #output prefixes
json_suffix = PredFlowPrefix


if not os.path.exists(outputDir):
        os.makedirs(outputDir)

# returns [text_docs,lemma_docs,sentence_docs, lemma_sent_docs] and {'StudyId': studyid, 'text_sentence_keys': text_keys,'lemma_sentence_keys': lemma_keys}         
nlpobjects, StudyIdData = stream_nlp_process(texts, nlp, studyid, filt, b, b2)

byter_1 = []
jasons = {}
for lab,docs in zip(labs, nlpobjects):
    bytes_data = pickle_nlp_objects(outputDir,lab,docs, nlp)
    for doc in docs:
        byter_1.append(bytes_data)


        
            
nlp_keys_to_json(outputDir, StudyIdData, json_suffix)
