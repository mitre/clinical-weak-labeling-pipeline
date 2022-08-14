#%load_ext autoreload
#%autoreload 2

import pandas as pd
import json
import os
import inspect
import sys
from pathlib import Path

df_filepaths = pd.read_csv("VIP_FilePaths.csv")
Path2NLPLibrary = df_filepaths.at[0,'PathActual']
Path2NLPInputFiles = df_filepaths.at[1,'PathActual']
Path2Data = df_filepaths.at[2,'PathActual']
Path2OutputData = df_filepaths.at[3,'PathActual']


sys.path.insert(1, Path2NLPLibrary)

from lib.EncapLabelingFun import *

StudyIndex = 'Study_N'#name of columns with the study number of the patient
FlowInfoPrefix = 'Flow_Bullet'#column with clinical report used for NLP
PredFlowPrefix = 'PredNormAb'#predicted flow normal/abnormal column
csvfilename = '17segment_2008to2019 + Deidentified_mod_RC.csv'
interp_colum = 'Interp_Bin'#true flow normal/abnormal column
inputs_Dir = 'inputs'#name of folder to store input generated
data_Dir = 'data'#name of folder where data is stored

print(LABELFUNCTIONS)




# print a function to review
print(inspect.getsource(binary_find))




#create a directory to story the input arguments json


outputDir = os.path.join(Path2NLPInputFiles, inputs_Dir)
 
if not os.path.exists(outputDir):
    os.makedirs(outputDir)
        
        
print(outputDir)



#Build rules for multiple extractions with same rule set and save to json

#This is an example with the ischemia and vessel territory.

#create a file prefix for outfiles
fileprefix = 'IschemiaVessel'

#Identify the concepts to look for associations between. These are token representations of a 
#2 different ideas that you would like to determine an association between in the text
# An example:
#      'There is evidence of ischemia in the lcx' > returns 1 for LCX+Ischemia
#      'There is no significant evidence of ischemia in the LCX but there is evidence of ischemia in the LAD' > returns 0 for LCX+Ischemia and 1 for LCX+LAD
#

# Since we want to search for 3 area associations (LAD, RCA, LCX) with the disease ischemia, the input arguments are built as follows:
#Identify the pathologies as the first concept as a list of tokens that represent a given concept. 

concept1 = ["ischemia", "ischemic"]

#identify the regions as the second concept
concept2 = [["rca", "rca territory", "rca territories", "rca/lcx", "lcx/rca"],  
            ["lad", "lad territory", "lad territories"], 
            ["lcx", "lcx territory", "lcx territories", "lcx_disease", "rca/lcx", "lcx/rca", 
             "lcx/diagonal", "circumflex"]]#--------------------ch

#Next identify true columns. #todo: need to write case that deals with if no true labels
columns = ['Ischemia_RCA', 'Ischemia_LAD', 'Ischemia_LCX']

#Identify names of outcolumns that will be created and contain the predicted value (0,1) based on results from snorkel model.
out_cols = ['predicted_IschemiaRCA', 'predicted_IschemiaLAD', 'predicted_IschemiaLCX']

#Input list of tuples for retokenization. If none, leave as empty list.
retokenization = [('perfusion', 'defect'),
    ('transmural', 'scar'),
    ('nontransmural', 'scar'),
    ('perfusion', 'defect'),
    ('rca', 'territory'),
    ('rca', 'territories'),
    ('lcx', 'territory'),
    ('lcx', 'disease'),
    ('lcx', 'territories'),
    ('lad', 'territory'),
    ('lad', 'territories'),
    ('consistent', 'with'),
    ('rca', 'territory'),
    ('small', 'sized')]

jasons = {}
for i in range(3):
    region = concept2[i]
    column = columns[i]
    out_col = out_cols[i]
    retokenization = retokenization
    #write in arguments for snorkel pipeline functions as dictionary here. The available functions and default arguments can be found by 
    # running print(inspect.getsource(binary_find)) or opening EncapLabelingFun.py
                    #funtions 1 -4 search for mentions of 'Ischemia' and a vessel ['LCX', "RCA", 'LAD']
    inputs = { 1 : {'fun': 'keywd_associations', 'Dict': dict(series='nlp_string',wordlist1 = concept1, wordlist2 = region,ents=False, invert=False), 'suffix' : 'token'}, 
                      2:{'fun':'common_ancestor', 'Dict': dict(series='nlp_string',wordlist1 = concept1, wordlist2 = region, ents=False, invert=False), 'suffix' : 'token'}, 
                      3:{'fun':'keywd_find_two', 'Dict': dict(series='nlp_string',wordlist1 = concept1, wordlist2 = region, ents=True, invert=False), 'suffix' : 'ent'},
                      4:{'fun':'keywd_find_two', 'Dict': dict(series='nlp_string',wordlist1 = concept1, wordlist2 = region, ents=False, invert=False), 'suffix' : 'token'},
              
                #because we are simply looking for whether the term is there or not there, the invert argument can be true. 
                #functions 5-8 'invert' rules 1-4 and simply indicate that non-negated mentions of the associated phrases are not found. 
              
                      5:{'fun': 'keywd_associations', 'Dict': dict(series='nlp_string',wordlist1 = concept1, wordlist2 = region,ents=False, invert=True), 'suffix' : 'token_negation'},
                      6:{'fun':'common_ancestor', 'Dict': dict(series='nlp_string',wordlist1 = concept1, wordlist2 = region, ents=False, invert=True), 'suffix' : 'token_negation'},
                      7:{'fun':'keywd_find_two', 'Dict': dict(series='nlp_string',wordlist1 = concept1, wordlist2 = region, ents=True, invert=True), 'suffix' : 'ent_negation'},
                      8: {'fun':'keywd_find_two', 'Dict': dict(series='nlp_string',wordlist1 = concept1, wordlist2 = region, ents=False, invert=True), 'suffix' : 'token_negation'}
                }
    #this creates a dictionary with the snorkel function arguments for the 3 dual concept searches ['Ischemia_RCA', 'Ischemia_LAD', 'Ischemia_LCX']
    jasons.update({column: {'pathology': concept1, 'concept2': region, 'true_col': column, 'predicted_column': out_col, 'inputArgs': inputs , 'retokenization_list' : retokenization} })
    
#write dictionary to json for use later in snorkel pipeline
with open(outputDir + f'/{fileprefix}_input_parameters.json', 'w') as out:
    json.dump(jasons, out)
    
    
    
    
    


# Scar + Vessel Snorkel arguments
fileprefix = 'ScarVessel'
concept1 = ["scar", "transmural scar", "nontransmural scar", "transmural", "nontransmural", "infarction", "infarct", "transmural/nontransmural"]
concept2 = [["rca", "rca territory", "rca territories", "rca/lcx", "lcx/rca"],  
            ["lad", "lad territory", "lad territories"], 
            ["lcx", "lcx territory", "lcx territories", "lcx_disease", "rca/lcx", "lcx/rca", 
             "lcx/diagonal", "circumflex"]]#--------------------ch
columns = ['Scar_RCA', 'Scar_LAD', 'Scar_LCX']
out_cols = ['predicted_scarRCA', 'predicted_scarLAD', 'predicted_scarLCX']
retokenization = [('perfusion', 'defect'),
    ('transmural', 'scar'),
    ('nontransmural', 'scar'),
    ('perfusion', 'defect'),
    ('rca', 'territory'),
    ('rca', 'territories'),
    ('lcx', 'territory'),
    ('lcx', 'disease'),
    ('lcx', 'territories'),
    ('lad', 'territory'),
    ('lad', 'territories'),
    ('consistent', 'with'),
    ('rca', 'territory'),
    ('small', 'sized')]

jasons = {}
for i in range(3):
    region = concept2[i]
    column = columns[i]
    out_col = out_cols[i]
    retokenization = retokenization
    inputs = { 1 : {'fun': 'keywd_associations', 'Dict': dict(series='nlp_string',wordlist1 = concept1, wordlist2 = region,ents=False, invert=False), 'suffix' : 'token'}, 
                      2:{'fun':'common_ancestor', 'Dict': dict(series='nlp_string',wordlist1 = concept1, wordlist2 = region, ents=False, invert=False), 'suffix' : 'token'}, 
                      3:{'fun':'keywd_find_two', 'Dict': dict(series='nlp_string',wordlist1 = concept1, wordlist2 = region, ents=True, invert=False), 'suffix' : 'ent'},
                      4:{'fun':'keywd_find_two', 'Dict': dict(series='nlp_string',wordlist1 = concept1, wordlist2 = region, ents=False, invert=False), 'suffix' : 'token'},
                      5:{'fun': 'keywd_associations', 'Dict': dict(series='nlp_string',wordlist1 = concept1, wordlist2 = region,ents=False, invert=True), 'suffix' : 'token_negation'},
                      6:{'fun':'common_ancestor', 'Dict': dict(series='nlp_string',wordlist1 = concept1, wordlist2 = region, ents=False, invert=True), 'suffix' : 'token_negation'},
                      7:{'fun':'keywd_find_two', 'Dict': dict(series='nlp_string',wordlist1 = concept1, wordlist2 = region, ents=True, invert=True), 'suffix' : 'ent_negation'},
                      8: {'fun':'keywd_find_two', 'Dict': dict(series='nlp_string',wordlist1 = concept1, wordlist2 = region, ents=False, invert=True), 'suffix' : 'token_negation'}
                }
    jasons.update({column: {'pathology': concept1, 'concept2': region, 'true_col': column, 'predicted_column': out_col, 'inputArgs': inputs, 'retokenization_list' : retokenization } })
    

with open(outputDir + f"/{fileprefix}_input_parameters.json", 'w') as out:
    json.dump(jasons, out)







jasons = {}

#In this case I am combing two single concept searches as they use the same rule sets to search but different keyword search lists
#The arguments file the same but employs only single concept rules. Note that the rule structure does not have to be the same between concept searches
#but can easily be stored as one input arguments json for ease of use. These rule sets could also be stored as separate jsons.

fileprefix = 'IschemiaScar'

#Here input the tokens associated with the concept of interest.
concept1 = ["scar", "transmural scar", "nontransmural scar", "transmural", "nontransmural", "infarction", "infarct", "transmural/nontransmural"]

column = ['Scar_Scar']
out_col = ['predicted_Scar']
#here I am using the same retokenization list for both dictionaries, but this is not required.
retokenization = [('perfusion', 'defect'),
    ('transmural', 'scar'),
    ('nontransmural', 'scar'),
    ('perfusion', 'defect'),
    ('rca', 'territory'),
    ('rca', 'territories'),
    ('lcx', 'territory'),
    ('lcx', 'disease'),
    ('lcx', 'territories'),
    ('lad', 'territory'),
    ('lad', 'territories'),
    ('consistent', 'with'),
    ('rca', 'territory'),
    ('small', 'sized')]

inputs = { 1 : {'fun':'keywd_find', 'Dict': dict(series='nlp_string',keywords = concept1, ents=False, invert=False), 'suffix' : 'token'}, 
                     2:{'fun':'keywd_find', 'Dict': dict(series='lemma',keywords = concept1, ents=False, invert=False), 'suffix' : 'lemma_token'}, 
                      #3:{'fun':'keywd_find', 'Dict': dict(series='lemma',keywords = concept1, ents=True, invert=False), 'suffix' : 'ents'}, 
                      4:{'fun':'keywd_find', 'Dict': dict(series='lemma',keywords = concept1, ents=True, invert=False), 'suffix' : 'lemma_ents'},
        
                   5 : {'fun':'keywd_find', 'Dict': dict(series='nlp_string',keywords = concept1, ents=False, invert=True), 'suffix' : 'token_negation'}, 
                     6:{'fun':'keywd_find', 'Dict': dict(series='lemma',keywords = concept1, ents=False, invert=True), 'suffix' : 'lemma_token_negation'}, 
                     # 7:{'fun':'keywd_find', 'Dict': dict(series='lemma',keywords = concept1, ents=True, invert=True), 'suffix' : 'ents_negation'}, 
                      8:{'fun':'keywd_find', 'Dict': dict(series='lemma',keywords = concept1, ents=True, invert=True), 'suffix' : 'lemma_ents_negation'}  
                }

jasons.update({column[0]: {'pathology':concept1, 'true_col': column[0], 'predicted_column': out_col[0], 'inputArgs': inputs,
                        'retokenization_list' : retokenization } })
    

#Single concept: Ischemia
concept1 = ["ischemia", "lcx_disease"]
#in the case of ischemia, we wanted to search primarily for 'ischemia' or 'lcx_disease', but if the term ischemia is not present, 
# we search for non-negated mentions of the second term list. So the function keywd_two_conditional() takes two input lists

secondarylst = ["perfusion_defect", "perfusion defect"]

column = ['Ischemia_Ischemia']
out_col = ['predicted_Ischemia']

inputs = { 1 : {'fun':'keywd_find', 'Dict': dict(series='nlp_string',keywords = concept1, ents=False, invert=False), 'suffix' : 'token'}, 
                      2:{'fun':'keywd_find', 'Dict': dict(series='lemma',keywords = concept1, ents=False, invert=False), 'suffix' : 'lemma_token'}, 
                      3:{'fun':'keywd_find', 'Dict': dict(series='lemma',keywords = concept1, ents=True, invert=False), 'suffix' : 'ents'}, 
                      4:{'fun':'keywd_find', 'Dict': dict(series='lemma',keywords = concept1, ents=True, invert=False), 'suffix' : 'lemma_ents'},
                      5:{'fun':'keywd_two_conditional', 'Dict': dict(series='nlp_string', wordlist1 = concept1, wordlist2 = secondarylst, ents=False, invert=False), 'suffix' : '2step'},
        
                      6: {'fun':'keywd_find', 'Dict': dict(series='nlp_string',keywords = concept1, ents=False, invert=True), 'suffix' : 'token_negation'}, 
                      7:{'fun':'keywd_find', 'Dict': dict(series='lemma',keywords = concept1, ents=False, invert=True), 'suffix' : 'lemma_token_negation'}, 
                      8:{'fun':'keywd_find', 'Dict': dict(series='lemma',keywords = concept1, ents=True, invert=True), 'suffix' : 'ents_negation'}, 
                      9:{'fun':'keywd_find', 'Dict': dict(series='lemma', keywords = concept1, ents=True, invert=True), 'suffix' : 'lemma_ents_negation'},
                      10:{'fun':'keywd_two_conditional', 'Dict': dict(series='nlp_string', wordlist1 = concept1, wordlist2 = secondarylst, ents=False, invert=True), 'suffix' : '2step_negation'}
                     }
          
           
           
jasons.update({column[0]: {'pathology': concept1, 'true_col': column[0], 'predicted_column': out_col[0], 'inputArgs': inputs,
                        'retokenization_list' : retokenization } })

with open(outputDir + f"/{fileprefix}_input_parameters.json", 'w') as out:
    json.dump(jasons, out)
    
    
    
    
    
    
    
    
    
    
    
    
    
#normal abnormal
#The normal and abonrmal rule set is built differently. Here the snorkel functions are used to search for the presence of the term 'normal' or the term 'abnormal' for classification
#As such, Rules 5-8 do not invert the searches for normal (ie the word normal is not present in a given string). Instead they search for the term 'abnormal'


#scar with vessel territory
fileprefix = 'NormAbnorm'
concept1_a = ["normal"]
concept1_b = ["abnormal"]

column = [interp_colum]#column with binary 0/1 for norm/abnorm
out_col = ['predicted_overall']
retokenization = []

jasons = {}
          # functions 1-4 search for mentions of normal
inputs = { 1 : {'fun': 'regex_finder', 'Dict': dict(series=PredFlowPrefix,regex_pat = '^norm|[^ab]norm',invert=False, lab=0), 'suffix' : 'string'}, 
                    2:{'fun':'keywd_find', 'Dict': dict(series='nlp_string',keywords = concept1_a, ents=False, invert=False, labs = [0,1]), 'suffix' : 'token'}, 
                     3:{'fun':'keywd_find', 'Dict': dict(series='lemma',keywords = concept1_a, ents=False, invert=False, labs = [0,1]), 'suffix' : 'lemma_token'}, 
                      4:{'fun':'keywd_find', 'Dict': dict(series='lemma',keywords = concept1_a, ents=True, invert=False, labs = [0,1]), 'suffix' : 'ents'}, 
                #functions 5-8 search for mentions of abnormal
                5: {'fun': 'regex_finder', 'Dict': dict(series=PredFlowPrefix,regex_pat = '^abnorm|abnormal',invert=False, lab=1), 'suffix' : 'string_abnormal'}, 
                     6:{'fun':'keywd_find', 'Dict': dict(series='nlp_string',keywords = concept1_b, ents=False, invert=False), 'suffix' : 'token_abn'}, 
                     7:{'fun':'keywd_find', 'Dict': dict(series='lemma',keywords = concept1_b, ents=False, invert=False), 'suffix' : 'lemma_token_abn'}, 
                       8:{'fun':'keywd_find', 'Dict': dict(series='lemma',keywords = concept1_b, ents=True, invert=False), 'suffix' : 'ents_abn'},    
              
                }
jasons.update({column[0]: {'wordlist1': concept1_a, 'wordlist2': concept1_b, 'true_col': column[0], 'predicted_column': out_col[0], 'inputArgs': inputs}})
                        #'retokenization_list' : retokenization } }
    

with open(outputDir + f"/{fileprefix}_input_parameters.json", 'w') as out:
    json.dump(jasons, out)
