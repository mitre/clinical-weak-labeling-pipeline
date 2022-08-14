#!/usr/bin/env python
# coding: utf-8

# This  creates the config.yaml file necessary to run the NLP pipeline. One can make changes using this notebook or by modifying the example config.yaml in the working directory. 
# 
# There are two componenets to the config file:
# 
#         - 1)There are several parts to the config. The first deals with the set up of the nlp pipeline general to all rules and nlp processes for a given data set and file paths. Those are in 'FILE_PATHS': 
#                  'DATA', and  'NLP_CFG':NLP_CFG,
#                
#         
#         - 2) the specific rule sets arguments stored as dictionary of dictionarys. The rule sets contain a set of directions for the snorkel pipeline to extract a set of concepts with a given set of rules. One or multiple keyword/concepts can be extracted with 1 rule set. Each rule set can extract single or dual concepts but not both. These are run once per rule set but there can be multiple rule sets per data set depending on the number of concepts to extract. If you wish to include multiple rule sets,  input additional dictionary containing the arguments in the RuleSets dictionary. The format must follow that specified in the examples. The current ruleset examples are stored in: 'RuleSets': {'IschemiaVessel': ISCH_VES_DICT, 'Ischemia': ISCH_DICT, 'Overall':OVERALL_DICT 



#string specifying name of  config file to be generated
import sys
CONFIGFILE = sys.argv[0]



from omegaconf import OmegaConf
from conceptextract import make_rules_from_config as mr




# Required: Load English tokenizer, tagger, parser, NER and word vectors to create nlp objects
#options are: "en_core_sci_sm", "en_core_sci_lg", "en_core_web_sm",  "en_core_web_lg"


Lang_model = "en_core_sci_scibert"



#optional list of pseudo negations. if none, comment out

Pseudo_negations=["with no improvement at rest",
                                         "with no improvement on rest",
                                         "not improve at rest", 
                                         "with no reversibility at rest",
                                         "with no improvement on the rest study",
                                         "with no significant improvement on rest imaging",
                                         "with no reversibility on rest"]

#optional list of pseudo negations.  
Preceding_negations=["rather than"]
#optional list of entity patterns. see spaCy documentation.  
Entity_pats =  [{"label": "GPE", "pattern": "ischemic"},
            {"label": "GPE", "pattern": "ischemia"},
            {"label": "GPE", "pattern": "rca"},
            {"label": "GPE", "pattern": "lad"},
            {"label": "GPE", "pattern": "lcx"},
            {"label": "GPE", "pattern": "rca/lcx"},
            {"label": "GPE", "pattern": "lcx/rca"},
            {"label": "GPE", "pattern": "lcx_disease"},
            {"label": "GPE", "pattern": "lcx/diagonal"},
            {"label": "GPE", "pattern": "circumflex"},
            {"label": "GPE", "pattern": [{"lower": "lcx"}, {"lower": "disease"}]},
            {"label": "GPE", "pattern": [{"lower": "rca"}, {"lower": "territory"}]},
            {"label": "GPE", "pattern": [{"lower": "rca"}, {"lower": "territories"}]},
            {"label": "GPE", "pattern": [{"lower": "lad"}, {"lower": "territory"}]},
            {"label": "GPE", "pattern": [{"lower": "lad"}, {"lower": "territories"}]},
            {"label": "GPE", "pattern": [{"lower": "lcx"}, {"lower": "territory"}]},
            {"label": "GPE", "pattern": [{"lower": "lcx"}, {"lower": "territories"}]},
            {"label": "GPE", "pattern": "scar",},
            {"label": "GPE", "pattern": [{"lower": "transmural"}, {"lower": "scar"}]},
            {"label": "GPE", "pattern": [{"lower": "nontransmural"}, {"lower": "scar"}]}, 
            {"label": "GPE", "pattern": "transmural"},
            {"label": "GPE", "pattern": "nontransmural"},
            {"label": "GPE", "pattern": "infarction"},
            {"label": "GPE", "pattern": "infarct"},
            {"label": "GPE", "pattern": "transmural/nontransmural"},
            {"label": "GPE", "pattern": [{"lower": "perfusion"}, {"lower": "defect"}]}]



Retokenization = [('perfusion', 'defect'),
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




#Identify the concepts to look for associations between. These are token representations of a 
#2 different ideas that you would like to determine an association between in the text
# An example:
#      'There is evidence of ischemia in the lcx' > returns 1 for LCX+Ischemia
#      'There is no significant evidence of ischemia in the LCX but there is evidence of ischemia in the LAD' > returns 0 for LCX+Ischemia and 1 for LCX+LAD
#

# Since we want to search for 3 area associations (LAD, RCA, LCX) with the disease ischemia, the input arguments are built as follows:
#Identify the pathologies as the first concept as a list of tokens that represent a given concept. 

#outfile prefix
outfile_prefix = 'IschemiaVessel'

ConceptDict = {'concept1' : [["ischemia", "ischemic"]],

#identify the regions as the second concept
'concept2' : [["rca", "rca territory", "rca territories", "rca/lcx", "lcx/rca"],  
            ["lad", "lad territory", "lad territories"], 
            ["lcx", "lcx territory", "lcx territories", "lcx_disease", "rca/lcx", "lcx/rca", 
             "lcx/diagonal", "circumflex"]],#--------------------ch

#Next identify true columns and rule names; in same order as the second concept
#if no true columns, enter None in true columns, input only rule_names
'true_columns' : ['Ischemia_RCA', 'Ischemia_LAD', 'Ischemia_LCX'],
'rule_names' : ['Ischemia_RCA', 'Ischemia_LAD', 'Ischemia_LCX']}

#add documentation for rulesdict are from EncapLabelingFun.py. Necessary argument for each rule used can be visualized by:
# > print(LABELFUNCTIONS) 
# > print(inspect.getsource(<label function>))
#todo: write custom rule adder. For now, you could add your own labeling functions to EncapLabelingFun.py following Snorkel labeling rule conventions

RulesDict = {
    1: mr.make_2concept_rule_dict('keywd_associations', False, False, 'nlp_doc', ('concept1', 'concept2')),
    2: mr.make_2concept_rule_dict('common_ancestor', False, False, 'nlp_doc', ('concept1', 'concept2')),
    3: mr.make_2concept_rule_dict('keywd_find_two', True, False, 'nlp_doc', ('concept1', 'concept2')),
    4: mr.make_2concept_rule_dict('keywd_find_two', False, False, 'nlp_doc', ('concept1', 'concept2')),
    5: mr.make_2concept_rule_dict('keywd_associations', False, True, 'nlp_doc', ('concept1', 'concept2')),
    6: mr.make_2concept_rule_dict('common_ancestor', False, True, 'nlp_doc', ('concept1', 'concept2')),
    7: mr.make_2concept_rule_dict('keywd_find_two', True, True, 'nlp_doc', ('concept1', 'concept2')),
    8: mr.make_2concept_rule_dict('keywd_find_two', False, True, 'nlp_doc', ('concept1', 'concept2'))
    
}



ISCH_VES_DICT = {'Concepts': ConceptDict, 'Rules' :RulesDict, 'num_concepts': 2, 'type': 'complement',
                 'out_prefix': outfile_prefix, 'data_column': 'Bullet3'}




#single concept search looking for presence of absence of a concept
#eg. Ischemia is present or not present. We search only for the presence of a single concept. 

outfile_prefix = 'Ischemia'

#include list of terms for primary search. 
#If using a conditional keyword search (eg. If terms in list 1 not present, search for terms in list 2), include secondary list, else leave secondary_terms as empty list
ConceptDict = {'concept1' : ["ischemia", "ischemic", 'lcx_disease'], 'concept2' : ["perfusion_defect", "perfusion defect"],


#Next identify true columns and rule names
#if no true columns, enter None in true columns, input only rule_names
'true_columns' : ['Ischemia_Ischemia'],
'rule_names' : ['Ischemia_Ischemia']}

#add documentation for rulesdict 
RulesDict = {  
    1: mr.make_kwrule_dict('keywd_find', False, False, 'nlp_doc', 'concept1'),
    2: mr.make_kwrule_dict('keywd_find', False, False, 'lemma_doc', 'concept1'),                  
    3: mr.make_kwrule_dict('keywd_find', True, False, 'lemma_doc', 'concept1'),
    #4: mr.make_kwrule_dict('keywd_find', True, False, 'lemma_doc', 'concept1'),
    4:  mr.make_2concept_rule_dict('keywd_two_conditional', False, False, 'nlp_doc', ('concept1', 'concept2')),
    5: mr.make_kwrule_dict('keywd_find', False, True, 'nlp_doc', 'concept1'),
    6: mr.make_kwrule_dict('keywd_find', False, True, 'lemma_doc', 'concept1'),                  
    7: mr.make_kwrule_dict('keywd_find', True, True, 'lemma_doc', 'concept1'),
    #9: mr.make_kwrule_dict('keywd_find', True, True, 'lemma_doc', 'concept1'),
    9:  mr.make_2concept_rule_dict('keywd_two_conditional', False, True, 'nlp_doc', ('concept1', 'concept2'))}



ISCH_DICT = {'Concepts': ConceptDict, 'Rules' :RulesDict, 'num_concepts': 1, 'type': 'complement', 
                 'out_prefix': outfile_prefix, 'data_column': 'Bullet3'}


#single concept search looking for presence of termA or presence of termB
#eg. The word Normal is present. Opposite: The word abnormal is present. OR The term 'short' is present. The term 'tall' is present. We search for a concept and its opposite (short:tall, high:low etc). 
outfile_prefix = 'Overall'

#include list of terms for primary search. 
#If using a conditional keyword search (eg. If terms in list 1 not present, search for terms in list 2), include secondary list, else leave secondary_terms as empty list
ConceptDict = {'concept1' :["normal"], 'concept2' : ["abnormal"], 


#Next identify true columns and rule names
#if no true columns, enter None in true columns, input only rule_names
'true_columns' : ['Interp2'],
'rule_names' : ['Interp2']}

#add documentation for rulesdict 
RulesDict = {  
    1: mr.make_regex_rule_dict('regex_finder', '^norm|[^ab]norm', False, 'text'),
    2: mr.make_kwrule_dict('keywd_find', False, False, 'nlp_doc', 'concept1'),
    3: mr.make_kwrule_dict('keywd_find', False, False, 'lemma_doc', 'concept1'),
    4: mr.make_kwrule_dict('keywd_find', True, False, 'lemma_doc', 'concept1'),
    5: mr.make_regex_rule_dict('regex_finder', '^norm|[^ab]norm', False, 'text'),
    6: mr.make_kwrule_dict('keywd_find', False, False, 'nlp_doc', 'concept2'),
    7: mr.make_kwrule_dict('keywd_find', False, False, 'lemma_doc', 'concept2'),
    8: mr.make_kwrule_dict('keywd_find', True, False, 'lemma_doc', 'concept2')

}


OVERALL_DICT = {'Concepts': ConceptDict, 'Rules' :RulesDict, 'num_concepts': 1, 'type': 'opposite',
                 'out_prefix': outfile_prefix, 'data_column': 'Bullet1'}



#file paths  variables to feed to yaml

#code_path_library: manual path direction to nlp library if necessary. default none
#adds your nlp library (eg the spacy packages to your path... optional arg)
#input directory: stores intermediate pickle of nlp objects and jsons containing rule
#set arguments. 
#outputs will contain the results precision/recall and internal consistency
#for each rule set.
#Data directory is where the cleaned data csv is stored.


FILE_PATHS = {'input_directory': "trial/inputs", 'output_directory': 'trial/outputs', 
        'code_path_library': None , 'data_directory': 'trial/data'}


# #Data_csv: the csv containing an index_column and columns with the text to process
# through the spaCy pipeline. Optionally include keyword filters to reduce processing
# time.

DATA = {'data_csv':'ClinicalTextParseLV_EX_demo2.csv', 
       'index_column': 'Study_No', 
        'data_columns_with_keywd_filter': [('Bullet3','ischemi(a|c)|scar|lcx_disease|transmural|nontransmural|infarction|infarct'), 
                                           ('Bullet1', '')]}
#Input agruments for custom spaCy pipeline
#language model: options are: "en_core_sci_sm", "en_core_sci_lg", "en_core_web_sm",  "en_core_web_lg"
# preceding/pseudo_negations: list of additional phrases to include
# entity_patterns:optional. see the spaCy documentation
#retokenization: optional list.  common phrases to be considered as. single token

NLP_CFG = {'Language_model': Lang_model,
             'Pseudo_negations': Pseudo_negations,
             'Preceding_negations': Preceding_negations,
             'Entity_Patterns': Entity_pats,
             'Retokenization_list': Retokenization}

Overall_Config = { 'FILE_PATHS': FILE_PATHS,
                 'DATA': DATA,
                 'NLP_CFG':NLP_CFG,
                  'RuleSets': {'IschemiaVessel': ISCH_VES_DICT, 'Ischemia': ISCH_DICT, 'Overall':OVERALL_DICT }}

conf = OmegaConf.create(Overall_Config)

#write to file

with open(CONFIGFILE, 'w') as file:
    OmegaConf.save(config=conf, f=file)




