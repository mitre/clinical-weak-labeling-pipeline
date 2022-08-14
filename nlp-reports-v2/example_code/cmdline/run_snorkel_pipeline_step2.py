#!/usr/bin/env python



import sys
CONFIGFILE = sys.argv[0]
RULESET = sys.argv[1]
CONCEPT_NUM = sys.argv[2]



import yaml
import pandas as pd
from conceptextract import keywds as kw
from conceptextract import path_fun as pf
from conceptextract import helper_fun as hf
from conceptextract import make_rules_from_config as mr



#load spacy pipeline
#this only needs to be loaded once per kernel
rp_nlp = mr.RuleParameters(CONFIGFILE)

print('load nlp negex')
nlp = kw.load_negex_model_ud(model = rp_nlp.nlp_cfg['Language_model'], 
                       language = 'en_clinical', 
                       chunks = ["no"], 
                       pseudo_negations= rp_nlp.nlp_cfg['Pseudo_negations'],
                      preceding_negations=rp_nlp.nlp_cfg['Pseudo_negations'],
                      entity_pats =  rp_nlp.nlp_cfg['Entity_Patterns'])



#Run
#first initialize rule parameters that contain all settings needed for a given rule set
rp = mr.RuleParameters(CONFIGFILE,rule_set= RULESET, num = CONCEPT_NUM)

snorkelrules =rp.run_LF_maker()

#parameters for the snorkel run. 

input_path = rp.FILE_PATHS['input_directory']
data_df = pd.read_csv(os.path.join(rp.FILE_PATHS['data_directory'],rp.DATA_INFO['data_csv']))
data_column = rp.metadata['data_column']
index_column =rp.DATA_INFO['index_column'] 
eval_columns = [index_column, data_column] +list(snorkelrules.keys())
filt = rp.sent_filt
retokenization_list = rp.retokenization_list
filePat = rp.metadata['out_prefix']
output_path = rp.FILE_PATHS['output_directory']

    
#loads previously saved nlp object
analytic_df = hf.load_nlp_pickles_to_df(nlp, input_path, data_df, data_column,
                                         eval_columns, filt, retokenization_list, index_column)
    
#runs snorkel pipeline creating precision and recall outputs
hf.run_snorkel_pipeline(snorkelrules, analytic_df, filePat, output_path, GoldStd = True)
    
  





