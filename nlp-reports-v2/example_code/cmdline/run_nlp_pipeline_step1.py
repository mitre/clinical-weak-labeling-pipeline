#!/usr/bin/env python


# *To run this pipeline:
# 
# - spaCy version greater than 3.0 neeeded
# 
# - ensure en_core_sci_scibert-0.4.0.tar is installed in a location your env can find. follow download instructions at https://github.com/allenai/scispacy
# 
# 


import sys
CONFIGFILE = sys.argv[0]




import pandas as pd
from conceptextract import keywds as kw
from conceptextract import path_fun as pf
from conceptextract import helper_fun as hf
from conceptextract import make_rules_from_config as mr
import yaml

rp = mr.RuleParameters(CONFIGFILE)


#run spacy pipeline and save objects 
#this only needs to be loaded once per kernel session. 


print('load nlp negex')
nlp = kw.load_negex_model_ud(model = rp.nlp_cfg['Language_model'], 
                       language = 'en_clinical', 
                       chunks = ["no"], 
                       pseudo_negations= rp.nlp_cfg['Pseudo_negations'],
                      preceding_negations=rp.nlp_cfg['Pseudo_negations'],
                      entity_pats =  rp.nlp_cfg['Entity_Patterns'])



data_df = pd.read_csv(os.path.join(rp.FILE_PATHS['data_directory'],rp.DATA_INFO['data_csv']))

if rp.FILE_PATHS['code_path_library']:
    pf.insert_nlp_lib_to_path(rp.FILE_PATHS['code_path_library'])
pf.create_dir(rp.FILE_PATHS['input_directory'])

tp = rp.DATA_INFO['data_columns_with_keywd_filter']

#checking the config is as expected
for column, filt in rp.DATA_INFO['data_columns_with_keywd_filter']:
    print(filt)

#run spacy pipeline and save to pickles for reuse by rule set and save 1 pickle object per data column processed


dicts = hf.run_nlp_pipeline(nlp, df = data_df, texts_to_parse = tp, study_index_col = rp.DATA_INFO['index_column'] ,
                         outputDir = rp.FILE_PATHS['input_directory'], b= 100, b2 =500)





