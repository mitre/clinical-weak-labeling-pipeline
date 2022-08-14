import pandas as pd

from spacy.pipeline import EntityRuler
from spacy.tokens import Token, Doc, Span
from spacy.matcher import PhraseMatcher
from spacy.matcher import Matcher
from spacy.tokens import DocBin

import json
import pickle
import glob
from tqdm import tqdm

from .keywds import *
from .EncapLabelingFun import *
from .spacy_extras import *

def make_analytic_df(df, eval_columns, nlp_objs_dict, filt, retokenization_list =[], StudyKey='Study_No'):
    """
    Function: 
    ----------
        make_analytic_df
    

    Description:
    ----------
        creates a pandas dataframes with keyword processes in a column which sets up for processing
        the NLP pipeline
        

    Inputs: 
    ----------
        df: pandas dataframe opened with pd.read_csv
        nlp_objs_dict of form: dict{'docs': {'ids': lst, texts: lst of nlp docs}} with keys: docs, sentences_docs, lemma_docs, lemma_sentences_docs
        nlp_Keys: keys saved as json files from function wrapp_fun2json
        filt: regex filter should be the same as the one used in the nlp_pipe streamed to create the nlp objects
        retokenization_list: saved as json files from function wrapp_fun2json
        StudyKey: name of column with the study number of the patient
        

    Outputs:
    ----------
        a pandas dataframe will a column of keyword process objects for regular text and the lemmitized
        varient of that text
        
    """
    sentences_dfs = {}

    for key in [x for x in nlp_objs_dict.keys() if 'sentences' in x]:
        print(key,' has lens ids:', len(nlp_objs_dict[key]['ids']),' & text:', len(nlp_objs_dict[key]['texts']))
        temp = pd.DataFrame({StudyKey: nlp_objs_dict[key]['ids'], 'sentence_list': nlp_objs_dict[key]['texts']}) 
        temp = temp.groupby([StudyKey]).agg(lambda x: tuple(x)).applymap(list).reset_index()
        sentences_dfs[key] = temp

    doc_df = pd.DataFrame({StudyKey: nlp_objs_dict['docs']['ids'], 'docs':nlp_objs_dict['docs']['texts']}).merge(sentences_dfs['docs_sentences'], how = 'left')
    
    lemma_df = pd.DataFrame({StudyKey:nlp_objs_dict['docs_lemma']['ids'], 'docs':nlp_objs_dict['docs_lemma']['texts']}).merge(
                        sentences_dfs['docs_sentences_lemmas'], how = 'left')
    
    alldocs = doc_df.merge(lemma_df, on= StudyKey, how= 'left', suffixes = ("_text", "_lemma"))
    
    alldocs['nlp_string'] = ''
    alldocs['lemma_string'] = ''

    #add keyword processes for each document in the pandas dataframe
    #includes both regular text and its lemmitized variant
    for i, row in alldocs.iterrows():
        kwp =KeywordProcess(row['docs_text'], filt, row['sentence_list_text'])
        kwp_l =KeywordProcess(row['docs_lemma'], filt, row['sentence_list_lemma'])
        if retokenization_list:
            for tup in retokenization_list:
                kwp.retokenize_sentences(tup[0], tup[1])
                kwp_l.retokenize_sentences(tup[0], tup[1])
        alldocs.at[i,'nlp_string'] = kwp
        alldocs.at[i,'lemma_string'] = kwp_l
    
    
    return df[eval_columns].merge(alldocs, how= 'left')


def stream_nlp_process(texts, nlp, studyid, filt, doc_batch, sent_batch):
    """
    Function: 
    ----------
        stream_nlp_process
    

    Description:
    ----------
        creates nlp objects and study id data used to save to jason files to be later used
        for NLP processing
        

    Inputs: 
    ----------
        texts: list of clinical reports to be sent through NLP, can make it by loading all reports with df = pd.read_csv then doing: texts = df['colunWreports'].astype('str').tolist()
        nlp: nlp generated with load_negex_model
        studyid: list of study ID's (patient number): studyid = df['columnWstudyIDs'].tolist()
        filt: regex sentence filter
        doc_batch: document batch size (eg 100)
        sent_batch: sentence batch size (eg 500)
        

    Outputs:
    ----------
        nlpobjects and StudyIdData to be saved as json file
        
    """
    #breaking down to lists to allow stream processing speeds process considerably when running full data set
    text_sentences = []
    text_keys = []
    text_docs = process_chunk(texts, nlp, batch_size = doc_batch)
    print('chunked docs')
    for index, doc in tqdm(zip(studyid, text_docs)):
        sents, i = split_sents(doc, index, nlp, filt)
        text_keys.extend(i)
        text_sentences.extend(sents)

    
    sentence_docs = process_chunk(text_sentences, nlp, batch_size = sent_batch)
    print('chunked sents')
    #creates lemma nlp objects   
    lemma_sentences = []
    lemma_keys = []
    lemma_pre = [lemmitize_sentence(doc) for doc in text_docs]
    
    lemma_docs = process_chunk(lemma_pre,nlp, batch_size = doc_batch)
    print('chunked lemmas')
    for index, doc in tqdm(zip(studyid, lemma_docs)):
        sents, i = split_sents(doc, index, nlp, filt)
        lemma_keys.extend(i)
        lemma_sentences.extend(sents)
    
    lemma_sent_docs = process_chunk(lemma_sentences, nlp, batch_size = sent_batch)
    print('chunked lemma sents')

    d = {'docs': {'text': text_docs, 'ids': studyid}, 'docs_lemma': {'text': lemma_docs, 'ids': studyid},
            'docs_sentences': {'text': sentence_docs, 'ids': text_keys},
            'docs_sentences_lemmas' : {'text': lemma_sent_docs, 'ids': lemma_keys}}
    assert d['docs']['text'] == text_docs

    return d


def pickle_nlp_objects(outputDir,doc_dict, data_column, nlp):
    """
    Function: 
    ----------
        pickle_nlp_objects
    

    Description:
    ----------
        function designed to preserve (pickle) nlp objects for later use
        this significantly speeds up NLP processing
        

    Inputs:
    ----------
        outputDir: directory to store pickled data
        lab: tuple list of (python zip()) data types to pickle eg [clincial_data, clincial_data_lemmas, clincial_data_sentences, clincial_data_sentences_lemmas] 
        dict of docs and ids for pickling from stream_nlp_process
        nlp: nlp generated with load_negex_model
        

    Outputs:
    ----------
        pickled nlp object
        
    """
    bytes_dicts = {}
    for key in doc_dict:
        lang = nlp.meta["lang"]
        pipeline = nlp.meta["pipeline"]
        bytes_data = nlp.to_bytes()
        doc_bin = DocBin(attrs = ["ID",
    "ORTH",
    "LOWER",
    "NORM",
    "SHAPE",
    "PREFIX",
    "SUFFIX",
    "LENGTH",
    "CLUSTER",
    "LEMMA",
    "POS",
    "TAG",
    "DEP",
    "ENT_IOB",
    "ENT_TYPE",
    "ENT_ID",
    "ENT_KB_ID",
    "HEAD",
    "SPACY",
    "PROB",
    "LANG",
    "IDX"], store_user_data = True)
        
        print(key)
        for doc in doc_dict[key]['text']:
            #print(doc.text)
            doc_bin.add(doc)
        
        print(key, 'no of docs',len(doc_bin))
        print(key, 'len keys', len(doc_dict[key]['ids']))
        bytes_data = doc_bin.to_bytes()
        doc_dict[key]['bytes'] = bytes_data
        bytes_dict = {k: doc_dict[key][k] for k in ('ids', 'bytes')}
        bytes_dicts[key] = bytes_dict
    print(bytes_dicts.keys())        
    with open(f"{outputDir}/{data_column}_nlp_objects.pickle","wb") as handle:
        pickle.dump(bytes_dicts, handle)
    return f'{data_column} pickled and saved in {outputDir}'


# def nlp_keys_to_json(outputDir, keysDict, lab):
#     """
#     Function: 
#     ----------
#         nlp_keys_to_json
    

#     Description:
#     ----------
#         stores the study id data from the stream_nlp_process function
#         to ma json file for later processing
        

#     Inputs: 
#     ----------
#         outputDir: directoy to store the json files
#         keysDict: StudyIdData from stream_nlp_process
#         lab: userdefined tag to add to the file name to identify it
        

#     Outputs:
#     ----------
#         jason file with study id data from the stream_nlp_process function
        
#     """
#     with open(f'{outputDir}/NLPpipeKeys_{lab}.json', 'w') as out:
#         json.dump(keysDict, out)



def run_nlp_pipeline(nlp, df, texts_to_parse, study_index_col, outputDir, b= 100, b2 =500):
    """
    texts_to_parse -> tuple of strings containing column to parse and keyword filter eg. ('clinical note', 'heart|vessel')
    df  -> pandas data frame containing columns below
    study_index_col -> str indicating column in df containing unique id
    #b, b2 -> int batch sizes for doc and targets column for nlp processing to pickle. Varying b and b2 will change the processing speed. 
    #they are set to values that give a reasonable processing time.  
    # For more info on optimizing stream processing see: https://spacy.io/api/language or
    # https://prrao87.github.io/blog/spacy/nlp/performance/2020/05/02/spacy-multiproces
    outputDir -> str indicating where to save pickles and pickle metadata
    
    out > pickle objects saved in output directory of nlp trees for all texts and dicts with nlp objects processed for each data column
    """
    evals = []
    for column, filt in texts_to_parse: 
        data_column = column
        keyword_filter = filt
        print('processing ', data_column, 'with filter ', filt)
        texts = df[data_column].astype('str').tolist()
        studyid = df[study_index_col].tolist()
    
    #labs = [data_column, data_column + '_Lemmas', data_column + '_sents', data_column + '_Lemmas_sents'] #output prefixes
    #json_suffix = data_column
        print('stream nlp process')
       
        doc_dict = stream_nlp_process(texts, nlp, studyid, keyword_filter, b, b2)
        bytes_dict = pickle_nlp_objects(outputDir,doc_dict, column, nlp)
        evals.append(doc_dict)
    return evals




def load_nlp_pickles_to_df(nlp, INPUT_PATH, df, data_column, eval_columns,filt, retokenization_list, index_column):
    pickle_files = glob.glob(INPUT_PATH + f'/{data_column}*.pickle')
    print('these are the pickles to unpickle: ' ,pickle_files)
    print('processing nlp objects and nlp object keys')
    

    for file in pickle_files:
        with open(file, "rb") as handle:
            unpickle = pickle.load(handle)
        for key in unpickle:
            doc_bin = DocBin().from_bytes(unpickle[key]['bytes'])
            docs = list(doc_bin.get_docs(nlp.vocab))
            print('reading in docs of len', len(docs), 'from', key)
            unpickle[key]['texts'] = docs
    print('creating analytic df for', data_column, ' with ', filt)
    # #make analytic df with nlp objects and merge back with original columns needed for evaluation
    forPipeDF = make_analytic_df(df, eval_columns = eval_columns, 
                                 nlp_objs_dict = unpickle,  filt = filt, 
                                 retokenization_list = retokenization_list, StudyKey=index_column)
    return forPipeDF


def run_snorkel_pipeline(snorkel_param, eval_df, filePat, outputDir, GoldStd = True):
    print('running snorkel pipeline')

    metrics_df = pd.DataFrame(columns=['labels','label','precision', 'recall', 'f1'])
    
    
    for key in snorkel_param:
        print(f'running {key} rules')
        rule_name = key
        out_col =snorkel_param[key]['predicted_column']
        inputs = snorkel_param[key]['inputArgs']
        true_column = snorkel_param[key]['true_col']

        print(eval_df.columns)
        coverage_overlap, preds_train, eval_df = snorkel_pipeline(inputs, eval_df, out_col, rule_name)
        if GoldStd:
            prf_dict  = precision_recall(eval_df, out_col, true_column, preds_train)
            metrics_df = metrics_df.append(prf_dict, ignore_index=True)
        
        coverage_overlap.to_csv(f'{outputDir}/{rule_name}_internal_metrics.csv')
    
    eval_df.to_csv(f'{outputDir}/{filePat}_data_predicted.csv')
    metrics_df.to_csv(f'{outputDir}/{filePat}_prf.csv')
    print(f'{filePat}_data_predicted.csv, {filePat}_prf.csv and internal_metrics.csvs created in {outputDir}')







    
    