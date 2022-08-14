import pandas as pd
from lib.keywds import *


from spacy.pipeline import EntityRuler
#from spacy.lemmatizer import Lemmatizer
from spacy.tokens import Token, Doc, Span
from spacy.matcher import PhraseMatcher
from spacy.matcher import Matcher
from spacy.tokens import DocBin

import json
import pickle


def make_analytic_df(df, eval_columns, nlp_objects, nlp_Keys, filt, retokenization_list =[], StudyKey='Study_No'):
    sentences_dfs = []
    for keys, sentences in zip([nlp_Keys['text_sentence_keys'], nlp_Keys['lemma_sentence_keys']],[nlp_objects[2], nlp_objects[3]]):
        temp = pd.DataFrame({StudyKey: keys, 'sentence_list': sentences})                  
        temp = temp.groupby([StudyKey]).agg(lambda x: tuple(x)).applymap(list).reset_index()
        sentences_dfs.append(temp)
    
    doc_df = pd.DataFrame({StudyKey:nlp_Keys['StudyId'], 'docs':nlp_objects[0]}).merge(sentences_dfs[0], how = 'left')
    lemma_df = pd.DataFrame({StudyKey:nlp_Keys['StudyId'], 'docs':nlp_objects[1]}).merge(sentences_dfs[1], how = 'left')
    alldocs = doc_df.merge(lemma_df, on= StudyKey, how= 'left', suffixes = ("_text", "_lemma"))
    
    alldocs['nlp_string'] = ''
    alldocs['lemma'] = ''
    for i, row in alldocs.iterrows():
        kwp =KeywordProcess(row['docs_text'], filt, row['sentence_list_text'])
        kwp_l =KeywordProcess(row['docs_lemma'], filt, row['sentence_list_lemma'])
        if retokenization_list:
            for tup in retokenization_list:
                kwp.retokenize_sentences(tup[0], tup[1])
                kwp_l.retokenize_sentences(tup[0], tup[1])
        alldocs.at[i,'nlp_string'] = kwp
        alldocs.at[i,'lemma'] = kwp_l
    
    #print('-------make_analytic_df: debug line')
    #print(alldocs.columns)
    #print(eval_columns)
    #print(df.columns)
    #print('-------make_analytic_df: debug line')
    return df[eval_columns].merge(alldocs, how= 'left')


def stream_nlp_process(texts, nlp, studyid, filt, doc_batch, sent_batch):
    #breaking down to lists to allow stream processing speeds process considerably when running full data set
    text_sentences = []
    text_keys = []
    text_docs = process_chunk(texts,nlp, batch_size = doc_batch)
    for index, doc in zip(studyid, text_docs):
        sents, i = split_sents(doc, index, nlp, filt)
        text_keys.extend(i)
        text_sentences.extend(sents)
    
    sentence_docs = process_chunk(text_sentences, nlp, batch_size = sent_batch)
 
    #creates lemma nlp objects   
    lemma_sentences = []
    lemma_keys = []
    lemma_pre = [lemmitize_sentence(doc) for doc in text_docs]
    lemma_docs = process_chunk(lemma_pre,nlp, batch_size = doc_batch)
    for index, doc in zip(studyid, lemma_docs):
        sents, i = split_sents(doc, index, nlp, filt)
        lemma_keys.extend(i)
        lemma_sentences.extend(sents)
    
    lemma_sent_docs = process_chunk(lemma_sentences, nlp, batch_size = sent_batch)
    return [text_docs,lemma_docs,sentence_docs, lemma_sent_docs], {'StudyId': studyid, 'text_sentence_keys': text_keys,'lemma_sentence_keys': lemma_keys}


def pickle_nlp_objects(outputDir,lab,docs, nlp):
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
    for doc in docs:
        doc_bin.add(doc)
    print(len(docs))
    bytes_data = doc_bin.to_bytes()
    with open(f"{outputDir}/{lab}_nlp_objects.pickle","wb") as handle:
         pickle.dump(bytes_data  ,handle)
    return bytes_data

def nlp_keys_to_json(outputDir, keysDict, lab):
    with open(f'{outputDir}/NLPpipeKeys_{lab}.json', 'w') as out:
        json.dump(keysDict, out)
    