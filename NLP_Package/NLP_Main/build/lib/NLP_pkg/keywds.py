
import spacy
import re
from lib.negation import Negex
from joblib import Parallel, delayed
from lib.lemmatization import lemmitize_sentence
import functools
import numpy as np

def load_negex_model(model = "en_core_sci_lg", language = 'en_clinical', chunks = ["no"], replace = False, pseudo_negations=list(),
        preceding_negations=list(),
        following_negations=list(),
        termination=list()):
    """loads nlp model and adds negation components
    model -> str specifying spaCy model
    language -> str ; supports ['en', 'en_clinical', 'en_clinical_sensitive']
    chunks -> ["yes"] or ["no]; use noun chunks in negation?
    replace -> bin ; Default False; replace termlist with those provided if True else extend termsets
    pseudo_negations: list
        list of phrases that extend or replaces  a negation, if empty, defaults are used
    preceding_negations: list
        negations that appear before an entity, if empty, defaults are used
    following_negations: list
        negations that appear after an entity, if empty, defaults are used
    termination: list
        phrases that "terminate" a sentence for processing purposes such as "but". If empty, defaults are used
    """
    nlp = spacy.load(model)      
    nlp.add_pipe("negex", config={
    "language":language,
        "ent_type":list(),
        "replace" : replace,
        "pseudo_negations" :pseudo_negations,
        "preceding_negations" :preceding_negations,
        "following_negations" :following_negations,
        "termination" :termination,
        "chunk_prefix" : chunks}, last=True)
    
    return nlp


def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

def flatten(list_of_lists):
    "Flatten a list of lists to a combined list"
    return [item for sublist in list_of_lists for item in sublist]


def process_text(string, nlp, sent_filter = None):
    doc = nlp(string.lower())
    if sent_filter:
        sentences = []
        for sent in doc.sents:
            if re.search(sent_filter, sent.text):
                sentences.append(nlp(sent.text))
    return doc, sentences

def process_chunk(strings, nlp, batch_size =50):
    preproc_pipe = []
    lowered = [string.lower() for string in strings]
    for doc in nlp.pipe(lowered, batch_size = batch_size):
        preproc_pipe.append(doc)
    return preproc_pipe
        
def split_sents(doc, key, nlp, sent_filter):
    sentences = []
    keys = []
    for sent in doc.sents:
        if re.search(sent_filter, sent.text):
            sentences.append(sent.text)
            keys.append(key)
    return sentences, keys


def preprocess_parallel(texts, df, nlp, batch_size = 50, chunksize=100):
    executor = Parallel(n_jobs=7, backend='multiprocessing', prefer="processes")
    process_chunk_part = functools.partial(process_chunk, nlp = nlp, batch_size = batch_size)
    do = delayed(process_chunk_part)
    tasks = (do(chunk) for chunk in chunker(texts, len(df), chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)

class KeywordProcess():
    "this runs keyword processing tasks in combination with negspacy"
    def __init__(self, doc, sent_filter=None, sentences = []):
      
        """
        nlp doc object for specified string  with specified language (str), dictionary(str) and noun chunk pref (lst ['yes'] or ['no'])
        defaults are: LANG = 'en_clinical', CHUNK = ["no"]
        sent_filter = default None or regex pattern for list of sentences to keep
        """
      
        self.sent_filter = sent_filter
        self.doc = doc
        if self.sent_filter:
            if isinstance(sentences, list):
                self.sentences = sentences
            else:
                self.sentences = []
                    
    def retokenize_sentences(self,pat1,pat2):
            sentences = self.sentences + [self.doc]
            for sent in sentences:
                self.retokenize_doc(sent, pat1, pat2)
                
    
    def retokenize_doc(self, doc, pat1, pat2):
        inits = [token.i for token in doc if token.text in [pat1]]

        ends  = [doc[i+1].text if i < len(doc) - 2 else 'ENDSTR' for i in inits]
        for i,t in zip(inits,ends):
            if t == pat2:
                with doc.retokenize() as retokenizer:
                    retokenizer.merge(doc[i:i+2])
                    
                    
    def get_matches(self, lookup, sents = False):
        """
        in > string -> containing text for parsing,  lookup -> list of strings, can be regex pattern
        out > dict of matches
        """
        if sents:
            sentences = self.sentences
        else:
            sentences = [self.doc]
        
        all_matches = []
        for sentence in sentences:
            out = 0
            #make dict of matches
            matches = {}
            for i in range(len(sentence)):
                token = sentence[i].text
                if token in lookup:
                    if token in matches.keys():
                        matches[token].append(i)
                    else:
                        matches[token]= [i]
                all_matches.append(matches)
        return all_matches
    
    
    def is_negated(self, doc, string, index, window, ents=True):
        """
        in: string, index > int, index of string of interest,
        window > int, creates window for entity search
        out: True if negated ; False if not negated
        """
        s= re.compile(r'^{}|\b{}'.format(string, string))
        end = index + window if index + window <= len(doc)-1 else len(doc)
        start = index - window if index - window > 0 else 0
        if ents:
            #print([x.text for x in doc[start:end].ents])
            neg = [x._.negex for x in doc[start:end].ents if s.search(x.text)]
            if not neg:
                neg = [x._.negex for x in doc.ents if s.search(x.text)]
        #try narrow negation range, if no success, try full doc search
        else:
            #print([x.text for x in doc[start:end]])
            neg = [x._.negex for x in doc[start:end] if s.search(x.text)]
            if not neg:
                    neg = [x._.negex for x in doc if s.search(x.text)]
        if neg:
            return neg[0]
        else:
            ph = 'ent list' if ents == True else 'token list'
            return f'{string} Not Found in {ph}'

    def keywd_find(self, lookup, sents =False, ents=True, verbose=False):
        """
        in > string -> containing text for parsing,  lookup -> list of strings, can be regex pattern
        out > 1 if keywd appears at least once and at least one of those keywords is not negated in entities 
        or noun chunks else 0.
        noun_chunks and ents control the spacy objects used for search (either noun_chunks or entities)
        """
        outs = []
        if sents:
            sentences = self.sentences
        else:
            sentences = [self.doc]
        
        answer = 0   
        for sentence in sentences:
            out = 0
            #make dict of matches
            matches = {}
            for i in range(len(sentence)):
                token = sentence[i].text
                if token in lookup:
                    if token in matches.keys():
                        matches[token].append(i)
                    else:
                        matches[token]= [i]
            #print(matches)
            for token in matches:
                #just pull entities immeadiatley around match index
                for k in matches[token]:
                    neg = self.is_negated(sentence,token,k,2, ents)
               
                    #print(neg)
                    if neg == False:
                        answer+=1  
        if verbose:
            print(matches)
            print(neg)
        return answer
               

    def common_member(self, a, b, outstr): 
        a_set = set(a) 
        b_set = set(b) 
        if (a_set & b_set): 
            return outstr
   
        
    def keywd_associations(self, baseList, assocList, sents = False, ents=True):
        """
        in > string -> for nlp processing,
         baseList -> list of base elements, can be regex
         assocList -> list of elements of interest
        out > count of number associations in children or ancestor IF negation is false for words in baseList
        Answers: Are elements in assocList associated with non-negated elements in baselist?
        """
        outs = []
        if sents:
            sentences = self.sentences
        else:
            sentences = [self.doc]
              
        for sentence in sentences:
            out = 0
        #make dict of matches
            matches = {}
            for i in range(len(sentence)):
                token = sentence[i].text
                if token in baseList:
                    if token in matches.keys():
                        matches[token].append(i)
                    else:
                        matches[token]= [i]
            #print(matches)
            for token in matches:
                #just pull entities immeadiatley around match index
                for k in matches[token]:
                    neg = self.is_negated(sentence,token,k,2, ents)
                    #print(neg)
                    if neg == False:
                #pulls out syntactic subtree for a given word, sometimes more expansive than children
                        tree = self.common_member([sentence[j].text
                                     for j in range(sentence[k].left_edge.i, sentence[k].right_edge.i+1)],
                                      assocList,outstr=1) 
                        #print([sentence[j].text for j in range(sentence[k].left_edge.i, sentence[k].right_edge.i+1)])

                        ancestor = self.common_member([t.text for t in sentence[k].ancestors],assocList,outstr=1)
                        #print([t.text for t in sentence[k].ancestors])
                        children = self.common_member([t.text for t in sentence[k].children],assocList,outstr=1)
                        #print([t.text for t in sentence[k].children])
                        if tree or ancestor or children:
                            out += 1
                outs.append(out)
        return outs
    
    
        
    def find_token(self, doc, i, wordlist):
        """
        helper fun
        return token.text, token.i if token.text in list of match patterns
        """
        token = None
        if doc[i].text in wordlist:
            t = doc[i].text
            token = t
        return token,i
        
    def common_ancestor(self, wordlist1, wordlist2, ents=True):
        """
        looser association search
        in > lst of strings to check for common ancestry
        out > 1 if there is at least 1 common ancestor
        NB: doubles of same word in given doc will return all matches
        """
        
        out = 0
        ancestors1 = []
        ancestors2 = []
        sentence = self.doc
        
        for i in range(len(sentence)):
            token1,i = self.find_token(sentence,i,wordlist1)
          
            if token1:
                neg = self.is_negated(sentence, token1,i,3, ents)
                if neg == False:
                    if sentence[i].dep_ == 'ROOT':
                        out +=1
                    else:
                        ancestors1.extend([t.text for t in sentence[i].ancestors])
                      
        
        for i in range(len(sentence)):   
            token2,j = self.find_token(sentence,i,wordlist2)
            if token2:
                neg = self.is_negated(sentence, token2,j,3, ents)
                if neg == False:
                    if sentence[j].dep_ == 'ROOT':
                        out +=1
                    else:
                        ancestors2.extend([t.text for t in sentence[j].ancestors])

        if (set(ancestors1) & set(ancestors2)):
    
            out += 1
        return out

    def get_subtree(self, lookup):
        """
        in > string -> containing text for parsing,  lookup -> list of strings
        out > syntactic subtree as string, useful for pulling out relevant phrase
        """
        trees = []
        for i in range(len(self.doc)):
            if self.doc[i].text in lookup:
                #pulls out syntactic subtree for a given word, sometimes more expansive than children
                tree = " ".join([self.doc[j].text for j in range(self.doc[i].left_edge.i,doc[i].right_edge.i+1)])
                trees.append(tree)
                return trees
                

    def populate_field(self, baseList, assocList, popval):
        assoc = self.keywd_associations(baseList, assocList)
        if assoc >0:
            return popval



