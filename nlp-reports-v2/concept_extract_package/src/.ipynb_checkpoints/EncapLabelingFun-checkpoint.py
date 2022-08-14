from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
import re
import numbers
import numpy as np

from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel
from snorkel.labeling.model import MajorityLabelVoter
from sklearn import metrics

ABSTAIN = -1
FOUND = 1
NOTFOUND = 0

def regex_finder(x, series, regex_pat, complement, lab=FOUND):
    """
    Function: 
    ----------
        regex_finder

    Description:
    ----------
        simple regex label

    Inputs: 
    ----------
        x: Pandas series of text objects
        series: column with text to be tested
        regex_pat: regex pattern for filtering
        complement: if true will look for false outcome from regex
        lab: default value to return

    Outputs:
    ----------
        label - default is FOUND, must specify NOTFOUND
        
    """
    
    if complement==False:
        return lab if re.search(regex_pat, str(x[series]).lower(), flags=re.I) else ABSTAIN
    else:
        return lab if re.search(regex_pat, str(x[series]).lower(), flags=re.I)==False else ABSTAIN


def binary_find(x, series, complement = False,  labs = [FOUND, NOTFOUND]):
    """
    Function: 
    ----------
        binary_find

    Description:
    ----------
        simple function that expects inputs to be a binary value

    Inputs:
    ----------
        x: Pandas series of text objects
        series: column with with binary 0/1 values
        complement: if true will look for 0
        labs: default value to return

    Outputs: 
    ----------
        label from labs
        
    """
    if complement == False:
        return labs[0] if  x[series] == 1  else ABSTAIN
    else:
        return labs[1] if  x[series] == 0  else ABSTAIN

    
def keywd_find(x, series, wordlist1, ents, complement,  labs = [FOUND, NOTFOUND]):
    """
    Function: 
    ----------
        keywd_find

    Description:
    ----------
        Is a keyword present without negation

    Inputs:
    ----------
        x: Pandas series of text objects
        series: column with keyword process objects
        keywords: list of variants of keyword to find
        ents: if true will use entities instead of tokens
        complement: if true returns if not found
        labs: found/notfound values to return

    Outputs: 
    ----------
        label from labs
        
    """
    if complement == False:
        return labs[0] if x[series].keywd_find(lookup = wordlist1, sents = True, ents=ents) > 0 else ABSTAIN
    else:
        return labs[1] if x[series].keywd_find(lookup = wordlist1, sents = True, ents=ents) == 0 else ABSTAIN
    
    
def keywd_find_two(x, series, wordlist1, wordlist2, ents, complement, labs = [FOUND, NOTFOUND]):
    """
    Function: 
    ----------
        keywd_find_two

    Description:
    ----------
        Rule - Are two keywords present (both)

    Inputs: 
    ----------
        x: Pandas series of text objects
        series: column with keyword process objects
        wordlist1: list of variants of first keyword to find
        wordlist2: list of variants of second keyword to find
        ents: if true will use entities instead of tokens
        complement: if true returns if not found
        labs: found/notfound values to return

    Outputs:
    ----------
        label from labs
        
    """
    find1 = x[series].keywd_find(wordlist1, sents = True, ents=ents)
    find2 = x[series].keywd_find(wordlist2, sents = True, ents=ents)
    if complement == False:
        return labs[0] if find1 > 0 and find2 > 0 else ABSTAIN
    else:
        return labs[1] if find1 == 0 or find2 == 0 else ABSTAIN

def keywd_two_conditional(x, series, wordlist1, wordlist2, ents, complement,  labs = [FOUND, NOTFOUND]):
    """
    Function: 
    ----------
        keywd_two_conditional

    Description:
    ----------
        Rule - Is value from list 1 available? If not, is value from list 2 available?
    This rule checks first for non-negated values from 1 and if none are available checks for non-negated values from list 2

    Inputs: 
    ----------
        x: Pandas series of text objects
        series: column with keyword process objects
        wordlist1: list of variants of first keyword to find
        wordlist2: list of variants of second keyword to find
        ents: if true will use entities instead of tokens
        complement: if true returns if not found
        labs: found/notfound values to return

    Outputs: 
    ----------
        label from labs
        
    """
    if complement == False:
        final = 0
        if  len(x[series].get_matches(lookup = wordlist1, sents = True)) > 0:
                #check and see if there are matches in this dict with primary list. if not check secondary list
            final += x[series].keywd_find(lookup = wordlist1, sents = True, ents = ents)     
        else:
            final +=  x.nlp_string.keywd_find(lookup = wordlist2, sents = True, ents = ents)
        return labs[0] if final > 0 else ABSTAIN
    else:
        final = 0    
        if  len(x.nlp_string.get_matches(lookup = wordlist1, sents = True)) > 0:
                #check and see if there are matches in this dict with primary list. if not check secondary list
            final += x.nlp_string.keywd_find(lookup = wordlist1, sents = True, ents = ents)        
        else:
            final +=  x.nlp_string.keywd_find(lookup = wordlist2, sents = True, ents = ents)
        return labs[1] if final == 0 else ABSTAIN
    
    
def keywd_associations(x, series, wordlist1, wordlist2, ents, complement, labs = [FOUND, NOTFOUND]):
    """
    Function: 
    ----------
        keywd_associations

    Description:
    ----------
        is a keyword from list 1 associated with a keword from list 2

    Inputs: 
    ----------
        x: Pandas series of text objects
        series: colum with keyword process objects
        wordlist1: list of variants of first keyword to find
        wordlist2: list of variants of second keyword to find and associate with first
        ents: if true will use entities instead of tokens
        complement: if true returns if not found
        labs: found/notfound values to return

    Outputs: 
    ----------
        label from labs
        
    """
    out = sum(x[series].keywd_associations(wordlist1,wordlist2, sents = True, ents=ents)) 
    #out = sum(x[series].keywd_associations(wordlist1,wordlist2, sents = True, ents=ents)) 
    if complement == False:
        return labs[0] if out >= 1 else ABSTAIN
    else:
        return labs[1] if out == 0 else ABSTAIN



def common_ancestor(x, series, wordlist1, wordlist2, ents, complement,  labs = [FOUND, NOTFOUND]):
    """
    Function: 
    ----------
        common_ancestor

    Description:
    ----------
        do keyowrds from list 1 and list 2 share a common ancestor

    Inputs:
    ----------
        x: Pandas series of text objects
        series: colum with keyword process objects
        wordlist1: list of variants of first keyword to find
        wordlist2: list of variants of second keyword to find and associate with first
        ents: if true will use entities instead of tokens
        complement: if true returns if not found
        labs: found/notfound values to return

    Outputs: 
    ----------
        label from labs
        
    """   
    if complement == False:
        return labs[0] if x[series].common_ancestor(wordlist1, wordlist2, ents=ents) == 1 else ABSTAIN
    else:
        return labs[1] if x[series].common_ancestor(wordlist1, wordlist2, ents=ents) == 0 else ABSTAIN



LABELFUNCTIONS = {
             'binary_find': binary_find, 
             'regex_finder' : regex_finder, 
             'keywd_associations': keywd_associations, 
             'common_ancestor': common_ancestor,
             'keywd_find_two': keywd_find_two,
             'keywd_find': keywd_find, 
             'keywd_two_conditional': keywd_two_conditional
                 }



def make_lf(label, fun, argDict, lab_suffix=''):
    """
    Function: 
    ----------
        make_lf

    Description:
    ----------
        the main guts of the NLP pipeline
        creates a labeling function for searching for keyords
        or associating words with other words

    Inputs:
    ----------
        label: userdefined label for the function
        fun: name of function to be constructed, must be in the list
            defined in LABELFUNCTIONS struct
        argDict: dict of args from base function as dict(arg1=... , arg2=...)
        lab_suffix: userdefined tag at end of name for function

    Outputs:
    ----------
        a labelling function (eg find a keyword) for the pipeline
        
    """
    return LabelingFunction(
        name=f"{label}_{fun}_{lab_suffix}",
        f=LABELFUNCTIONS[fun],
        resources=argDict,
    )


### run labels through labeling pipeline
def snorkel_pipeline(inputs, df, out_col, label = 'COLUMNOFINTEREST'):
    """
    Function: 
    ----------
        snorkel_pipeline

    Description:
    ----------
        created the labeling functions and processes all documents in the pandas
        with the NLP, then returns the results

    Inputs: 
    ----------
        inputs: inputs dictionary composed of labeling function arguments
        df: pandas dataframe from make_analytic_df function
        out_col: str of column name for predicted outputs
        label: name of column with known (true) values as binary (will accept 0,1,-1)

    Outputs: 
    ----------
        coverage_overlap, df and predicted labels array
        
    """
    lfs = []
    
    #loops through all desired labeling functions and makes them
    for key in inputs:
        labeling_function = make_lf(label, fun = inputs[key]['fun'], argDict= inputs[key]['Dict'], lab_suffix = inputs[key]['suffix'])
        lfs.append(labeling_function)

    #apply labeling functions with NLP and process all documents with NLP
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df)

    #np.savetxt(outfile, L_train, delimiter=',') 

    coverage_overlap = LFAnalysis(L=L_train, lfs=lfs).lf_summary()

    #going to use majority label voter for final NLP decision
    majority_model = MajorityLabelVoter()
    preds_train = majority_model.predict(L=L_train)
    df[out_col] = preds_train
    print(f"number of non identical codes out of labeled set for {label}")
    print(df[df[out_col] == -1].shape)
    return coverage_overlap, preds_train, df

def precision_recall(df, out_col, true_col, preds_train, avg = 'binary', labels = [0, 1]):
    """
    Function: 
    ----------
        precision_recall

    Description:
    ----------
        calculate the precision and recall from an NLP process using known (desired) outcome
        in true_col

    Inputs:
    ----------
        df: pandas dataframe from make_analytic_df function
        out_col: str of column name for predicted outputs
        true_col: name of column with known (true) values as binary (will accept 0,1,-1)
        preds_train: array with predicted values
        avg: averaging method
        labels: output labels

    Outputs: 
    ----------
        dict of precision, recall, f1, df with predicted column 
    """
   
    y_true = df[true_col].to_numpy()
    y_pred = df[out_col].to_numpy()
    if avg == 'binary':
        y_pred = np.where(y_pred==-1, 0, y_pred)
    precision = metrics.recall_score(y_true, y_pred, labels = labels, average = avg)
    recall = metrics.precision_score(y_true, y_pred, labels = labels, average = avg)
    f1 = metrics.f1_score(y_true, y_pred, labels = labels, average = avg)
    
    print(f'{true_col}: precision: {precision} , recall: {recall}, f1: {f1}')

    labels = df[true_col].unique()
    return {'labels': labels, 'label':true_col, 'precision': precision, 'recall' :recall, 'f1':f1}