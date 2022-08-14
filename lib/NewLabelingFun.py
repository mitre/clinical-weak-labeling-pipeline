from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
import re
import numpy as np

from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.labeling.model import MajorityLabelVoter

from sklearn import metrics


ABSTAIN = -1
FOUND = 1
NOTFOUND = 0


def regex_finder(x, series, regex_pat, invert, lab=FOUND):
    """
    simple regex label
    in: Pandas series of text objects
    out > label > default is FOUND, must specify NOTFOUND
    """
    if invert==False:
        return lab if re.search(regex_pat, x[series].lower(), flags=re.I) else ABSTAIN
    else:
        return lab if re.search(regex_pat, x[series].lower(), flags=re.I)==False else ABSTAIN


def binary_find(x, series, invert = False,  labs = [FOUND, NOTFOUND]):
    """
    in: Pandas series of binary objects binary > [0,1]
    out: label from  []
    """
    if invert == False:
        return labs[0] if  x[series] == 1  else ABSTAIN
    else:
        return labs[1] if  x[series] == 0  else ABSTAIN

    
def keywd_find(x,series, keywords, ents, invert,  labs = [FOUND, NOTFOUND]):
    if invert == False:
        return labs[0] if x[series].keywd_find(lookup = keywords, sents = True, ents=ents) > 0 else ABSTAIN
    else:
        return labs[1] if x[series].keywd_find(lookup = keywords, sents = True, ents=ents) == 0 else ABSTAIN
    
    
def keywd_find_two(x, series, wordlist1, wordlist2, ents, invert, labs = [FOUND, NOTFOUND]):
    """
    Rule: Are two keywords present 
    in: Pandas series of KeywordProcess objects
    out: label from []
    """
    find1 = x[series].keywd_find(wordlist1, sents = True, ents=ents)
    find2 = x[series].keywd_find(wordlist2, sents = True, ents=ents)
    if invert == False:
        return labs[0] if find1 > 0 and find2 > 0 else ABSTAIN
    else:
        return labs[1] if find1 == 0 or find2 == 0 else ABSTAIN

def keywd_two_conditional(x, series, wordlist1, wordlist2, ents, invert,  labs = [FOUND, NOTFOUND]):
    """
    Rule: Is value from list 1 available? If not, is value from list 2 available
    This rule checks first for non-negated values from 1 and if none are available checks for non-negated values from list 2
    in: Pandas series of KeywordProcess objects
    out: label from []
    """
    if invert == False:
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
    
    
def keywd_associations(x, series, wordlist1, wordlist2, ents, invert, labs = [FOUND, NOTFOUND]):
    """
    Rule 2 extraction of bullet 2
   in: Pandas series of KeywordProcess objects
    out: label from [ out: label from  []
    """
    out = sum(x[series].keywd_associations(wordlist1,wordlist2, sents = True, ents=ents)) 
    
    if invert == False:
        return labs[0] if out >= 1 else ABSTAIN
    else:
        return labs[1] if out == 0 else ABSTAIN



def common_ancestor(x, series, wordlist1, wordlist2, ents, invert,  labs = [FOUND, NOTFOUND]):
    """
    Rule 2 extraction of bullet 2
    in: Pandas series of KeywordProcess objects
     out: label from  []
    """   
    if invert == False:
        return labs[0] if x[series].common_ancestor(wordlist1, wordlist2, ents=ents) == 1 else ABSTAIN
    else:
        return labs[1] if x[series].common_ancestor(wordlist1, wordlist2, ents=ents) == 0 else ABSTAIN



LABELFUNCTIONS = {'binary_find':binary_find, 
             'regex_finder' :regex_finder, 
             'keywd_associations': keywd_associations, 
             'common_ancestor':common_ancestor,
             'keywd_find_two':keywd_find_two,
             'keywd_find': keywd_find, 
             'keywd_two_conditional': keywd_two_conditional
                 }



def make_lf(label, fun, argDict, lab_suffix=''):
    """
    wrap python functions defined in labeling_functions.py for reusability
    label > str
    invert > bin
    fun > key from function dictionary pointing to argument for use
    argDict > dict of args from base function as dict(arg1=... , arg2=...)
    
    """
    return LabelingFunction(
        name=f"{label}_{fun}_{lab_suffix}",
        f=LABELFUNCTIONS[fun],
        resources=argDict,
    )


### run labels through labeling pipeline
def snorkel_pipeline(inputs, df, out_col, label = 'COLUMNOFINTEREST'):
    """
    in: inputs dictionary composed of labeling function arguments, out_col > str of column name for predicted outputs, label > str 
    out: coverage_overlap df and predicted labels array
    """
    lfs = []
    for key in inputs:
        labeling_function = make_lf(label, fun = inputs[key]['fun'], argDict= inputs[key]['Dict'], lab_suffix = inputs[key]['suffix'])
        lfs.append(labeling_function)

    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df)
    coverage_overlap = LFAnalysis(L=L_train, lfs=lfs).lf_summary()

    #going to use majority label voter
    majority_model = MajorityLabelVoter()
    preds_train = majority_model.predict(L=L_train)
    df[out_col] = preds_train
    print(f"number of non identical codes out of labeled set for {label}")
    print(df[df[out_col] == -1].shape)
    return coverage_overlap, preds_train, df

def precision_recall(df, out_col, true_col, preds_train, avg = 'binary', labels = [0, 1]):
    """
    in: df, , true_col > str of column name containing true values as binary (will accept 0,1,-1),preds_train > array with     predicted values
    out: dict of precision, recall, f1, df with predicted column 
    """
   
    y_true = df[true_col].to_numpy()
    y_pred = df[out_col].to_numpy()
    if avg == 'binary':
        y_pred = np.where(y_pred==-1, 0, y_pred)
    precision = metrics.recall_score(y_true, y_pred,labels = labels, average = avg)
    recall=metrics.precision_score(y_true, y_pred,labels = labels, average = avg)
    f1=metrics.f1_score(y_true, y_pred,labels = labels, average = avg)
    labels = df[true_col].unique()
    print(f'{true_col}: precision: {precision} , recall: {recall}, f1: {f1}')
    return {'labels': labels, 'label':true_col, 'precision': precision, 'recall' :recall, 'f1':f1}