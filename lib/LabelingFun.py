from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
import re



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


def binary_find(x, series, invert = False):
    """
    in: Pandas series of binary objects binary > [0,1]
    out: label from  []
    """
    if invert == False:
        return FOUND if  x[series] == 1  else ABSTAIN
    else:
        return NOTFOUND if  x[series] == 0  else ABSTAIN

def keywd_find(x,series, kewords, ents, invert):
    if invert == False:
        return FOUND if x[series].keywd_find(lookup = keywords, sents = True, ents=ents) > 0 else ABSTAIN
    else:
        return NOTFOUND if x[series].keywd_find(lookup = keywords, sents = True, ents=ents) == 0 else ABSTAIN
    
def keywd_find_two(x, series, wordlist1, wordlist2, ents, invert):
    """
    Rule: Are two keywords present 
    in: Pandas series of KeywordProcess objects
    out: label from []
    """
    find1 = x['nlp_string'].keywd_find(wordlist1, sents = True, ents=ents)
    find2 = x['nlp_string'].keywd_find(wordlist2, sents = True, ents=ents)
    if invert == False:
        return FOUND if find1 > 0 and find2 > 0 else ABSTAIN
    else:
        return NOTFOUND if find1 == 0 or find2 == 0 else ABSTAIN

def keywd_two_conditional(x, series, wordlist1, wordlist2, ents, invert):
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
            final += x[series].keywd_find(lookup = keywords, sents = True, ents = False)     
        else:
            final +=  x.nlp_string.keywd_find(lookup = secondary_keywords, sents = True, ents = False)
        return FOUND if final > 0 else ABSTAIN
    else:
        final = 0    
        if  len(x.nlp_string.get_matches(lookup = keywords, sents = True)) > 0:
                #check and see if there are matches in this dict with primary list. if not check secondary list
            final += x.nlp_string.keywd_find(lookup = keywords, sents = True, ents = False)        
        else:
            final +=  x.nlp_string.keywd_find(lookup = secondary_keywords, sents = True, ents = False)
        return NOTFOUND if final == 0 else ABSTAIN
    
    
def keywd_associations(x, series, wordlist1, wordlist2, ents, invert):
    """
    Rule 2 extraction of bullet 2
   in: Pandas series of KeywordProcess objects
    out: label from [ out: label from  []
    """
    out = sum(x[series].keywd_associations(wordlist1,wordlist2, sents = True, ents=ents)) 
    
    if invert == False:
        return FOUND if out >= 1 else ABSTAIN
    else:
        return NOTFOUND if out == 0 else ABSTAIN



def common_ancestor(x, series, wordlist1, wordlist2, ents, invert):
    """
    Rule 2 extraction of bullet 2
    in: Pandas series of KeywordProcess objects
     out: label from  []
    """   
    if invert == False:
        return FOUND if x[series].common_ancestor(wordlist1, wordlist2, ents=ents) == 1 else ABSTAIN
    else:
        return NOTFOUND if x[series].common_ancestor(wordlist1, wordlist2, ents=ents) == 0 else ABSTAIN



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


