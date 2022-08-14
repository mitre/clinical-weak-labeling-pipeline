
import spacy
#from spacy.pipeline import EntityRuler
#from spacy.lemmatizer import Lemmatizer
import re

def remove_extra_spaces(text):
    text2 = re.sub(' +', ' ', text)
    return text2

def lemmitize_sentence(doc):
    tokens = [] 
    for token in doc: 
        tokens.append(token) 
    lemmatized_sentence = " ".join([token.lemma_ for token in doc]); #if tok.is_alpha and tok.text.lower() not in stopwords
    return lemmatized_sentence

def clean_text(text, char_to_replace):
    
    for key, value in char_to_replace.items():#get rid of % signs
    # Replace key character with value character in string
        text = text.replace(key, value)

    text = remove_extra_spaces(text)
    return text.lower()