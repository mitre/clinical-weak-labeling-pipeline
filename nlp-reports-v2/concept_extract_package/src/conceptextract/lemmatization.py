
import spacy
#from spacy.pipeline import EntityRuler
#from spacy.lemmatizer import Lemmatizer
import re

def remove_extra_spaces(text):
    """
    Function: 
    ----------
        remove_extra_spaces

    Description:
    ----------
        gets rid of extra spaces in text

    Inputs: 
    ----------
        text: a text string

    Outputs:
    ----------
        text with extra spaces removed
        
    """
    text2 = re.sub(' +', ' ', text)
    return text2

def lemmitize_sentence(doc):
    """
    Function: 
    ----------
        lemmitize_sentence

    Description:
    ----------
        redces the complexity of a sentence, removes inflection and tense from words
        reducing all words to their base form called a lemma

    Inputs:
    ----------    
        doc: from nlp.pipe

    Outputs:
    ----------
        doc with lemmitized sentences
        
    """
    tokens = [] 
    for token in doc: 
        tokens.append(token) 
    lemmatized_sentence = " ".join([token.lemma_ for token in doc]); #if tok.is_alpha and tok.text.lower() not in stopwords
    return lemmatized_sentence

def clean_text(text, char_to_replace):
    """
    Function: 
    ----------
        clean_text

    Description:
    ----------
        removes extra spaces and gets rid of unwanted characters the user specifies

    Inputs: 
    ----------
        text: text string
        char_to_replace: character to be replaced and the replacement character eg to get rid of %: char_to_replace = ['%', '']
            
    Outputs:
    ----------
        text with characters replaced if they are present, and extra spaces are removed
        
    """
    for key, value in char_to_replace.items():#get rid of % signs
    # Replace key character with value character in string
        text = text.replace(key, value)

    text = remove_extra_spaces(text)
    return text.lower()