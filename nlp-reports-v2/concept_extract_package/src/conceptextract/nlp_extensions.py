import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy import displacy

#nltk.download()

num_pattern = [{"TEXT" : {"REGEX": "\d+(\.\d+)?"}}]

@Language.component("custom_sentencizer")
def custom_sentencizer(doc):
    '''
        function:
            custom_sentencizer

        description:
            this function create custom boundary conditions for identifying the
            boundaries of a sentence due to the fact that the default sentenizer
            with spacy is sub par.  You will need to disable the ner (named entity 
            recognizer) in spacy.load to make this work.

            Add to pipeline with:
                nlp.add_pipe("custom_sentencizer", before="parser")

        inputs:
            doc : an spacy nlp document

        outputs:
            doc : split by sentences according to the user-defined chriteria
    '''

    for i, token in enumerate(doc[:-2]):  # The last token cannot start a sentence
        prev_test = "GGG"
        if(i > 0):
            prev_token = doc[i-1]
            prev_test = prev_token.text

        curr_test = token.text
        if (curr_test == ".") & (prev_test.isdigit() == False):
            doc[i+1].is_sent_start = True
        else:
            doc[i+1].is_sent_start = False  # Tell the default sentencizer to ignore this token

    return doc