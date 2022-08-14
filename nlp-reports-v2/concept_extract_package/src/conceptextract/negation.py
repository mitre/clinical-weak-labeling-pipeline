from .termsets import LANGUAGES
from spacy.tokens import Token, Doc, Span
from spacy.tokenizer import Tokenizer
from spacy.matcher import PhraseMatcher
import logging
from spacy.language import Language
from typing import List


LIST = List[str]

@Language.factory("negex", default_config={
    "language":"en_clinical",
        "ent_type":list(),
        "replace" : False,
        "pseudo_negations" :list(),
        "preceding_negations" :list(),
        "following_negations" :list(),
        "termination" :list(),
        "chunk_prefix" :list()})

def negex(nlp, name, language: str, ent_type:LIST,
        replace :bool,
        pseudo_negations: LIST,
        preceding_negations:LIST,
        following_negations: LIST,
        termination: LIST,
        chunk_prefix: LIST):
    """
    Function: 
    ----------
        negex

    Description:
    ----------
        creates a negex object

    Inputs: 
    ----------
        nlp: nlp generated with load_negex_model
        name: unused
        language: language code, if using default termsets (e.g. "en" for english)
        ent_type: list of entity types to negate
        replace: indicator variable for extension or replacement of termset(s) lists (replaces when True; extends when False)
        pseudo_negations: list of phrases that extend or replaces  a negation, if empty, defaults are used
        preceding_negations: negations that appear before an entity, if empty, defaults are used
        following_negations: negations that appear after an entity, if empty, defaults are used
        termination: phrases that "terminate" a sentence for processing purposes such as "but". If empty, defaults are used
        chunk_prefix: ["yes"] or ["no]; use noun chunks in negation?

    Outputs:
    ----------
        negex object
        
    """

    return Negex(nlp, name, ent_type=ent_type,
        replace = replace,
        pseudo_negations = pseudo_negations,
        preceding_negations = preceding_negations,
        following_negations = following_negations,
        termination = termination,
        chunk_prefix = chunk_prefix)



class Negex:
    """
    Class: 
    ----------
        Negex

    Description:
    ----------
        A spaCy pipeline component which identifies negated tokens in text.
	    Based on: NegEx - A Simple Algorithm for Identifying Negated Findings and Diseasesin Discharge Summaries
        Chapman, Bridewell, Hanbury, Cooper, Buchanan

    Inputs
    ----------
        nlp: nlp generated with load_negex_model
        ent_type: list of entity types to negate
        language: language code, if using default termsets (e.g. "en" for english)
        replace: indicator variable for extension or replacement of termset(s) lists (replaces when True; extends when False)
        pseudo_negations: list of phrases that extend or replaces  a negation, if empty, defaults are used
        preceding_negations: negations that appear before an entity, if empty, defaults are used
        following_negations: negations that appear after an entity, if empty, defaults are used
        termination: phrases that "terminate" a sentence for processing purposes such as "but". If empty, defaults are used
        
	"""

    def __init__(
        self,
        nlp,
        name,
        language="en_clinical",
        ent_type=list(),
        replace = False,
        pseudo_negations=list(),
        preceding_negations=list(),
        following_negations=list(),
        termination=list(),
        chunk_prefix=list()
    ):
        """
        Function: 
        ----------
            Negex

        Description:
        ----------
            negex class definition

        Inputs: 
        ----------
            nlp: nlp generated with load_negex_model
            name: unused
            language: language code, if using default termsets (e.g. "en" for english)
            ent_type: list of entity types to negate
            replace: indicator variable for extension or replacement of termset(s) lists (replaces when True; extends when False)
            pseudo_negations: list of phrases that extend or replaces  a negation, if empty, defaults are used
            preceding_negations: negations that appear before an entity, if empty, defaults are used
            following_negations: negations that appear after an entity, if empty, defaults are used
            termination: phrases that "terminate" a sentence for processing purposes such as "but". If empty, defaults are used
            chunk_prefix: ["yes"] or ["no]; use noun chunks in negation?

        Outputs:
        ----------
            nlp object
            
        """
        if not language in LANGUAGES:
            raise KeyError(
                f"{language} not found in languages termset. "
                "Ensure this is a supported language or specify "
                "your own termsets when initializing Negex."
            )
        termsets = LANGUAGES[language]
        if not Span.has_extension("negex"):
            Span.set_extension("negex", default=False, force=True)
        
        if not Token.has_extension("negex"):
            Token.set_extension("negex", default=False, force=True)

        negationDict = {'pseudo_negations':pseudo_negations, 'preceding_negations':preceding_negations, 
                            'following_negations':following_negations,'termination':termination}
        
        if not replace:
            for negType in negationDict.keys():
                if negationDict.get(negType):
                    try:
                        termsets[negType].extend(negationDict.get(negType))
                        print(f'the {negType} for this session is: {termsets.get(negType)}\n\n')
                    except:
                        raise KeyError(f"{negType} not specified for this language.")

            pseudo_negations = termsets["pseudo_negations"]
            preceding_negations = termsets["preceding_negations"]
            following_negations = termsets["following_negations"]
            termination = termsets["termination"]

        else:
            for negType in negationDict.keys():
                if not negationDict.get(negType):
                    raise KeyError(f"{line} not specified for this language.")
           

        # efficiently build spaCy matcher patterns
        tokenizer = Tokenizer(nlp.vocab)
        self.pseudo_patterns = list(tokenizer.pipe(pseudo_negations))
        self.preceding_patterns = list(tokenizer.pipe(preceding_negations))
        self.following_patterns = list(tokenizer.pipe(following_negations))
        self.termination_patterns = list(tokenizer.pipe(termination))

        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self.matcher.add("pseudo", None, *self.pseudo_patterns)
        self.matcher.add("Preceding", None, *self.preceding_patterns)
        self.matcher.add("Following", None, *self.following_patterns)
        self.matcher.add("Termination", None, *self.termination_patterns)
        self.nlp = nlp
        self.ent_type = ent_type

        self.chunk_prefix = list(tokenizer.pipe(chunk_prefix))

    def get_patterns(self):
        """
        Function: 
        ----------
            get_patterns

        Description:
        ----------
            returns phrase patterns used for various negation dictionaries

        Inputs:
        ----------
            none (processes internal class patterns)
        
        Outputs:
        ----------
            patterns: pattern_type: [patterns]
            
        """
        patterns = {
            "pseudo_patterns": self.pseudo_patterns,
            "preceding_patterns": self.preceding_patterns,
            "following_patterns": self.following_patterns,
            "termination_patterns": self.termination_patterns,
        }
        for pattern in patterns:
            logging.info(pattern)
        return patterns

    def process_negations(self, doc):
        """
        Function: 
        ----------
            process_negations

        Description:
        ----------
            Find negations in doc and clean candidate negations to remove pseudo negations

        Inputs:
        ----------
            doc: spaCy Doc object

        Outputs:
        ----------
            preceding: list of tuples for preceding negations
            following: list of tuples for following negations
            terminating: list of tuples of terminating phrases
            
        """
        ###
        # does not work properly in spacy 2.1.8. Will incorporate after 2.2.
        # Relying on user to use NER in meantime
        # see https://github.com/jenojp/negspacy/issues/7
        ###
        # if not doc.is_nered:
        #     raise ValueError(
        #         "Negations are evaluated for Named Entities found in text. "
        #         "Your SpaCy pipeline does not included Named Entity resolution. "
        #         "Please ensure it is enabled or choose a different language model that includes it."
        #     )
        preceding = list()
        following = list()
        terminating = list()

        matches = self.matcher(doc)
        pseudo = [
            (match_id, start, end)
            for match_id, start, end in matches
            if self.nlp.vocab.strings[match_id] == "pseudo"
        ]

        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "pseudo":
                continue
            pseudo_flag = False
            for p in pseudo:
                if start >= p[1] and start <= p[2]:
                    pseudo_flag = True
                    continue
            if not pseudo_flag:
                if self.nlp.vocab.strings[match_id] == "Preceding":
                    preceding.append((match_id, start, end))
                elif self.nlp.vocab.strings[match_id] == "Following":
                    following.append((match_id, start, end))
                elif self.nlp.vocab.strings[match_id] == "Termination":
                    terminating.append((match_id, start, end))
                else:
                    logging.warnings(
                        f"phrase {doc[start:end].text} not in one of the expected matcher types."
                    )
        return preceding, following, terminating

    def termination_boundaries(self, doc, terminating):
        """
        Function: 
        ----------
            termination_boundaries

        Description:
        ----------
            Create sub sentences based on terminations found in text.

        Inputs: 
        ----------
            doc: spaCy Doc object
            terminating: list of tuples with (match_id, start, end)

        Outputs:
        ----------
            boundaries: list of tuples with (start, end) of spans

        """
        sent_starts = [sent.start for sent in doc.sents]
        terminating_starts = [t[1] for t in terminating]
        starts = sent_starts + terminating_starts + [len(doc)]
        starts.sort()
        boundaries = list()
        index = 0
        for i, start in enumerate(starts):
            if not i == 0:
                boundaries.append((index, start))
            index = start
        return boundaries

    def negex(self, doc):
        """
        Function: 
        ----------
            negex

        Description: 
        ----------
        Negates entities of interest
        
        Inputs: 
        ----------
            doc: spaCy Doc object

        Outputs:
        ----------
            spaCy Doc object with negations
            
        """
        preceding, following, terminating = self.process_negations(doc)
        boundaries = self.termination_boundaries(doc, terminating)
        for b in boundaries:
            sub_preceding = [i for i in preceding if b[0] <= i[1] < b[1]]
            sub_following = [i for i in following if b[0] <= i[1] < b[1]]
           #entities
            for e in doc[b[0] : b[1]].ents:
                if self.ent_type:
                    if e.label_ not in self.ent_type:
                        continue
                if any(pre < e.start for pre in [i[1] for i in sub_preceding]):
                    e._.negex = True
                    continue
                if any(fol > e.end for fol in [i[2] for i in sub_following]):
                    e._.negex = True
                    continue
                if self.chunk_prefix:
                    if any(
                        c.text.lower() == doc[e.start].text.lower()
                        for c in self.chunk_prefix
                        ):
                        e._.negex = True
            #tokens         
            for t in doc[b[0] : b[1]]:
                if any(pre < t.i for pre in [i[1] for i in sub_preceding]):
                    t._.negex = True
                    continue
                if any(fol > t.i for fol in [i[2] for i in sub_following]):
                    t._.negex = True
                   
        return doc

    def __call__(self, doc):
        return self.negex(doc)
