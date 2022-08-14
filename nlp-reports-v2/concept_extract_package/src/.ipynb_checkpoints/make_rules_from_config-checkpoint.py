import pandas as pd
import yaml
import os


#generate functions quickly when making config
def make_kwrule_dict(RULE, ENTITY, COMP_OPP, SERIES, CONCEPT):
    return {'rule': RULE, 'ents': ENTITY, 'complement': COMP_OPP, 'series': SERIES, 'wordlist1': CONCEPT}

def make_2concept_rule_dict(RULE, ENTITY, COMP_OPP, SERIES, CONCEPTS):
    return {'rule': RULE, 'ents': ENTITY, 'complement': COMP_OPP, 'series': SERIES, 'wordlist1' : CONCEPTS[0], 'wordlist2' : CONCEPTS[1]}

def make_regex_rule_dict(RULE, REGEX_PAT, COMP_OPP, SERIES):
    return {'rule': RULE, 'regex_pat': REGEX_PAT, 'complement': COMP_OPP, 'series': SERIES}



class RuleParameters():
    "This creates the snorkel structured rules for a single rule set from the config.yaml"
    def __init__(self, config_path, rule_set = '', num = None):
        """
        Class that generates snorkel rules from conifg.yaml inputs for a single ruleset
        in > str path to config.yaml as specificied in make_config notebook, rule_set > str specifying ruleset dictionary
        
        out > if ruleset specified, Sets of rules for each rule set to feed into snorkel rules
        out > with default settings, config object that can run nlp pipeline
        
        if init with default rule_set and num, will return config settings to run NLP portion of pipeline only
        if rule_set and number of concepts are specified, will be able to run LF maker
        """
        
        with open(config_path, 'r') as f:
            self.CFG = yaml.safe_load(f)
        
        
        self.FILE_PATHS = self.CFG['FILE_PATHS']
        
        self.DATA_INFO = self.CFG['DATA']
        self.retokenization_list = self.CFG['NLP_CFG']['Retokenization_list']
        self.nlp_cfg = self.CFG['NLP_CFG']
        
        self.num = num
        
        if num is not None:
            try:
                self.argDict = self.CFG['RuleSets'][rule_set]
                self.metadata = {k:v for (k,v) in self.argDict.items() if k not in ['Rules', 'Concepts']}
                self.RulesDicts = self.argDict['Rules']
        
    
                if self.metadata['num_concepts']==2:
                    self.concepts1 = self.argDict['Concepts']['concept1']
                    self.concepts2 =self.argDict['Concepts']['concept2']
                else:
                    self.ConceptsDict = self.argDict['Concepts']
                    
                self.sent_filt = [tup[1] for tup in self.DATA_INFO['data_columns_with_keywd_filter'] if tup[0] == 
                         self.metadata['data_column']]
                
                print(f'creating rules parameter object for {len(self.RulesDicts)} labeling functions for {self.metadata["num_concepts"]} concept ruleset {rule_set} with base text in {self.metadata["data_column"]}')
            
            except Exception as e:
            
                print(f' is {rule_set} in config rulesets?')
                print(e)
                
           
            
       

            
       
        
    def _update_series_input(self, seriestype):
        """
        in >  one of: 'text', 'nlp_doc', 'lemma_doc'
        out > indicator for appropriate column to run rule on
        """
        try:
            if seriestype == 'nlp_doc':
                return 'nlp_string'
            if seriestype== 'lemma_doc':
                return 'lemma_string'
            if seriestype =='text':
                return self.metadata['data_column']
        except:
            print('not a known series type. use one of: "text", "nlp_doc", "lemma_doc"')
        
    def _suffix_maker(self, ents, complements_or_opposite, seriestype, cnt = 0):
        """
        entity > bin, complement_or_opposite > bin, cnt > cnt, int to maintain uniqueness of label
        """
        pre = seriestype
        if ents is not None:
            su ='token' if ents == False  else 'entity'
            ffix = '_complement' if complements_or_opposite == True else ''
        else:
            su = 'string'
            ffix = '_complement'  if complements_or_opposite == True else ''
             
        xx = "_" + str(cnt)
                
        return pre + '_' +su+ffix +xx


    def _make_LFargDict(self, rulesdict, conceptsdict):
        """
         #cfg['RuleSets']['Overall']['Rules'][2]
         make a single LF function arguments to feed to LabelingFunction()
        """
        
        varnames = [key for key in rulesdict if key != 'rule']
        LFargDict = {}
        
        for varname in varnames:
            LFargDict[varname] = rulesdict[varname]
        


        LFargDict['series'] = self._update_series_input(LFargDict['series'])
        updates = [k for k in LFargDict.keys() if 'wordlist' in k]
        for el in updates:
            LFargDict[el] = conceptsdict[LFargDict[el]]
            
        return(LFargDict)


    def _single_rule_maker(self, rulesdict, conceptsdict, cnt=0):
        """
        create LF dict for single LF ruleset
        dicts from cfg file
        """
    
        if 'ents' in rulesdict.keys():
            suf_ent = rulesdict['ents']
            #print(suf_ent)
        else:
            suf_ent = None
        d = {'fun': rulesdict['rule'], 'Dict': self._make_LFargDict(rulesdict, conceptsdict),
         'suffix': self._suffix_maker(suf_ent, rulesdict['complement'], rulesdict['series'], cnt)}
        #print(d['suffix'])
        return d



    def singleConcept_LF_maker(self, GoldStd = True):
        """
        create LF dict and assoc metadata for all LF rules in given set for a single concept (eg: 'Scar' or 'Ischemia')
        return snorkelrules and metadata to run pipepline
        """
        SnorkelRules = {}
    
        RulesDicts = self.RulesDicts
        ConceptsDict = self.ConceptsDict
        metadata = self.metadata
    
    
        if GoldStd:
            true_columns = self.argDict['Concepts']['true_columns'] 
    
        rule_labels = self.argDict['Concepts']['rule_names']
    
        lfdict = {}

        for i in range(len(rule_labels)):
            rule_label = rule_labels[i]
            true_column = true_columns[i]
            predicted_column = f'predicted_{rule_labels[i]}'
        
            for i, key in enumerate(RulesDicts):
                d = self._single_rule_maker(RulesDicts[key], ConceptsDict, i)
                lfdict[key] = d
            concept1 = ConceptsDict['concept1']
        
        
            SnorkelRules.update({rule_label: {'pathology': concept1, 'true_col': true_column,
                            'predicted_column': predicted_column, 'inputArgs': lfdict  
                            } })
        
        return SnorkelRules


    def _dualConcept_expander(self, GoldStd = True):
        """
        in > dual concept argDict
        run  expansion and create concepts dict with concept expansion for dual concepts
        """
        concepts1 = self.concepts1
        concepts2 = self.concepts2
    
        ConceptsDictExpanded = {}
        count = 0
        for c1 in concepts1:
            for c2 in concepts2:
                d = {'concept1': c1, 'concept2': c2}
                count += 1
                ConceptsDictExpanded[count - 1] = d
    
        true_columns = self.argDict['Concepts']['true_columns'] 
        rule_labels = self.argDict['Concepts']['rule_names']

            
    
        #check len to make sure expansion worked as expected
        assert len(ConceptsDictExpanded) == len(rule_labels)
        if GoldStd == True:
            assert len(ConceptsDictExpanded) == len(true_columns)
        
            for i, (tc, rl) in enumerate(zip(true_columns, rule_labels)):
                ConceptsDictExpanded[i]['true_columns'] = tc
                ConceptsDictExpanded[i]['rule_names'] = rl
                print(f'making LF dict for {rl} with concepts {ConceptsDictExpanded[i]["concept1"]} and \
                {ConceptsDictExpanded[i]["concept2"]}')
        else:
            for i, rl in enumerate(rule_labels):
                ConceptsDictExpanded[i]['true_columns'] = 'no true col'
                ConceptsDictExpanded[i]['rule_names'] = rl
                print(f'making LF dict for {rl} with concepts {ConceptsDictExpanded[i]["concept1"]} \
                and {ConceptsDictExpanded[i]["concept2"]}')
    
        return ConceptsDictExpanded


    def dualConcept_LF_maker(self, GoldStd = True):
        """
        create LF dict for all LF rules in given set for a dual concept (eg: 'Scar' in the "RCA")
        return snorkelrules and metadata to run pipepline
        """
        metadata = self.metadata
        RulesDicts = self.RulesDicts
        ConceptsDicts = self._dualConcept_expander(GoldStd)

        SnorkelRules = {}
        for key in ConceptsDicts:
            Dict = ConceptsDicts[key]
            lfdict = {}
            for key in RulesDicts:
                d = self._single_rule_maker(RulesDicts[key], Dict)
                lfdict[key] = d
    
            predicted_column = f'predicted_{Dict["rule_names"]}'
            name = Dict['rule_names']

            SnorkelRules.update({name: {'pathology': Dict['concept1'], 'concept2': Dict['concept2'], 'true_col': Dict['true_columns'], 'predicted_column': predicted_column, 'inputArgs': lfdict  
                            } })
        return SnorkelRules
    
    def run_LF_maker(self, GoldStd=True):
        """
        wraps single and dual concept LF maker
        returns snorkel rule set
        """
        num = self.num
        if num == 1:
            snorkelrules = self.singleConcept_LF_maker(GoldStd)
        elif num ==2:
            snorkelrules = self.dualConcept_LF_maker(GoldStd)
        else:
            snokelrules = None
            print('number of concepts must be 1 or 2')
        return snorkelrules
            
