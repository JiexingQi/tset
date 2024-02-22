import os
import re
import json

class Preprocess(object):
    def __init__(self):
        
        path = './datasets/lc_quad_2'
        ent_labels = json.load(open(os.path.join(path, 'entities.json'), 'rb'))
        rel_labels = json.load(open(os.path.join(path, 'relations.json'), 'rb'))
        
        vocab=['"', '(', 'rdfs:label', 'by', 'ask', '>', 'select', 'que', 'limit', 'jai', 'mai', 
        '?sbj', ')', 'lang', 'year', '}', '?value', 'peint', 'desc', 'where', 'ce', 'distinct', 
       'filter', 'lcase', 'order', 'la', '<', 'asc', 'en', 'contains', 'as', ',', 'strstarts', 
       '{', "'", 'j', 'count', '=', '.', '?vr0', '?vr1', '?vr2', '?vr3', '?vr4', '?vr5', '?vr6', 
       '?vr0_label', '?vr1_label', '?vr2_label', '?vr3_label', '?vr4_label', '?vr5_label', '?vr6_label',
       'wd:', 'wdt:', 'ps:', 'p:', 'pq:', '?maskvar1','null']

        vocab_dict={}
        for i,text in enumerate(vocab):
            vocab_dict[text]='<extra_id_'+str(i)+'>'

        for kk in ent_labels:
            if ent_labels[kk] is None: ent_labels[kk] = vocab_dict['null']

        self.ent_labels = ent_labels
        self.rel_labels = rel_labels
        self.vocab_dict = vocab_dict

    def process(self, wikisparql):
        sparql = wikisparql.replace('(',' ( ').replace(')',' ) ').replace('{',' { ')\
        .replace('}',' } ').replace(':',': ').replace(',',' , ').replace("'"," ' ")\
        .replace('.',' . ').replace('=',' = ').lower()
        sparql = ' '.join(sparql.split())
        
        _ents = re.findall( r'wd: (?:.*?) ', sparql)
        _ents_for_labels = re.findall( r'wd: (.*?) ', sparql)
        
        _rels = re.findall( r'wdt: (?:.*?) ',sparql)
        _rels += re.findall( r' p: (?:.*?) ',sparql)
        _rels += re.findall( r' ps: (?:.*?) ',sparql)
        _rels += re.findall( r'pq: (?:.*?) ',sparql)
        
        _rels_for_labels = re.findall( r'wdt: (.*?) ',sparql)
        _rels_for_labels += re.findall( r' p: (.*?) ',sparql)
        _rels_for_labels += re.findall( r' ps: (.*?) ',sparql)
        _rels_for_labels += re.findall( r'pq: (.*?) ',sparql)

        for j in range(len(_ents_for_labels)):
            if '}' in _ents[j]: 
                _ents[j]=''
            _ents[j] = _ents[j] + self.ent_labels[_ents_for_labels[j]]+' '
            
        for j in range(len(_rels_for_labels)):
            if _rels_for_labels[j] not in self.rel_labels:
                self.rel_labels[_rels_for_labels[j]] = self.vocab_dict['null']
            _rels[j] = _rels[j] + self.rel_labels[_rels_for_labels[j]]+' '
    
            
        split = sparql.split()
        for idx, item in enumerate(split):
            if item in self.ent_labels:
                split[idx] = self.ent_labels[item]
            elif item in self.rel_labels:
                split[idx] = self.rel_labels[item]

            if item in self.vocab_dict:
                split[idx] = self.vocab_dict[item]
        
        gold_query = ' '.join(split).strip()
        
        return gold_query
        
    def _preprocess(self, data):
        res = { 'input_sparql': data['input'],
                'input_process': self.process(data['input']),
                'target_sparql': data['target'],
                'target_process': self.process(data['target']),
               }

        return res