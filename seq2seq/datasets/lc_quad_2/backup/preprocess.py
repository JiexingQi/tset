import os
import re
import json
import stanza

class Preprocess(object):
    def __init__(self):
        
        path = './seq2seq/datasets/lc_quad_2'
        ent_labels = json.load(open(os.path.join(path, 'entities.json'), 'rb'))
        rel_labels = json.load(open(os.path.join(path, 'relations.json'), 'rb'))
        
        vocab = ['"', '(', 'rdfs:label', 'by', 'ask', '>', 'select', 'que', 'limit', 'jai', 'mai', 
        '?sbj', ')', 'lang', 'year', '}', '?value', 'peint', 'desc', 'where', 'ce', 'distinct', 
       'filter', 'lcase', 'order', 'la', '<', 'asc', 'en', 'contains', 'as', ',', 'strstarts', 
       '{', "'", 'j', 'count', '=', '.', '?vr0', '?vr1', '?vr2', '?vr3', '?vr4', '?vr5', '?vr6', 
       '?vr0_label', '?vr1_label', '?vr2_label', '?vr3_label', '?vr4_label', '?vr5_label', '?vr6_label',
       'wd:', 'wdt:', 'ps:', 'p:', 'pq:', '?maskvar1', '[DEF]','[Entity]', '[Relation]', 'null']

        vocab_dict={}
        for i,text in enumerate(vocab):
            vocab_dict[text]='<extra_id_'+str(i)+'>'

        for kk in ent_labels:
            if ent_labels[kk] is None: ent_labels[kk] = vocab_dict['null']

        self.ent_labels = ent_labels
        self.rel_labels = rel_labels
        self.vocab_dict = vocab_dict
        
        # self.nlp_tokenize = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', 
        #                             tokenize_pretokenized = False, use_gpu=True)

    
    def _preprocess(self, data):
        wikisparql = data['sparql_wikidata']
        raw_question = data['question']
        if raw_question is None:
            raw_question = data['NNQT_question']
        raw_question = raw_question.replace('}','').replace('{','')

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

        # _ents += _rels
    
        newvars = ['?vr0','?vr1','?vr2','?vr3','?vr4','?vr5']
        
        variables = set([x for x in sparql.split() if x[0] == '?'])
        for idx,var in enumerate(sorted(variables)):
            if var == '?maskvar1':
                continue         
            sparql = sparql.replace(var,newvars[idx])
            
        split = sparql.split()
        for idx, item in enumerate(split):
            if item in self.ent_labels:
                split[idx] = self.ent_labels[item]
            elif item in self.rel_labels:
                split[idx] = self.rel_labels[item]

            if item in self.vocab_dict:
                split[idx] = self.vocab_dict[item]
        
        gold_query = ' '.join(split).strip()
        
        
        # doc = self.nlp_tokenize(raw_question)
        # question_toks = [w.text for s in doc.sentences for w in s.words]
        # question = ' '.join(question_toks)
        question = raw_question
        tail = ''
        
        for ent in _ents:
            ent = ent.replace('wd:',self.vocab_dict['wd:']+' ')

            ent_split = ent.split()
            index = 2 if bool(re.match('[pq][0-9]+', ent_split[1])) else 1
            tail = tail+' '+self.vocab_dict['[Entity]']+' '+ent_split[0]+' '+ ' '.join(ent_split[index:])


        for rel in _rels:
            rel=rel.replace('wdt:', self.vocab_dict['wdt:']+' ')
            rel=rel.replace('p:', self.vocab_dict['p:']+' ')
            rel=rel.replace('ps:', self.vocab_dict['ps:']+' ')
            rel=rel.replace('pq:', self.vocab_dict['pq:']+' ')

            rel_split = rel.split()
            index = 2 if bool(re.match('[pq][0-9]+', rel_split[1])) else 1
            tail = tail+' '+self.vocab_dict['[Relation]']+' '+rel_split[0]+' '+ ' '.join(rel_split[index:])
                
            # rel_item = ' '.join(rel_split[:index])+' '+self.vocab_dict[',']+' '+' '.join(rel_split[index:])
            
        schema = ' '.join(tail.split()).strip()
        question_input = ' '.join(question.split()).strip() + ' ' + schema
        
        res = {'sparql_process': gold_query,
               'question_process': question_input,
            #    'question_toks': question_toks,
               'question_toks': question.split(),
               'schema': schema,
               'raw_question': raw_question,}

        return res