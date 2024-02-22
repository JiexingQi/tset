import re

def readable(string):
    vocab=['"', '(', 'rdfs:label', 'by', 'ask', '>', 'select', 'que', 'limit', 'jai', 'mai', 
        '?sbj', ')', 'lang', 'year', '}', '?value', 'peint', 'desc', 'where', 'ce', 'distinct', 
       'filter', 'lcase', 'order', 'la', '<', 'asc', 'en', 'contains', 'as', ',', 'strstarts', 
       '{', "'", 'j', 'count', '=', '.', '?vr0', '?vr1', '?vr2', '?vr3', '?vr4', '?vr5', '?vr6', 
       '?vr0_label', '?vr1_label', '?vr2_label', '?vr3_label', '?vr4_label', '?vr5_label', '?vr6_label',
       'wd:', 'wdt:', 'ps:', 'p:', 'pq:', '?maskvar1', '[DEF]','null']

    vocab_dict = {}
    for i,text in enumerate(vocab):
        vocab_dict['<extra_id_'+str(30+i)+'>'] = text
                        
    for key in vocab_dict:
        string = string.replace(key,' '+vocab_dict[key]+' ')
        
    vals = string.split()
            
    for k in range(len(vals)):
        if bool(re.match(r'q[0-9]+',vals[k])):
            vals[k] = 'Q'+vals[k][1:]
        elif bool(re.match(r'p[0-9]+',vals[k])):
            vals[k] = 'P'+vals[k][1:]
                        
    return ' '.join(vals)


def change(string):
    string=string.replace('( ','(').replace(' )',')').replace('{ ',' {')\
    .replace(' }','}').replace(': ',':').replace(' , ',', ').replace(" ' ","'")\
    .replace("' ","'"). replace(" '","'").replace(' = ', '=').strip()
    
    rep_dec=re.findall('[0-9] \. [0-9]',string)
    for dec in rep_dec:
        string=string.replace(dec,dec.replace(' . ','.'))
    
    return ' '.join(string.split())


def process(string):
    string_prefix = 'PREFIX p: <http://www.wikidata.org/prop/> PREFIX pq: <http://www.wikidata.org/prop/qualifier/> PREFIX ps: <http://www.wikidata.org/prop/statement/> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wds: <http://www.wikidata.org/entity/statement/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> '
    return string_prefix + change(readable(string))