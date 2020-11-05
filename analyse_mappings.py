
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDFS
from tera.DataIntegration import LogMapMapping, Alignment
from tera.DataAggregation import Taxonomy, EcotoxTaxonomy
from tera.DataAccess import TaxonomyAPI

import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
from tqdm import tqdm

import glob
import pandas as pd

from random import choices

EPSILON = np.finfo(np.float32).eps

def precision(refrence, alignment):
    return len(refrence.intersection(alignment))/(len(alignment) + EPSILON)

def recall(refrence, alignment):
    return len(refrence.intersection(alignment))/(len(refrence) + EPSILON)

def f_measure(refrence, alignment, b = 1):
    p = precision(refrence, alignment)
    r = recall(refrence, alignment)
    return (1+b**2)*(p*r)/(b**2*p + r + EPSILON)

def reverse_mapping(mapping):
    out = {}
    for k in mapping:
        if isinstance(mapping[k],(list,set)):
            for a in mapping[k]:
                out[a] = [k]
        else:
            out[mapping[k]] = [k]
    return out

def make_mapping_unique(mapping):
    return {k:mapping[k].pop() for k in mapping}

def load_mapping(f, th=0.8, filtered=False, ref1=[], ref2=[], unique=True):
    mapping1 = set()
    mapping2 = set()
    m = LogMapMapping(f, strip=False, threshold=th, unique=unique)
    m.load()
    um = m.mappings
    if unique:
        um = make_mapping_unique(um)
    for k in um:
        if filtered:
            if not k in ref1: continue
        e = um[k]
        if unique and not isinstance(e,list):
            e = [e]
        for a in e:
            mapping1.add((k,a,m.scores[(k,a)]))
    
    rm = reverse_mapping(um)
    if unique:
        rm = make_mapping_unique(rm)
    for k in rm:
        if filtered:
            if not k in ref2: continue
        e = rm[k]
        if unique and not isinstance(e,list):
            e = [e]
        for a in e:
            mapping2.add((k,a,m.scores[(a,k)]))
            
    return mapping1, mapping2

def make_unique(fm, k = 1):
    tmp = defaultdict(list)
    
    for e1,e2,score in fm:
        tmp[e1].append((e2,score))
    out = set()
    for e1 in tmp:
        sorted_tmp = dict(sorted(tmp[e1], key=lambda x: x[1], reverse=True))
        for e2 in list(sorted_tmp.keys())[:k]:
            out.add((e1,e2))
    return out

def make_unique_with_voting(mapping, voting_mapping, k=1):
    tmp = set()
    for e1,e2,score1 in mapping:
        for a1,a2,score2 in voting_mapping:
            if a1==e1 and a2==e2:
                tmp.add((e1,e2,score2))
            else:
                tmp.add((e1,e2,score1))
    return make_unique(tmp,k=k)
            

def filter_species_only(mapping):
    out = set()
    for e1,e2,score in mapping:
        try:
            int(e1.split('/')[-1])
            int(e2.split('/')[-1])
            out.add((e1,e2,score))
        except:
            pass
    return out
            
def main(filtered=False,unique=False,ncbi_instances=None,ecotox_instances=None,top_k=1):
    
    reference_mapping1 = set()
    reference_mapping2 = set()
    
    unique = unique and top_k < 2
    ref1 = set()
    ref2 = set()
    
    for fn in glob.glob('./reference_mappings/*txt'):
        df = pd.read_csv(fn,'|')
        for s,o in zip(df['Species ECOTOX Number'],df['Species NCBI TaxID']):
            if s and o:
                try:
                    s = 'https://cfpub.epa.gov/ecotox/taxon/'+str(int(s))
                    o = 'https://www.ncbi.nlm.nih.gov/taxonomy/taxon/'+str(int(o))
                    reference_mapping1.add((str(s),str(o)))
                    ref1.add(str(s))
                    ref2.add(str(o))
                except:
                    pass
    
    print('Number of reference mappings: ',len(reference_mapping1),'\n')
    
    
    mapping1 = {}
    
    for d,fn,th in zip(['logmap_outputs/','aml_output/','string_matcher_output/'],['/logmap_mappings.txt','.rdf','.txt'],[0.0,0.0,0.6]):
        final_mapping1 = set()
        final_mapping2 = set()
        for part in range(0,11):
            try:
                f = filename = d+str(part)+fn
                tmp1, tmp2 = load_mapping(f,th=th,filtered=filtered,ref1=ref1,ref2=ref2,unique=False)
                final_mapping1 |= tmp1
            except:
                pass
            
        if ncbi_instances and ecotox_instances:
            tmp = set()
            for a,b,score in final_mapping1:
                if a in ecotox_instances and b in ncbi_instances:
                    tmp.add((a,b,score))
            final_mapping1 = tmp 
            
        mapping1[d] = final_mapping1
    
    tmp1,tmp2 = {},{}
    for d1 in mapping1:
        for d2 in mapping1:
            if d1==d2: continue
            tmp1[d1+' UNION '+d2] = mapping1[d1] | mapping1[d2]
            tmp = set([(a,b) for a,b,_ in mapping1[d1]]).intersection(set([(a,b) for a,b,_ in mapping1[d2]]))
            tmp1[d1+' INTERSECTION '+d2] = set([(a,b,c) for a,b,c in mapping1[d1] if (a,b) in tmp]) | set([(a,b,c) for a,b,c in mapping1[d2] if (a,b) in tmp])
            tmp1[d1+' (AMBIGUITY RESOLVED WITH %s)' % d2] = mapping1[d1] | set([(a,b,2) for a,b in make_unique(mapping1[d2],k=1).intersection(make_unique(mapping1[d2],k=1))])
    
    mapping1['ALL'] = set.union(*[mapping1[d] for d in mapping1])
    tmp = set.intersection(*[set([(a,b) for a,b,_ in mapping1[d]]) for d in mapping1])
    mapping1['CONSENSUS'] = set.union(*[set([(a,b,c) for a,b,c in mapping1[d] if (a,b) in tmp]) for d in mapping1])
            
    mapping1 = {**tmp1,**mapping1}
    
    if unique:
        for k in mapping1:
            mapping1[k] = make_unique(mapping1[k],k=top_k)
    else:
        for k in mapping1:
            mapping1[k] = set([(a,b) for a,b,c in mapping1[k]])
            
    
    for d in mapping1:
        print(d, 'number of mappings', len(mapping1[d]))
        print(d,'precision',precision(reference_mapping1,mapping1[d]))
        print(d,'recall',recall(reference_mapping1,mapping1[d]))
        print(d,'f1',f_measure(reference_mapping1,mapping1[d]))
        print('\n')
    
    for d1 in mapping1:
        print(d1)
        print('# mapping',len(mapping1[d1]))
        for d2 in mapping1:
            tmp = mapping1[d1] - mapping1[d2]
            try:
                print('Disagreement ',d1,'\t',d2,len(tmp)/len(mapping1[d1]))
            except:
                pass
        print('\n')
    
    for d in mapping1:
        mappings = mapping1[d]
        filename = 'output_mappings/'+d.replace('/','').replace(' ','_')+'.txt'
        with open(filename,'w') as f:
            s = '|'.join(['ecotox','ncbi']) +'\n'
            f.write(s)
            for e1,e2 in mappings:
                s = '|'.join([e1,e2]) +'\n'
                f.write(s)
    
    with open('output_mappings/final_mappings.txt','w') as f:
        s = '|'.join(['e1','e2','score']) +'\n'
        f.write(s)
        for e1,e2 in mapping1['logmap_outputs/ UNION aml_output/']:
            s = '|'.join([e1,e2,"1"]) +'\n'
            f.write(s)

if __name__ == '__main__':
    #Filter species.
    graph = Graph()
    graph.load('../TERA_OUTPUT/ncbi.nt',format='nt')
    ncbi_instances = set(graph.subjects(object=URIRef('https://www.ncbi.nlm.nih.gov/taxonomy/rank/species')))
    ncbi_instances = set(map(str,ncbi_instances))
        
    graph = Graph()
    graph.load('../TERA_OUTPUT/ecotox_taxonomy.nt',format='nt')
    ecotox_instances = set(graph.subjects(object=URIRef('https://cfpub.epa.gov/ecotox/rank/species')))
    ecotox_instances = set(map(str,ecotox_instances))
    
    #1-N
    main(filtered=False,unique=False, ncbi_instances=ncbi_instances,ecotox_instances=ecotox_instances)
    #1-N precision
    main(filtered=True,unique=False)
    #1-1
    main(filtered=False,unique=True, ncbi_instances=ncbi_instances,ecotox_instances=ecotox_instances)
    #1-1 precision
    main(filtered=True,unique=True)
    
    
    
    







