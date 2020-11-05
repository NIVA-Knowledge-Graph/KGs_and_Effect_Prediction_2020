### analyse KGs

import numpy as np
from rdflib import Graph, URIRef
import pandas as pd

def relational_density(graph):
    return len(graph)/len(set(graph.predicates()))

def entity_density(graph):
    return 2*len(graph)/len(set(graph.subjects())|set(graph.objects()))

def relation_probability(graph):
    out = {}
    for p in set(graph.predicates()):
        out[p] = len(set(graph.triples((None,p,None))))/len(graph)
    return out

def entity_probability(graph):
    out = {}
    for e in set(graph.subjects())|set(graph.objects()):
        out[e] = (len(set(graph.triples((e,None,None))))+len(set(graph.triples((None,None,e)))))/len(graph)
    return out

def relational_entropy(graph):
    prob = relation_probability(graph)
    return sum(map(lambda x: -prob[x]*np.log(prob[x]),set(graph.predicates())))

def entity_entropy(graph):
    prob = entity_probability(graph)
    return sum(map(lambda x: -prob[x]*np.log(prob[x]),set(graph.subjects())|set(graph.objects())))

def absolute_density(graph):
    ent=set(graph.subjects())|set(graph.objects())
    return len(graph)/(len(ent)*(len(ent)-1))

def main():
    pdf = [pd.read_csv('./data/chemicals.csv'),pd.read_csv('./data/chemicals_extended.csv'),pd.read_csv('./data/chemicals_similarity.csv')]
    
    kg1 = pd.concat(pdf)
    kg2 = pd.read_csv('./data/taxonomy.csv')
    kg1 = list(zip(kg1['subject'], kg1['predicate'], kg1['object']))
    kg2 = list(zip(kg2['subject'], kg2['predicate'], kg2['object']))
    
    pdf = [pd.read_csv('../../Conv_Embedding/data/FB15k-237/train.txt',header=None,sep='\t'),
           pd.read_csv('../../Conv_Embedding/data/FB15k-237/valid.txt',header=None,sep='\t'),
           pd.read_csv('../../Conv_Embedding/data/FB15k-237/test.txt',header=None,sep='\t')]
    kg3 = pd.concat(pdf)
    kg3 = list(zip(kg3[0],kg3[1],kg3[2]))
    
    pdf = [pd.read_csv('../../Conv_Embedding/data/WN18/train.txt',header=None,sep='\t'),
           pd.read_csv('../../Conv_Embedding/data/WN18/valid.txt',header=None,sep='\t'),
           pd.read_csv('../../Conv_Embedding/data/WN18/test.txt',header=None,sep='\t')]
    kg4 = pd.concat(pdf)
    kg4 = list(zip(kg4[0],kg4[1],kg4[2]))
    
    
    pdf = [pd.read_csv('../../Conv_Embedding/data/WN18RR/train.txt',header=None,sep='\t'),
           pd.read_csv('../../Conv_Embedding/data/WN18RR/valid.txt',header=None,sep='\t'),
           pd.read_csv('../../Conv_Embedding/data/WN18RR/test.txt',header=None,sep='\t')]
    kg5 = pd.concat(pdf)
    kg5 = list(zip(kg5[0],kg5[1],kg5[2]))
    
    pdf = [pd.read_csv('../../Conv_Embedding/data/YAGO3-10/train.txt',header=None,sep='\t'),
           pd.read_csv('../../Conv_Embedding/data/YAGO3-10/valid.txt',header=None,sep='\t'),
           pd.read_csv('../../Conv_Embedding/data/YAGO3-10/test.txt',header=None,sep='\t')]
    kg6 = pd.concat(pdf)
    kg6 = list(zip(kg6[0],kg6[1],kg6[2]))
    
    
    for name, kg in zip(['KGc','KGs','FB15k-237','WN18','WN18RR','YAGO3-10'],[kg1,kg2,kg3,kg4,kg5,kg6]):
        tmp = Graph()
        for t in kg:
            try:
                tmp.add(tuple(map(URIRef,t)))
            except:
                tmp.add(tuple(map(lambda x: URIRef('http://example.org/'+str(x)),t)))
        print(name)
        print('RD',relational_density(tmp))
        print('ED',entity_density(tmp))
        print('RE',relational_entropy(tmp))
        print('EE',entity_entropy(tmp))
        print('ABS',absolute_density(tmp))
        print('\n')
    
    
if __name__ == "__main__":
    main()
    
    
