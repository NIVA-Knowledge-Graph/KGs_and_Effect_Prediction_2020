#STRING MATCHER

from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF

from fuzzywuzzy import process

from collections import defaultdict
from tqdm import tqdm

import glob
import pandas as pd
from itertools import product

import string 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
import numpy as np
stopwords = stopwords.words('english')
stopwords.extend(['sp','ssp','var'])

MIN_CONF = 0.5
ALPHA = 20

from time import time

import argparse

def clean_string(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

def cosine_sim_vectors(vec1, vec2):
    vec1 = vec1.reshape((-1,1))
    vec2 = vec2.reshape((-1,1))
    return cosine_similarity(vec1,vec2)[0][0]
    

def main(args):

    g1 = Graph()
    g1.parse(args.ontology1,format=args.ontology1.split('.')[-1])
    g2 = Graph()
    g2.parse(args.ontology2,format=args.ontology2.split('.')[-1])

    start_time = time()

    taxa1 = set(g1.subjects())
    taxa2 = set(g2.subjects())

    strings1 = defaultdict(list)
    strings2 = defaultdict(list)
    for g, lexical in zip([g1,g2],[strings1,strings2]):
        for s,p,o in g:
            if isinstance(o,Literal):
                lexical[s].append(str(o))

    lexical1 = defaultdict(list)
    lexical2 = defaultdict(list)


    for g, taxa, pred, lexical in zip([g1,g2],[taxa1,taxa2],[URIRef('https://cfpub.epa.gov/ecotox/latinName'),URIRef('https://www.ncbi.nlm.nih.gov/taxonomy/scientific_name')],[lexical1,lexical2]):
        for s,p,o in g:
            if p == pred and isinstance(o,Literal):
                tmp = str(o).split(' ')
                for t in tmp:
                    t = t.strip().lower()
                    if not t in stopwords:
                        lexical[t].append(s)

    candidate_labels = set(lexical1.keys()).intersection(set(lexical2.keys()))

    scores = defaultdict(lambda: float(0))

    for word in tqdm(candidate_labels):
        if word in lexical1 and word in lexical2:
            candidates1 = lexical1[word]
            candidates2 = lexical2[word]
            if len(candidates1) + len(candidates2) > 2*args.alpha: continue
        else:
            continue
        
        for candidate1 in candidates1:
            candidate_label1 = strings1[candidate1]
            for candidate2 in candidates2:
                candidate_label2 = strings2[candidate2]
                for label in candidate_label2:
                    label = clean_string(label)
                    candidate_label1 = list(map(clean_string,candidate_label1))
                    if not args.cosine:
                        match,score = process.extractOne(label,candidate_label1)
                        score /= 100
                    else:
                        score = 0
                        for cand_label in candidate_label1:
                            vectorizer = CountVectorizer(ngram_range=(2,3),analyzer='char_wb').fit_transform([label,cand_label])
                            vec1,vec2 = vectorizer.toarray()
                            score = max(score, cosine_sim_vectors(vec1,vec2))
                            
                    if score > args.confidence:
                        scores[(candidate1,candidate2)] = max(scores[(candidate1,candidate2)],score)
    
    print((time()-start_time)/60,'m')
    
    with open(args.output+'.txt','w') as f:
        for k in scores:
            k1,k2 = k
            tmp = '|'.join([str(k1),str(k2),str(scores[k])]) + '\n'
            f.write(tmp)
    
    g = Graph()
    ns = Namespace('http://knowledgeweb.semanticweb.org/heterogeneity/')
    alignment = BNode()
    g.add((alignment, RDF.type, ns['alignmentAlignment']))
    for p, l in zip(['alignmentxml','alignmentlevel','alignmenttype','alignmentonto1','alignmentonto2','alignmenturi1','alignmenturi2'],
                    ['yes','0','??',"http://logmap-tests/oaei/source.owl" ,"http://logmap-tests/oaei/target.owl","http://logmap-tests/oaei/source.owl","http://logmap-tests/oaei/target.owl"]):
        g.add((alignment, ns[p], Literal(l)))
    
    for k in scores:
        k1,k2 = k
        tmp = BNode()
        g.add((tmp,RDF.type,ns['alignmentCell']))
        g.add((tmp,ns['alignmententity1'],URIRef(k1)))
        g.add((tmp,ns['alignmententity2'],URIRef(k2)))
        g.add((tmp,ns['alignmentmeasure'],Literal(scores[k])))
        g.add((tmp,ns['alignmentrelation'],Literal('=')))
        g.add((alignment, ns['alignmentmap'], tmp))
        
    g.serialize(args.output+'.rdf',format='xml')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ontology1")
    parser.add_argument("ontology2")
    parser.add_argument("output", help='output file name')
    parser.add_argument("confidence", type=float, default=0.5)
    parser.add_argument("alpha", type=int, default=10, help="Max entities to consider during indexing.")
    parser.add_argument("--cosine", help="Use cosine similarity over Levenshtein distance.", action='store_true')
    args = parser.parse_args()
    print(args)
    main(args)
        
        
        
        
        
        
        
        
        
        
