## create data

"""
Steps:
1. Part
    1.1. Load ncbi taxonomy.
    1.2. Load ecotox effects.
    1.3. Use ncbi to ecotox mapping to replace species ids in ecotox effects.
    1.4. Use pubchem to cas mapping to replace chemical ids in ecotox effects.

2. Part
    2.1. Identify all species and chemicals used in relevant effects (eg. LC50).
    2.2. Find all triples in ncbi taxonomy which is connected to any effect species.
    2.3. Find all triples in chemble/pubchem/mesh which is connected to any effect chemical. 
    
3. Part
    3.1. Export effect data as tuples (chemical, species, concentration) .
    3.2. Export taxonomy triples as tuples (subject,predicate,object) .
    3.3. Export chemical triples as tuples (subject,predicate,object) .
    
"""

from tera.DataAggregation import Taxonomy, Effects, Traits
from tera.DataAccess import EffectsAPI
from tera.DataIntegration import DownloadedWikidata, LogMapMapping
from tera.utils import strip_namespace, unit_conversion
from tqdm import tqdm
from rdflib import Graph, URIRef, Literal, BNode, Namespace
import pandas as pd
import pubchempy as pcp
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pubchempy as pcp
from itertools import product

import networkx as nx
from rdflib.namespace import RDFS, RDF

import re

EPSILON = 10e-9

def get_subgraph(to_visit, graph, backtracking=0):
    print(backtracking, len(to_visit))
    out = set()
    visited = set()
    while to_visit:
        curr = to_visit.pop()
        visited.add(curr)
        tmp = set(graph.triples((curr,None,None)))
        out |= set([(s,p,o) for s,p,o in tmp if not isinstance(o,Literal)])
        to_visit |= set([o for _,_,o in tmp if not isinstance(o,Literal)])
        to_visit -= visited
        
    if backtracking > 0:
        tmp = set()
        for s in set([s for s,_,_ in out]):
            tmp |= set(graph.subjects(object=s))
        for t in out:
            graph.remove(t)
        return out | get_subgraph(tmp, graph, backtracking-1)
        
    return out


def get_longest_path(s,root,graph,p):
    #return triples in longest path to root:
    
    q = """select ?u ?v {
	<%s> <%s>* ?u .
	?u ?p ?v .
	?v <%s>* <%s> .

        } """ % (str(s),str(p),str(p),str(root))
    
    results = graph.query(q)
    g = nx.DiGraph()
    for u,v in results:
        g.add_edge(str(u),str(v))
    
    longest_path = []
    try:
        for path in nx.all_simple_paths(g, source=str(s), target=str(root)):
            if len(path) > len(longest_path):
                longest_path = path
    except nx.exception.NodeNotFound:
        pass
    
    return longest_path
    

def plot_data(filename):
    df = pd.read_csv(filename)
    y = np.asarray(df['concentration'])
    plt.hist(y,bins='auto')
    plt.show()

def main():
    
    t = Taxonomy(directory='../taxdump/', verbose=True)
    
    ne = DownloadedWikidata(filename='./data/ncbi_to_eol.csv', verbose=False)
    
    n = list(set(t.graph.subjects(predicate=t.namespace['rank'],
                                object=t.namespace['rank/species'])))

    tr = Traits(directory='../eol/', verbose=True)
    out = defaultdict(list)
    for p in set(tr.graph.predicates()):
        names = list(tr.graph.objects(subject=p,predicate=RDFS.label))
        if names:
            out['Entity'].append(str(p))
            out['Name'].append(str(names.pop(0)))
    df = pd.DataFrame(data=out)
    df.to_csv('traits_list.csv')
    
    conv = ne.convert(n, strip=True)
    converted = [(tr.namespace[i],k) for k,i in conv.items() if i != 'no mapping']
    

    tr.replace(converted)
    
    ed = Effects(directory='../ecotox_data/',verbose=False)
    
    n = list(set(t.graph.subjects(predicate=t.namespace['rank'], object=t.namespace['rank/species'])))
    
    species = LogMapMapping(filename='./data/final_mappings.txt',strip=True).convert(n,strip=True,reverse=True)
    species = [(ed.namespace['taxon/'+i],k) for k,i in species.items() if i != 'no mapping']
    
    ed.replace(species)
    
    n = list(set(ed.graph.objects(predicate=ed.namespace['chemical'])))
    chemicals = DownloadedWikidata(filename='./data/cas_to_mesh.csv').convert(n,reverse=False,strip=True)
    chemicals = [(k,URIRef('http://id.nlm.nih.gov/mesh/'+str(i))) for k,i in chemicals.items() if i != 'no mapping']
    
    ed.replace(chemicals)
    
    print('Part 1. done.')
    
    _,species = zip(*species)
    _,chemicals = zip(*chemicals)
    
    chemicals = set(map(str, chemicals))
    species = set(map(str, species))
    
    endpoints = EffectsAPI(dataobject=ed, verbose=True).get_endpoint(c=None, s=None)
             
    effects = [str(ed.namespace['effect/'+ef]) for ef in ['MOR','NER','DVP','GRO','IMM','INJ','ITX','MPH','PHY','REP']]
    d = defaultdict(list)
    for c,s,cc,cu,ep,ef,sd,sdu in tqdm(endpoints):
        if str(s) in species and str(c) in chemicals:
            if str(ef) in effects:
                try:
                    factor = unit_conversion(str(cu),'http://qudt.org/vocab/unit#MilligramPerLitre')
                except:
                    factor = 0
                    
                if factor > 0:
                    cc = float(cc)
                    cc = cc*factor
                    cc = np.log(cc+EPSILON)
                    
                    ep = str(ep).split('/')[-1]
                    try:
                        num = float('.'.join([re.findall(r'\d+', s).pop(0) for s in ep.split('.')]))
                    except IndexError:
                        continue
                    
                    d['degree'].append(num/100)
                    d['chemical'].append(str(c))
                    d['species'].append(str(s))
                    d['concentration'].append(cc)
                    d['effect'].append(str(ef))
    
    df = pd.DataFrame(data=d)
    df.to_csv('./data/data.csv')
    
    
    print('Part 2. done.')
    
    tmp = set([URIRef(a) for a in set(df['species'])])
    triples = get_subgraph(tmp, t.graph+tr.graph, backtracking=0)
    s,p,o = zip(*triples)
    data = {'subject':s, 
            'predicate':p, 
            'object':o}
    df = pd.DataFrame(data=data)
    df.to_csv('./data/taxonomy.csv')
    
    entities = set([s for s,p,o in triples]) | set([o for s,p,o in triples])
    print('Relational density, KGs', len(set(triples))/len(set([p for s,p,o in triples])))
    print('Entity density KGs', len(set(triples))/len(entities))
    print('Absolute density KGs', len(set(triples))/(len(entities)*(len(entities)-1)))
  
        
        
    df = pd.read_csv('./data/data.csv')
    mesh_graph = Graph()
    mesh_graph.parse('../mesh/mesh.nt',format='nt')
    
    triples = get_subgraph(set([URIRef(a) for a in set(df['chemical'])]), mesh_graph, backtracking=0)
    s,p,o = zip(*triples)
    data = {'subject':s, 
            'predicate':p, 
            'object':o}
    df = pd.DataFrame(data=data)
    df.to_csv('./data/chemicals.csv')
    
    """
    MOA 
    cco=http://rdf.ebi.ac.uk/terms/chembl#
    moa cco:hasMolecule, cco:hasTarget, cco:isTargetForMechanism
    TARGETS
    
    """
    triples = set()
    df = pd.read_csv('./data/data.csv')
    cco = Namespace('http://rdf.ebi.ac.uk/terms/chembl#')
    mapping = DownloadedWikidata(filename='./data/chembl_to_mesh.csv')
    c = set(df['chemical'])
    to_look_for = mapping.convert(c,reverse=True,strip=True)
    to_look_for = set([URIRef('http://rdf.ebi.ac.uk/resource/chembl/molecule/'+str(i)) for k,i in to_look_for.items() if i != 'no mapping'])
    chembl_graph = Graph()
    chembl_graph.load('../chembl/chembl_26.0_moa.ttl',format='ttl')
    chembl_graph.load('../chembl/chembl_26.0_target.ttl',format='ttl')
    chembl_graph.load('../chembl/chembl_26.0_targetrel.ttl',format='ttl')
    
    for s,p,o in get_subgraph(to_look_for,chembl_graph,backtracking=1):
        tmp = mapping.convert([str(o)],strip=True)[str(o)]
        o = 'http://id.nlm.nih.gov/mesh/' + tmp if tmp != 'no mapping' else str(o)
        tmp = mapping.convert([str(s)],strip=True)[str(s)]
        s = 'http://id.nlm.nih.gov/mesh/' + tmp if tmp != 'no mapping' else str(s)
        triples.add((str(s),str(p),str(o)))

    """
    take pubchem data
    
    cheminf:http://semanticscience.org/resource/
    Compounds: cheminf:CHEMINF_000478/80 (component), rdf:type, vocab:hasParentCompound, cheminf:CHEMINF_000482 (similarity) 
    """
    
    mapping1 = DownloadedWikidata(filename='./data/cid_to_mesh.csv')
    mapping2 = DownloadedWikidata(filename='./data/chebi_to_mesh.csv')
    
    mapping = mapping1 + mapping2
    
    to_look_for = mapping1.convert(set(df['chemical']),reverse=True,strip=True)
    to_look_for = set([URIRef('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/'+str(i)) for k,i in to_look_for.items() if i != 'no mapping'])
            
    pc_graph = Graph()
    d = '../pubchem/compound/general/'
    pc_graph.load(d+'pc_compound_type.ttl',format='ttl')
    pc_graph.load(d+'pc_compound_role.ttl',format='ttl')
    pc_graph.load(d+'pc_compound2drugproduct.ttl',format='ttl')
    pc_graph.load(d+'pc_compound2component.ttl',format='ttl')
    pc_graph.load(d+'pc_compound2parent.ttl',format='ttl')
    
    for s,p,o in get_subgraph(to_look_for,pc_graph,backtracking=0):
        tmp = mapping.convert([str(s)],strip=True)[str(s)]
        s = 'http://id.nlm.nih.gov/mesh/' + tmp if tmp !='no mapping' else s
        tmp = mapping.convert([str(o)],strip=True)[str(o)]
        o = 'http://id.nlm.nih.gov/mesh/' + tmp if tmp !='no mapping' else o
        triples.add((str(s),str(p),str(o)))
        
    
    s,p,o = zip(*triples)
    data = {'subject':s, 
            'predicate':p, 
            'object':o}
    df = pd.DataFrame(data=data)
    df.to_csv('./data/chemicals_extended.csv')
    
    entities = set([s for s,p,o in triples]) | set([o for s,p,o in triples])
    print('Relational density, KGc', len(set(triples))/len(set([p for s,p,o in triples])))
    print('Entity density KGc', len(set(triples))/len(entities))
    print('Absolute density KGc', len(set(triples))/(len(entities)*(len(entities)-1)))
    
    print('Part 3. done.')
    
def chemical_similarity():
    
    def tanimoto(fp1, fp2):
        fp1_count = bin(fp1).count('1')
        fp2_count = bin(fp2).count('1')
        both_count = bin(fp1 & fp2).count('1')
        return float(both_count) / (fp1_count + fp2_count - both_count)
    
    df = pd.read_csv('./data/data.csv')
    mapping = DownloadedWikidata(filename='./data/cid_to_mesh.csv')
    
    to_look_for = mapping.convert(set(df['chemical']),reverse=True,strip=True)
    to_look_for = set([URIRef('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID'+str(i)) for k,i in to_look_for.items() if i != 'no mapping'])
    
    triples = set()
    
    fp = {}
    for c in tqdm(to_look_for):
        try:
            compound = pcp.Compound.from_cid(int(c.split('CID')[-1]))
            fp[c] = int(compound.fingerprint,16)
        except:
            pass
        
    
    for c1, c2 in product(to_look_for,to_look_for):
        if c1 == c2: continue
        if c1 in fp and c2 in fp:
            score = tanimoto(fp[c1],fp[c2])
        else: score = 0
        if score >= 0.9: #default similarity in PubChem, https://jcheminf.biomedcentral.com/articles/10.1186/s13321-016-0163-1#Sec10
            tmp = mapping.convert([str(c1)],strip=True)[str(c1)]
            c1 = 'http://id.nlm.nih.gov/mesh/' + tmp if tmp !='no mapping' else c1
            tmp = mapping.convert([str(c2)],strip=True)[str(c2)]
            c2 = 'http://id.nlm.nih.gov/mesh/' + tmp if tmp !='no mapping' else c2
            
            triples.add((c1,'http://semanticscience.org/resource/CHEMINF_000482', c2))
            
    s,p,o = zip(*triples)
    data = {'subject':s, 
            'predicate':p, 
            'object':o}
    df = pd.DataFrame(data=data)
    df.to_csv('./data/chemicals_similarity.csv')
    
if __name__ == '__main__':
    main()
    chemical_similarity()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
