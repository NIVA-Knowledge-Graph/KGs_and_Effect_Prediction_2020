

from rdflib import URIRef, Graph, Literal
from rdflib.namespace import RDFS, RDF

#graph = Graph()
#graph.load('../TERA_OUTPUT/effects.nt',format='nt')

#tests = set(graph.subjects(object=URIRef('https://cfpub.epa.gov/ecotox/cas/134623')))
#tests = tests.intersection(set(graph.subjects(object=URIRef('https://cfpub.epa.gov/ecotox/taxon/1'))))

#for j,t in enumerate(tests):
    #g = Graph()
    #results = set(graph.objects(subject=t,predicate=URIRef('https://cfpub.epa.gov/ecotox/hasResult')))
    
    #for i,r in enumerate(results):
        #g += graph.triples((t,None,None))
        
        #g += graph.triples((r,None,None))
        
        #for o in g.objects():
            #g += graph.triples((o,None,None))
        #g.serialize('./example_tests/%s_%s.nt' %(str(j),str(i)) ,format='nt')
        


graph = Graph()
graph.load('../TERA_OUTPUT/ecotox_taxonomy.nt',format='nt')
for s,p,o in graph:
    if p == RDF.type:
        graph.remove((s,p,o))
        graph.add((s,RDFS.subClassOf,o))
    
for s,p,o in graph:
    if isinstance(o,Literal):
        graph.remove((s,p,o))
        graph.add((s,RDFS.label,o))
        
graph.serialize('../TERA_OUTPUT/ecotox_taxonomy_aml.nt',format='nt')

graph = Graph()
graph.load('../TERA_OUTPUT/ncbi.nt',format='nt')

for d in range(12):
    new_graph = Graph()
    division = URIRef('https://www.ncbi.nlm.nih.gov/taxonomy/division/'+str(d))
    for t in graph.subjects(object=division):
        new_graph += graph.triples((t,None,None))
        new_graph += graph.triples((None,None,t))
        
    
    new_graph.remove((None,RDF.type,division))
    new_graph.remove((None,RDFS.subClassOf,division))
    
    new_graph.serialize('../TERA_OUTPUT/ncbi%s.nt' % str(d), format='nt')
    
    for s,p,o in new_graph:
        if p == RDF.type:
            new_graph.remove((s,p,o))
            new_graph.add((s,RDFS.subClassOf,o))
    
    
    for s,p,o in new_graph:
        if isinstance(o,Literal):
            new_graph.remove((s,p,o))
            new_graph.add((s,RDFS.label,o))

    new_graph.serialize('../TERA_OUTPUT/ncbi%s_aml.nt' % str(d), format='nt')
