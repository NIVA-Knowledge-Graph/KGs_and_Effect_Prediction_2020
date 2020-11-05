
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, SpectralEmbedding, TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from sklearn.preprocessing import normalize
import networkx as nx
from collections import defaultdict
from models.utils import load_data

from sklearn.model_selection import train_test_split

from sklearn import svm
import matplotlib.patches as mpatches

models = [
            'DistMult',
          'TransE',
          'HolE',
          'ComplEx',
          'RotatE',
          'pRotatE',
          'HAKE',
          'ConvE',
          'ConvKB'
          ]

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import numpy.polynomial.polynomial as poly

from analyse_results import load_predictions


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot_embeddings(X_prime,y_true,y_pred,filename):
    rgba_colors = np.zeros((len(y_true),4))
    
    colours = ['green' if y==0 else 'red' for y in y_true]
    
    fig, ax = plt.subplots()
    im = ax.scatter(x=X_prime[:,0],y=X_prime[:,1],c=colours)
    ax.set_xlabel('Principal component 1', fontsize=18)
    ax.set_ylabel('Principal component 2', fontsize=18)
    ax.legend(fontsize=18)
    red_patch = mpatches.Patch(color='red', label='Lethal')
    blue_patch = mpatches.Patch(color='green', label='Non-lethal')
    plt.legend(handles=[red_patch, blue_patch])
    fig.tight_layout()
    fig.set_size_inches(10, 10)
    plt.savefig(filename)
    plt.close()
    
def count(X):
    out_c = defaultdict(int)
    out_s = defaultdict(int)
    
    for c,s,conc in X:
        out_c[c] += 1
        out_s[s] += 1
    return out_c, out_s
    
def plot_fix_entity(X,y_true,y_pred,embedding_file1, embedding_file2, ids_file1, ids_file2, entity_to_fix,filename='plot.png'):
    X1 = np.load(embedding_file1)
    ids1 = dict(np.load(ids_file1))
    X2 = np.load(embedding_file2)
    ids2 = dict(np.load(ids_file2))
    
    if entity_to_fix in ids1:
        y_true = np.asarray([y for y,x in zip(y_true,X) if x[0]==entity_to_fix])
        y_pred = np.asarray([y for y,x in zip(y_pred,X) if x[0]==entity_to_fix])
        X = [(c,s,conc) for c,s,conc in X if c == entity_to_fix]
        X = np.asarray([np.concatenate([X1[int(ids1[entity_to_fix])],X2[int(ids2[s])],[conc]],axis=0) for _,s,conc in X if s in ids2])
    else:
        y_true = np.asarray([y for y,x in zip(y_true,X) if x[1]==entity_to_fix])
        y_pred = np.asarray([y for y,x in zip(y_pred,X) if x[1]==entity_to_fix])
        X = [(c,s,conc) for c,s,conc in X if s == entity_to_fix]
        X = np.asarray([np.concatenate([X1[int(ids1[c])],X2[int(ids2[entity_to_fix])],[conc]],axis=0) for c,_,conc in X if c in ids1])
        
    X = normalize(X,axis=0)
    pca = TSNE(n_components=2)
    X = pca.fit_transform(X)
    
    plot_embeddings(X,y_true,y_pred,filename)
    
def load_results(sm):
    res = {}
    for m1, m2 in product(models,models):
        df = pd.read_csv('results/'+'_'.join([sm,'simple','pretrained',m1,m2])+'.csv',index_col='metric')
        res[(m1,m2)] = df.loc['auc','value']
    return res
    
def predictions(filename):
    arr = np.load(filename)
    y_pred, y_true = arr[:-1,:], arr[-1,:]
    return np.around(np.mean(y_pred,axis=0))
    
    
def main():
    check_embeddings()
    """
    Get results. 2 best and 2 worst.
    define names for pre-trained and finetuned embedding files
    take top 4 most used chemicals and species and plot in 2d.
    """
    d = 'results/pretrained_embeddings/'
    for sampling_method in ['none','chemical','species','both']:
        X,y_true = load_data('./data/%s_data_test.csv' % sampling_method)
        count_c, count_s = count(X)
        count_c = dict(sorted(count_c.items(), key=lambda x: x[1], reverse=True))
        count_s = dict(sorted(count_s.items(), key=lambda x: x[1], reverse=True))
        results = load_results(sampling_method)
        for reverse in [True,False]:
            for res in list(sorted(results.items(),key=lambda x:x[1],reverse=reverse))[:1]:
                m,_ = res
                m1,m2 = m
                embedding_file1 = d+'_'.join([m1,'chemical','entity_embeddings']) + '.npy'
                embedding_file2 = d+'_'.join([m2,'taxonomy','entity_embeddings']) + '.npy'
                
                ids_file1 = d+'_'.join([m1,'chemical','entity_ids']) + '.npy'
                ids_file2 = d+'_'.join([m2,'taxonomy','entity_ids']) + '.npy'
                
                y_hat = predictions('results/predictions_%s_simple_pretrained_%s_%s.npy' % (sampling_method,m1,m2))
                
                for c in list(count_c.keys())[:2]:
                    plot_fix_entity(X,y_true,y_hat,embedding_file1,embedding_file2,ids_file1,ids_file2,c,filename='plots/%s_%s_%s.png' % (sampling_method,c.split('/')[-1],'best' if reverse else 'worst'))
                
                for s in list(count_s.keys())[:2]:
                    plot_fix_entity(X,y_true,y_hat,embedding_file1,embedding_file2,ids_file1,ids_file2,s,filename='plots/%s_%s_%s.png' % (sampling_method,s.split('/')[-1],'best' if reverse else 'worst'))
    
def check_embeddings():
    
    d = './results/pretrained_embeddings/'
    for model1 in models:
        f = d+model1+'_chemical_entity_embeddings.npy'
        X1 = np.load(f)
        try:
            assert not np.any(np.isnan(X1))
        except:
            print(model1,'chemical')
            
    for model2 in models:
        f = d+model2+'_taxonomy_entity_embeddings.npy'
        X2 = np.load(f)
        try:
            assert not np.any(np.isnan(X2))
        except:
            print(model2,'species')
        
def explained_variance():
    # explained variance of embeddings
    d = './results/pretrained_embeddings/'
    
    metric = r'Sensitivity'
    #Sensitivity and specificity
    mt = 'simple'
    
    aucs = []
    evs = []
    
    for j,s in enumerate(['none','chemical','species','both']):
        tmp1,tmp2 = [],[]
        for model1, model2 in product(models,models):
            
            X,y = load_data('./data/%s_data_test.csv' % s)
            y = np.asarray(y)

            f = d+model1+'_chemical_entity_embeddings.npy'
            X1 = np.load(f)
            f = d+model1+'_chemical_ids.npy'
            ids1 = dict(np.load(f))

            f = d+model2+'_taxonomy_entity_embeddings.npy'
            X2 = np.load(f)
            f = d+model2+'_taxonomy_ids.npy'
            ids2 = dict(np.load(f))
        
            X = np.asarray([np.concatenate([X1[int(ids1[c])],X2[int(ids2[s])],[conc]],axis=0) for c,s,conc in X if c in ids1 and s in ids2])
            X = normalize(X,norm='l2',axis=0) # normalize over each feature
        
            pca = PCA(n_components=10)
            pca.fit(X)
            ev = sum(pca.explained_variance_ratio_)
            
            f = 'results/%s_%s_pretrained_%s_%s.csv' % (s,mt,model1,model2)
            p = load_predictions(f.replace('/','/predictions_').replace('csv','npy'))
            
            auc = p['value'][0]
                
            tmp1.append(ev)
            tmp2.append(auc)
            
        evs.append(tmp1)
        aucs.append(tmp2)
            
    colours = ['red','blue','green','black']
    labels = [r'$\it{(i)}$',r'$\it{(ii)}$',r'$\it{(iii)}$',r'$\it{(iv)}$']
    plt.figure(figsize=(10,10))
    for i in range(4):
        x = evs[i]
        y = aucs[i]
        my_fitting, stats = poly.polyfit(x,y,1, full=True)
        R2 = stats[0][0]
        plt.scatter(x,y,color=colours[i])
        plt.plot(np.unique(x), np.poly1d(my_fitting[::-1])(np.unique(x)),color=colours[i],linewidth=4,label=labels[i])
    
    plt.xlabel('Explained variance', fontsize=18)
    plt.ylabel(metric, fontsize=18)
    plt.legend(fontsize=18)

    plt.savefig('./plots/%s_ev_vs_%s.png' % (mt,metric))

    #plt.show()

    
def reduce_kg_to_hier(kg,hier_rels):
    tmp = []
    for s,p,o in kg:
        for direction,rel in hier_rels:
            if p == rel:
                if direction > 0:
                    tmp.append((s,o))
                else:
                    tmp.append((o,s))
    return tmp

def reduce_kg_to_objects(kg,objects,type_rel):
    tmp = set()
    for s,p,o in kg:
        if p == type_rel and o in objects:
            tmp.add(s)
    return [(s,p,o) for s,p,o in kg if s in tmp and o in tmp]

def to_networkx_graph(edges):
    g = nx.DiGraph()
    for n1,n2 in edges:
        g.add_edge(n1,n2)
    return g

def calculate_score(g,embeddings,ids):
    leafs = [x for x in g.nodes() if g.in_degree(x)==0]
    score = 0
    d = defaultdict(set)
    for l in leafs:
        if l in g:
            for s in g.successors(l):
                d[s].add(l)
    
    transforms = []
    
    for k in d:
        if len(d[k]) > 1:
            X = np.mean(np.asarray([embeddings[int(ids[n])] for n in d[k]]),axis=0)
            M = embeddings[int(ids[k])]
            T = np.linalg.solve(np.diag(X), M)
            transforms.append(T)
        for n in d[k]:
            if n in g: g.remove_node(n)
    
    for T1, T2 in product(transforms,transforms):
        score += np.mean(abs(T1-T2))
    score = score/max(1,len(transforms))**2
    
    if len(d.keys()) > 0:
        score += calculate_score(g,embeddings,ids)
    
    return score
    
def locality():
    
    kg1 = pd.read_csv('./chemicals0.csv')
    kg2 = pd.read_csv('./taxonomy0.csv')
    
    kg1 = list(zip(kg1['subject'], kg1['predicate'], kg1['object']))
    kg2 = list(zip(kg2['subject'], kg2['predicate'], kg2['object']))
    
    type_rel = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
    
    kg1_hier_rels = [(1,'http://id.nlm.nih.gov/mesh/vocab#broaderConcept'),
                     (-1,'http://id.nlm.nih.gov/mesh/vocab#narrowerConcept')]
    kg1_hier_objects = ['http://id.nlm.nih.gov/mesh/vocab#Concept']
    
    kg2_hier_rels = [(1,'http://www.w3.org/2000/01/rdf-schema#subClassOf')]
    kg2_hier_objects = ['https://www.ncbi.nlm.nih.gov/taxonomy/Taxon']
    
    kg1 = reduce_kg_to_objects(kg1,kg1_hier_objects,type_rel)
    kg2 = reduce_kg_to_objects(kg2,kg2_hier_objects,type_rel)
    kg1 = reduce_kg_to_hier(kg1,kg1_hier_rels)
    kg2 = reduce_kg_to_hier(kg2,kg2_hier_rels)
    
    tmp1 = []
    
    for model in models:
        g1 = to_networkx_graph(kg1)
        f = 'pretrained/'+model+'_chemical_embeddings.npy'
        X = np.load(f)
        X = normalize(X,norm='l2',axis=1)
        f = 'pretrained/'+model+'_chemical_ids.npy'
        ids = dict(np.load(f))
        tmp1.append(calculate_score(g1,X,ids))
            
    tmp2 = []
    
    for model in models:
        g2 = to_networkx_graph(kg2)
    
        f = 'pretrained/'+model+'_taxonomy_embeddings.npy'
        X = np.load(f)
        X = normalize(X,norm='l2',axis=1)
        f = 'pretrained/'+model+'_taxonomy_ids.npy'
        ids = dict(np.load(f))
        tmp2.append(calculate_score(g2,X,ids))
    
    mat = []
    for a,b in product(tmp1,tmp2):
        mat.append(a*b)
    
    mat = np.reshape(np.asarray(mat),(len(models),len(models)))
    mat = np.around(mat,2)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    im = ax.imshow(mat)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(models)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(models)
    ax.set_yticklabels(models)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(models)):
        for j in range(len(models)):
            text = ax.text(j, i, mat[i, j],
                        ha="center", va="center", color="w")

    #ax.set_title("Locality metric")
    ax.set_xlabel('Chemical KG', fontsize=18)
    ax.set_ylabel('Taxonomy KG', fontsize=18)
    fig.tight_layout()
    plt.savefig('./plots/locality.png')
    plt.close()


if __name__ == '__main__':
    main()
    #explained_variance()
    #locality()
    
    
    
    
    
    
