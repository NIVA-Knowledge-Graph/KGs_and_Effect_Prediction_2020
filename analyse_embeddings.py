
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

from analyse_results import load_results

def explained_variance():
    # explained variance of embeddings
    d = './results/pretrained_embeddings/'
    
    metric = r'Sensitivity' # Change to Specificty, YI, YI_max, etc.
    k = 0 # Must correspont to the metric. See load_results in analyse_results.py.
    mt = 'simple'
    
    vs = []
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
            
            v = p['value'][k]
                
            tmp1.append(ev)
            tmp2.append(v)
            
        evs.append(tmp1)
        vs.append(tmp2)
            
    colours = ['red','blue','green','black']
    labels = [r'$\it{(i)}$',r'$\it{(ii)}$',r'$\it{(iii)}$',r'$\it{(iv)}$']
    plt.figure(figsize=(10,10))
    for i in range(4):
        x = evs[i]
        y = vs[i]
        my_fitting, stats = poly.polyfit(x,y,1, full=True)
        R2 = stats[0][0]
        plt.scatter(x,y,color=colours[i])
        plt.plot(np.unique(x), np.poly1d(my_fitting[::-1])(np.unique(x)),color=colours[i],linewidth=4,label=labels[i])
    
    plt.xlabel('Explained variance', fontsize=18)
    plt.ylabel(metric, fontsize=18)
    plt.legend(fontsize=18)

    plt.savefig('./plots/%s_ev_vs_%s.png' % (mt,metric))


if __name__ == '__main__':
    explained_variance()

    
    
    
    
    
