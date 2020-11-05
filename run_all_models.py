
from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)

from models.utils import load_data, train_test_split_custom
from models.sim_embedding_models import fit_sim_model
from models.pretrained_auto_keras import fit_onehot, fit_pretrained, fit_hier_embeddings, fit_hier_kg_combination

import numpy as np
import pandas as pd

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import glob
import json

import argparse
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle

RANDOM_SEED = 42

KGE_EMBEDDINGS_DIR = './results/pretrained_embeddings/'

models = [
            'DistMult', 
            'TransE',
            'HolE',
            'ComplEx',
            'ConvE',
            'ConvKB',
            'RotatE',
            'pRotatE',
            'HAKE'
        ]

def save_data(filename,data):
    X,y = data
    y = ['https://cfpub.epa.gov/ecotox/effect/MOR' if a == 1 else 'https://cfpub.epa.gov/ecotox/effect/NON' for a in y]
    x1,x2,x3 = zip(*X)
    df = pd.DataFrame(data={'effect':y,'chemical':x1,'species':x2,'concentration':x3})
    df.to_csv(filename)
    
def main(args, params):
    
    #To approx 0.15/0.15/0.70 split in total data when splitting chemicals/species.
    sizes = {'none':(0.225,0.225),'species':(0.22,0.24),'chemical':(0.23,0.23),'both':(0.45,0.47)}
    SAMPLING = args.sampling
    if args.CREATE_DATA:
        valid_size,test_size = sizes[SAMPLING]
        
        X,y = load_data(DATA_FILE)
        train, valid, test = train_test_split_custom(X, y, valid_size=valid_size, test_size=test_size, sampling=SAMPLING, random_state=RANDOM_SEED)
        print(len(valid[1])/sum(map(len,[train[1],test[1],valid[1]])),len(test[1])/sum(map(len,[train[1],test[1],valid[1]])))
        save_data('data/%s_data_train.csv' % SAMPLING, train)
        save_data('data/%s_data_valid.csv' % SAMPLING, valid)
        save_data('data/%s_data_test.csv' % SAMPLING, test)
    try:
        train = load_data('data/%s_data_train.csv' % SAMPLING)
        valid = load_data('data/%s_data_valid.csv' % SAMPLING)
        test = load_data('data/%s_data_test.csv' % SAMPLING)
        print('Train Split',len(train[1])/sum(map(len,[train[1],test[1],valid[1]])))
        print('Valid Split',len(valid[1])/sum(map(len,[train[1],test[1],valid[1]])))
        print('Test Split',len(test[1])/sum(map(len,[train[1],test[1],valid[1]])))
        oversample = RandomOverSampler(sampling_strategy='minority')
        train = oversample.fit_resample(*train)
        train = shuffle(*train)
        test = shuffle(*test)
        valid = shuffle(*valid)
        
    except:
        args.CREATE_DATA = True
        return main(args,params)
    
    
    params['cw'] = None
    
    if args.SIMPLE: SAMPLING+='_simple'
    else: SAMPLING+='_complex'
    
    if args.model == "onehot":
        fit_onehot(train, valid, test,
                   results_file='results/%s_one_hot.csv' % SAMPLING, 
                   hp_file = 'pred_hp/%s_one_hot.csv' % SAMPLING,
                   params=params)
        
    if args.model == "hier": 
        fit_hier_embeddings(train, valid, test,
                            chemical_hier_embeddings_files,
                            taxonomy_hier_embeddings_files,
                            results_file='results/%s_hierarchy_embedding.csv' % SAMPLING,
                            hp_file='pred_hp/%s_hierarchy_embedding.csv' % SAMPLING,
                            params=params)
        
 
    if args.model == "pretrained":
        for model1 in models:
            for model2 in models:
                fit_pretrained(train, valid, test,
                               KGE_EMBEDDINGS_DIR+model1,
                               KGE_EMBEDDINGS_DIR+model2,
                               results_file='results/%s_pretrained_' % SAMPLING +model1+'_'+model2+'.csv',
                               hp_file='pred_hp/%s_pretrained_' % SAMPLING +model1+'_'+model2+'.csv',
                               params=params)
            
    if args.model == "allpretrained":
        fit_pretrained(train, valid, test,
                            [KGE_EMBEDDINGS_DIR+m for m in models],
                            [KGE_EMBEDDINGS_DIR+m for m in models],
                            results_file='results/%s_all_pretrained_' % SAMPLING+'.csv',
                            hp_file='pred_hp/%s_all_pretrained_' % SAMPLING +'.csv',
                            params=params)
        
    #Select best models from pretrained and run them using sim embedding.
    if args.model in ['pretrainedensemble','sim']:
        best_models_auc = {}
        for model1 in models:
            for model2 in models:
                df = pd.read_csv('results/%s_pretrained_' % SAMPLING +model1+'_'+model2+'.csv',index_col='metric')
                best_models_auc[(model1,model2)] = df.loc['ba','value']
                
        best_models_auc = sorted(best_models_auc.items(),key=lambda x: x[1], reverse=True)

    if args.model == "sim":
        m,_ = best_models_auc[args.num_models-1]
        model1, model2 = m
        if args.MAX_TRIALS < 1:
            hp_file = "sim_hp/%s_joint_finetune_" % SAMPLING + model1+"_"+model2+".csv.json"
        else:
            hp_file = None
            
        hps = {}
        try:
            with open('pretrained_hp/%s_chemical_kg.json' % model1,'r') as f:
                tmp = json.load(f)
                for k in tmp:
                    hps[k+'1'] = tmp[k]
        except:
            pass 
        try:
            with open('pretrained_hp/%s_taxonomy_kg.json' % model2,'r') as f:
                tmp = json.load(f)
                for k in tmp:
                    hps[k+'2'] = tmp[k]
        except:
            pass
        
        try:
            with open('pred_hp/%s_pretrained_%s_%s.csv' % (SAMPLING,model1,model2),'r') as f:
                tmp = json.load(f)
                hps = {**hps,**tmp}
        except: 
            pass
        
        if hp_file:
            try:
                with open(hp_file, 'r') as f:
                    tmp = json.load(f)
                    hps = {**hps,**tmp}
                    
            except:
                print(model1,model2,'Missing HP file. Using default')

        params['use_pretrained'] = args.USE_PRETRAINED
        if not args.USE_PRETRAINED: SAMPLING+='_non_init'
        
        fit_sim_model(train, valid, test,
                        model1,
                        model2,
                        results_file='results/%s_joint_finetune_' % SAMPLING +model1+'_'+model2+'.csv',
                        embedding_file='sim_embeddings/%s_joint_finetune_' % SAMPLING +model1+'_'+model2,
                        hps = hps,
                        params=params)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run models.')
    parser.add_argument('sampling', metavar='S', type=str, default="none", help='sampling stratergy (none,chemical,species,both)')
    parser.add_argument('model', metavar="M",type=str, default="onehot", help='model to use (onehot,pretrained,sim)')
    parser.add_argument("num_models", metavar="N", type=int, default=1, help='relevant for sim and pretrainedensemble')
    
    parser.add_argument("--NUM_RUNS", type=int, default=1)
    parser.add_argument("--MAX_EPOCHS", type=int, default=10)
    parser.add_argument("--PATIENCE", type=int, default=10)
    
    parser.add_argument("--MAX_TRIALS", type=int, default=1)
    parser.add_argument("--SEARCH_MAX_EPOCHS", type=int, default=10)
    parser.add_argument("--CREATE_DATA", action='store_true')
    parser.add_argument("--USE_PRETRAINED", action='store_true')
    parser.add_argument("--SIMPLE", action='store_true')
    
    
    args = parser.parse_args()
    print(args)
    main(args, vars(args))
    
    
    
