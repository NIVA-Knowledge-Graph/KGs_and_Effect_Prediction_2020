## Use pretrained embeddings to train a auto-sklearn classisifier.

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, Concatenate, BatchNormalization, LayerNormalization
try:
    from tensorflow_addons.callbacks import TimeStopping
except:
    pass
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall, Accuracy, Metric, TrueNegatives,TruePositives,FalseNegatives,FalsePositives
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from models.utils import CVTuner, reset_weights, create_class_weight
from models.utils import base_model, compile_model, l3_reg

from tensorflow_addons.metrics import F1Score, FBetaScore
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split

from models.hptuner import HPTuner

from numpyencoder import NumpyEncoder

from sklearn.metrics import classification_report, f1_score,roc_auc_score,precision_score,recall_score,confusion_matrix

import json
import random
import string
from sklearn.preprocessing import normalize
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tqdm import tqdm
MAX_EPOCHS = 1000
LEARNING_RATE=0.001
VAL = 0.2
SECONDS_PER_TRAIL = 600
MAX_TRIALS = 20
PATIENCE = 5
PARAMS = {
        'SECONDS_PER_TRAIL':SECONDS_PER_TRAIL,
        'MAX_TRIALS':MAX_TRIALS,
        'MAX_EPOCHS':MAX_EPOCHS,
        'PATIENCE':PATIENCE
    }

def build_model(hp):
    
    params = hp.copy()
    
    ci = Input((1,))
    si = Input((1,))
    conci = Input((1,))
    
    c = ci
    s = si
    conc = conci
    
    
    
    if 'one-hot' in params and params['one-hot']:
        c = Embedding(params['num_entities1'],200,trainable=False)(c)
        s = Embedding(params['num_entities2'],200,trainable=False)(s)
    else:
        c = Embedding(params['num_entities1'],
                                            len(params['init_entities1'][0]),
                                            weights=[params['init_entities1']],
                                            trainable=False)(c)
            
        s = Embedding(params['num_entities2'],
                                            len(params['init_entities2'][0]),
                                            weights=[params['init_entities2']],
                                            trainable=False)(s)
        
    c = tf.squeeze(c,axis=1)
    s = tf.squeeze(s,axis=1)
    c = LayerNormalization(axis=-1)(c)
    s = LayerNormalization(axis=-1)(s)
        
    x = base_model(c,s,conc,params)
    
    model = Model(inputs=[ci,si,conci],outputs=[x])
    compile_model(model,params)
    
    return model

def tune(X_train, X_valid, X_test, y_train, y_valid, y_test,
         hypermodel,
         hptuner,
         params,
         results_file,
         hp_file):
    
    bs = len(y_train)
    
    hptuner.add_fixed_hp('learning_rate',LEARNING_RATE)
        
    if hptuner.runs > 1:
        hptuner.add_value_hp('branching_num_layers_chemical',1,4,dtype=int)
        for i in map(str,range(1,4)): hptuner.add_list_hp('branching_units_chemical_'+i,list(map(int,2**np.arange(4,11))))
        hptuner.add_value_hp('branching_num_layers_species',1,4,dtype=int)
        for i in map(str,range(1,4)): hptuner.add_list_hp('branching_units_species_'+i,list(map(int,2**np.arange(4,11))))
        hptuner.add_value_hp('branching_num_layers_conc',1,4,dtype=int)
        for i in map(str,range(1,4)): hptuner.add_list_hp('branching_units_conc_'+i,list(map(int,2**np.arange(2,6))))
        hptuner.add_value_hp('num_layers',1,4,dtype=int)
        for i in map(str,range(1,4)): hptuner.add_list_hp('units_'+i,list(map(int,2**np.arange(4,11))))
    elif params['SIMPLE']:
        hp = hptuner.next_hp_config()
        hptuner.add_result(0.0)
    else:
        try:
            with open(hp_file, 'r') as fp:
                loaded_hps = json.load(fp)
                for k in loaded_hps:
                    hptuner.add_fixed_hp(k,loaded_hps[k])
        except FileNotFoundError:
            print(hp_file,'not found. Using default')
        except json.decoder.JSONDecodeError:
            print(hp_file,'read error. Running trials.')
            hptuner.set_runs(20)
            tune(X_train, X_valid, X_test, y_train, y_valid, y_test,
                        hypermodel,
                        hptuner,
                        params,
                        results_file,
                        hp_file)
            return
    
    with tqdm(total=hptuner.runs) as pbar:
        while hptuner.is_active:
            hp = hptuner.next_hp_config()
            model = hypermodel(hp)
            hist = model.fit(X_train,y_train,
                      validation_data=(X_valid,y_valid),
                      epochs=params['SEARCH_MAX_EPOCHS'],
                      batch_size=bs,
                      class_weight = params['cw'],
                      callbacks=[EarlyStopping('val_loss',mode='min',patience=params['PATIENCE'])],
                      verbose=0)
            
            hptuner.add_result(hist.history['val_auc'][-1])
            pbar.update(1)
            
    best_hps = hptuner.best_config()
    results = []
    out = []
    for _ in tqdm(range(params['NUM_RUNS'])):
        hp = best_hps.copy()
        model = hypermodel(hp) 
        model.fit(X_train, y_train,
                  validation_data=(X_valid, y_valid),
                epochs=params['MAX_EPOCHS'],
                batch_size=bs,
                class_weight = params['cw'],
                callbacks=[EarlyStopping('val_loss',mode='min',patience=params['PATIENCE'],restore_best_weights=True)],
                verbose=0)
        results.append(model.evaluate(X_test,y_test,verbose=0))
        out.append(model.predict(X_test,verbose=0))
        
    var = np.std(np.asarray(results),axis=0)
    results = np.mean(np.asarray(results),axis=0)
    
    df = pd.DataFrame(data={'metric':model.metrics_names,'value':list(results), 'std':list(var)})
    df.to_csv(results_file)
    
    out = np.reshape(np.asarray(out),(params['NUM_RUNS'],-1))
    out = np.concatenate([out,np.reshape(y_test,(1,-1))],axis=0)
    np.save(results_file.replace('/','/predictions_').replace('.csv','.npy'),out)
    
    
    if params['MAX_TRIALS']>10:
        with open(hp_file, 'w') as fp:
            json.dump(best_hps, fp, cls=NumpyEncoder)
        
    tf.keras.backend.clear_session()

class PriorModel:
    def __init__(self):
        pass
    def fit(self,X,y):
        u,uw = np.unique(y,return_counts=True)
        self.lookup = uw/sum(uw)
    
    def predict(self,X):
        return np.asarray([np.argmax(self.lookup) for _ in range(len(X))])

def fit_onehot(train, valid, test, results_file='results.csv',hp_file='hp.json', params=dict()):
    #one hot
    params = {**PARAMS,**params}
    
    X_train, y_train = train
    X_valid, y_valid = valid
    X_test, y_test = test
    
    entities1 = set([x[0] for x in X_train]) | set([x[0] for x in X_test]) | set([x[0] for x in X_valid])
    entities2 = set([x[1] for x in X_train]) | set([x[1] for x in X_test]) | set([x[1] for x in X_valid])
    
    me1 = {k:i for i,k in enumerate(entities1)}
    me2 = {k:i for i,k in enumerate(entities2)}
    
    #f1 = lambda x: tf.keras.utils.to_categorical(me1[x],num_classes=len(me1))
    #f2 = lambda x: tf.keras.utils.to_categorical(me2[x],num_classes=len(me2))
    f1 = lambda x: me1[x]
    f2 = lambda x: me2[x]
    
    X_train,y_train = np.asarray([(f1(a),f2(b),float(x)) for a,b,x in X_train]), np.asarray(y_train)
    X_test,y_test = np.asarray([(f1(a),f2(b),float(x)) for a,b,x in X_test]), np.asarray(y_test)
    X_valid,y_valid = np.asarray([(f1(a),f2(b),float(x)) for a,b,x in X_valid]), np.asarray(y_valid)
    
    X_train = [np.asarray(a) for a in zip(*X_train)]
    X_test = [np.asarray(a) for a in zip(*X_test)]
    X_valid = [np.asarray(a) for a in zip(*X_valid)]
    
    hptuner = HPTuner(runs=params['MAX_TRIALS'],objectiv_direction='max')
    hptuner.add_fixed_hp('one-hot',True)
    hptuner.add_fixed_hp('num_entities1',len(me1))
    hptuner.add_fixed_hp('num_entities2',len(me2))
    
    bm = lambda x: build_model(x)
    tune(X_train, X_valid, X_test, y_train, y_valid, y_test, 
        bm,
        hptuner,
        params,
        results_file,
        hp_file)
    
def fit_pretrained(train, valid, test, model1, model2, results_file='results.csv',hp_file='hp.json',params=dict()):
    #pretrained
    
    params = {**PARAMS,**params}
    
    X_train, y_train = train
    X_valid, y_valid = valid
    X_test, y_test = test
    
    entities1 = set([x[0] for x in X_train]) | set([x[0] for x in X_test]) | set([x[0] for x in X_valid])
    entities2 = set([x[1] for x in X_train]) | set([x[1] for x in X_test]) | set([x[1] for x in X_valid])
    
    me1 = {k:i for i,k in enumerate(entities1)}
    me2 = {k:i for i,k in enumerate(entities2)}
    
    f1 = lambda x: me1[x]
    f2 = lambda x: me2[x]
    
    X_train,y_train = np.asarray([(f1(a),f2(b),float(x)) for a,b,x in X_train]), np.asarray(y_train)
    X_test,y_test = np.asarray([(f1(a),f2(b),float(x)) for a,b,x in X_test]), np.asarray(y_test)
    X_valid,y_valid = np.asarray([(f1(a),f2(b),float(x)) for a,b,x in X_valid]), np.asarray(y_valid)
    
    X_train = [np.asarray(a) for a in zip(*X_train)]
    X_test = [np.asarray(a) for a in zip(*X_test)]
    X_valid = [np.asarray(a) for a in zip(*X_valid)]
    
    hptuner = HPTuner(runs=params['MAX_TRIALS'],objectiv_direction='max')
    hptuner.add_fixed_hp('num_entities1',len(me1))
    hptuner.add_fixed_hp('num_entities2',len(me2))
    
    def f(f1,f2,items):
        #This is for sannity. Entites should be in the same order.
        ids = dict(np.load(f2))
        out = np.load(f1)
        return np.asarray([out[int(ids[k])] for k in items])
    
    hptuner.add_fixed_hp('init_entities1',f(model1+'_chemical_entity_embeddings.npy',model1+'_chemical_entity_ids.npy',entities1))
    hptuner.add_fixed_hp('init_entities2',f(model2+'_taxonomy_entity_embeddings.npy',model2+'_taxonomy_entity_ids.npy',entities2))
        
    bm = lambda x: build_model(x)
    tune(X_train, X_valid, X_test, y_train, y_valid, y_test, 
        bm,
        hptuner,
        params,
        results_file,
        hp_file)
    
        
def load_hier_embeddings(f,entities):
    X = np.diag(np.zeros(len(entities)))
    df = pd.read_csv(f)
    cols = list(df.columns[1:-1])
    for c1 in cols:
        try:
            i = entities.index(c1)
        except:
            pass
        for c2 in cols:
            try:
                j = entities.index(c2)
                X[i,j] = min(1,df.iloc[cols.index(c1)+1,cols.index(c2)+1])
            except:
                pass
    return X

def create_hier_data(X_train, X_valid, X_test, y_train, y_valid, y_test, chemical_embedding_files, taxonomy_embedding_files):
    entities11, entities21, _ = map(set,zip(*X_train))
    entities12, entities22, _ = map(set,zip(*X_test))
    entities1 = entities11 | entities12
    entities2 = entities21 | entities22
    
    entities1 = list(entities1)
    entities2 = list(entities2)
    
    X1 = []
    for f in chemical_embedding_files:
        X1.append(load_hier_embeddings(f,entities1))
    X1 = np.concatenate(X1,axis=1)
    
    X2 = []
    for f in taxonomy_embedding_files:
        X2.append(load_hier_embeddings(f,entities2))
    X2 = np.concatenate(X2,axis=1)
    
    me1 = {k:i for i,k in enumerate(entities1)}
    me2 = {k:i for i,k in enumerate(entities2)}
    rme1 = {i:k for k,i in me1.items()}
    rme2 = {i:k for k,i in me2.items()}
    
    X_train = np.asarray([[X1[me1[a],:],X2[me2[b],:],[float(c)]] for a,b,c in X_train])
    X_test = np.asarray([[X1[me1[a],:],X2[me2[b],:],[float(c)]] for a,b,c in X_test])
    X_valid = np.asarray([[X1[me1[a],:],X2[me2[b],:],[float(c)]] for a,b,c in X_valid])
    
    
    X_train = [np.asarray(a) for a in zip(*X_train)]
    X_test = [np.asarray(a) for a in zip(*X_test)]
    X_valid = [np.asarray(a) for a in zip(*X_valid)]
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def fit_hier_embeddings(train, valid, test, chemical_embedding_files, taxonomy_embedding_files,results_file='results.csv',hp_file='hp.json',params=dict()):
    
    params = {**PARAMS,**params}
    X_train, y_train = train
    X_valid, y_valid = valid
    X_test, y_test = test

    X_train, X_valid, X_test, y_train, y_valid, y_test = create_hier_data(X_train, X_valid, X_test, y_train, y_valid, y_test,chemical_embedding_files, taxonomy_embedding_files)
    
    y_train = np.asarray(y_train).reshape((-1,1))
    y_test = np.asarray(y_test).reshape((-1,1))
    y_valid = np.asarray(y_valid).reshape((-1,1))
    
    hp = HyperParameters()
    
    bm = lambda x: build_model(x,len(X_train[0][0]),len(X_train[1][0]))
    tune(X_train, X_valid, X_test, y_train, y_valid, y_test, 
         bm,
         hp,
         params,
         results_file,
         hp_file)
    
def fit_hier_kg_combination(train, valid, test, model1, model2, chemical_embedding_files, taxonomy_embedding_files, k=1, results_file='results.csv', hp_file='hp.json', params=dict()):
    
    params = {**PARAMS,**params}
    X_train, y_train = train
    X_valid, y_valid = valid
    X_test, y_test = test
    
    X_train_hier, X_valid_hier, X_test_hier, y_train_hier, y_valid_hier, y_test_hier = create_hier_data(X_train, X_valid, X_test, y_train, y_valid, y_test,
                                                                            chemical_embedding_files, 
                                                                            taxonomy_embedding_files)
    
    X_train_e, X_valid_e, X_test_e, y_train_e, y_valid_e, y_test_e = load_pretrained_kg_embeddings(X_train, X_valid, X_test, y_train, y_valid, y_test,
                                                                             f1=model1,
                                                                             f2=model2)
    
    assert np.array_equal(y_train_e,y_train_hier) and np.array_equal(y_test_e,y_test_hier)
    
    y_train = np.asarray(y_train).reshape((-1,1))
    y_test = np.asarray(y_test).reshape((-1,1))
    y_valid = np.asarray(y_valid).reshape((-1,1))
    
    
    X_train = [np.concatenate([x1,x2],axis=1) for x1,x2 in zip(X_train_e,X_train_hier)]
    X_test = [np.concatenate([x1,x2],axis=1) for x1,x2 in zip(X_test_e,X_test_hier)]
    X_valid = [np.concatenate([x1,x2],axis=1) for x1,x2 in zip(X_valid_e,X_valid_hier)]
    
    X_train[-1] = np.mean(X_train[-1],axis=1)
    X_test[-1] = np.mean(X_test[-1],axis=1)
    X_valid[-1] = np.mean(X_valid[-1],axis=1)
    
    hp = HyperParameters()
    
    bm = lambda x: build_model(x,len(X_train[0][0]),len(X_train[1][0]))
    
    tune(X_train, X_valid, X_test, y_train, y_valid, y_test, 
         bm,
         hp,
         params,
         results_file,
         hp_file)
    
    
    
    
    
