## all in model

from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Dense, Input, Dropout, BatchNormalization, Embedding, LayerNormalization
import tensorflow as tf
import kerastuner as kt
from tqdm import tqdm

from tensorflow.keras.optimizers import Adam, RMSprop

from KGEkeras import DistMult, HolE, TransE, HAKE, ConvE, ComplEx, ConvR, RotatE, pRotatE, ConvKB
from KGEkeras import loss_function_lookup

import numpy as np
import pandas as pd

from random import choice, choices

from tensorflow.keras.metrics import AUC, Precision, Recall, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from models.utils import f1, f2, CVTuner, reset_weights, load_data, create_class_weight
from models.utils import generate_negative, oversample_data, undersample_data
from models.hptuner import HPTuner

from tensorflow.keras.backend import log
import json 
from collections import defaultdict

from tensorflow_addons.metrics import F1Score, FBetaScore
from tensorflow.keras.models import load_model

import random
import string

tf.keras.backend.clear_session()
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers.experimental.preprocessing import Normalization

from sklearn.metrics import classification_report, f1_score,roc_auc_score,precision_score,recall_score,confusion_matrix

from models.utils import base_model, compile_model

VAL=0.2
LEARNING_RATE=0.001
SECONDS_PER_TRAIL = 600
MAX_TRIALS = 20
MAX_EPOCHS = 1000
PATIENCE = 5
LR_REDUCTION = 100
PARAMS = {
        'SECONDS_PER_TRAIL':SECONDS_PER_TRAIL,
        'MAX_TRIALS':MAX_TRIALS,
        'MAX_EPOCHS':MAX_EPOCHS,
        'PATIENCE':PATIENCE
    }

models = {
            'DistMult':DistMult,
            'TransE':TransE,
            'HolE':HolE,
            'ComplEx':ComplEx,
            'RotatE':RotatE,
            'pRotatE':pRotatE,
            'HAKE':HAKE,
            'ConvE':ConvE,
            'ConvR':ConvR,
            'ConvKB':ConvKB,
         }

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, kg1, kg2, x, y, ns1=10, ns2=10, batch_size=32, shuffle=True):
        self.batch_size = min(batch_size,len(y))
        self.kg1 = kg1 
        self.kg2 = kg2 
        self.ns1 = ns1
        self.ns2 = ns2
        self.shuffle = shuffle
        self.x = x
        self.y = y
        self.num_e1 = len(set([s for s,_,_ in kg1]) | set([o for _,_,o in kg1]))
        self.num_e2 = len(set([s for s,_,_ in kg2]) | set([o for _,_,o in kg2]))
        self.indices = list(range(len(x)))
        self.on_epoch_end()

    def __len__(self):
        return len(self.y) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        
        X = np.asarray([self.x[i] for i in batch])
        y = np.asarray([self.y[i] for i in batch])
        
        tmp_kg1 = choices(self.kg1, k=len(batch)//self.ns1)
        tmp_kg2 = choices(self.kg2, k=len(batch)//self.ns2)
        negative_kg1 = generate_negative(tmp_kg1,N=self.num_e1,negative=self.ns1)
        negative_kg2 = generate_negative(tmp_kg2,N=self.num_e2,negative=self.ns2)
        Xtr, ytr = oversample_data(kgs=[tmp_kg1,negative_kg1,tmp_kg2,negative_kg2],x=X,y=y)
        
        return Xtr, ytr

def build_model(hp,norm_params=None):
    
    params = hp.copy()
    
    params1 = {k.replace('1',''):params[k] for k in params if not '2' in k}
    params2 = {k.replace('2',''):params[k] for k in params if not '1' in k}
    
    params1['e_dim'],params1['r_dim'] = params1['dim'],params1['dim']
    params2['e_dim'],params2['r_dim'] = params2['dim'],params2['dim']
    params1['name'] = 'chemical_embedding_model'
    params2['name'] = 'species_embedding_model'
    
    m1 = models[params1['embedding_model']]
    m2 = models[params2['embedding_model']]
    
    embedding_model1 = m1(**params1)
    embedding_model2 = m2(**params2)
    
    triple1 = Input((3,))
    triple2 = Input((3,))
    ftriple1 = Input((3,))
    ftriple2 = Input((3,))
    
    ci = Input((1,))
    si = Input((1,))
    conc = Input((1,))
    inputs = [triple1, ftriple1, triple2, ftriple2, ci, si, conc]
        
    score1 = embedding_model1(triple1)
    fscore1 = embedding_model1(ftriple1)
    loss_function1 = loss_function_lookup(params1['loss_function'])
    loss1 = loss_function1(score1,fscore1,params1['margin'] or 1, 1)
    
    score2 = embedding_model2(triple2)
    fscore2 = embedding_model2(ftriple2)
    loss_function2 = loss_function_lookup(params2['loss_function'])
    loss2 = loss_function2(score2,fscore2,params2['margin'] or 1, 1)
    
    c = embedding_model1.entity_embedding(ci)
    s = embedding_model2.entity_embedding(si)
    c = tf.squeeze(c,axis=1)
    s = tf.squeeze(s,axis=1)
    
    c = LayerNormalization(axis=-1)(c)
    s = LayerNormalization(axis=-1)(s)
    
    x = base_model(c,s,conc,params)
    
    model = Model(inputs=inputs, outputs=[x])
    model.add_loss(params1['loss_weight']*loss1 + params2['loss_weight']*loss2)
        
    if params['use_pretrained']:
        for layer in embedding_model1.layers: 
            if isinstance(layer,Embedding):
                layer.trainable=False
        for layer in embedding_model2.layers: 
            if isinstance(layer,Embedding):
                layer.trainable=False
        
    compile_model(model,hp)
    
    return model

        
def fit_sim_model(train, valid, test, model1, model2, results_file='results.csv', embedding_file='sim_embeddings', hps = dict(), params=dict()):
    
    X_train, y_train = train
    X_valid, y_valid = valid
    X_test, y_test = test
    
    params = {**PARAMS,**params}
        
    pdf = [pd.read_csv('./data/chemicals.csv'),pd.read_csv('./data/chemicals_extended.csv'),pd.read_csv('./data/chemicals_similarity.csv')]
    kg1 = pd.concat(pdf)
    kg2 = pd.read_csv('./data/taxonomy.csv')
    
    kg1 = list(zip(kg1['subject'], kg1['predicate'], kg1['object']))
    kg2 = list(zip(kg2['subject'], kg2['predicate'], kg2['object']))

    entities1 = set([s for s, p, o in kg1]) | set([o for s, p, o in kg1])
    relations1 = set([p for s, p, o in kg1])
    entities2 = set([s for s, p, o in kg2]) | set([o for s, p, o in kg2])
    relations2 = set([p for s, p, o in kg2])
    
    me1 = {k:i for i,k in enumerate(entities1)}
    me2 = {k:i for i,k in enumerate(entities2)}
    mr1 = {k:i for i,k in enumerate(relations1)}
    mr2 = {k:i for i,k in enumerate(relations2)}
    kg1 = [(me1[s],mr1[p],me1[o]) for s,p,o in kg1]
    kg2 = [(me2[s],mr2[p],me2[o]) for s,p,o in kg2]
    
    hptuner = HPTuner(runs=params['MAX_TRIALS'],objectiv_direction='max')
    
    if params['use_pretrained']:
        def f(f1,f2,items):
            #This is for sannity. Entites should be in the same order.
            ids = dict(np.load(f2))
            out = np.load(f1)
            return np.asarray([out[int(ids[k])] for k in items])
        
        d = 'results/pretrained_embeddings/'
        hptuner.add_fixed_hp('init_entities1',f(d+model1+'_chemical_entity_embeddings.npy',d+model1+'_chemical_entity_ids.npy',entities1))
        hptuner.add_fixed_hp('init_entities2',f(d+model2+'_taxonomy_entity_embeddings.npy',d+model2+'_taxonomy_entity_ids.npy',entities2))
        hptuner.add_fixed_hp('init_relations1',f(d+model1+'_chemical_relational_embeddings.npy',d+model1+'_chemical_relation_ids.npy',relations1))
        hptuner.add_fixed_hp('init_relations2',f(d+model2+'_taxonomy_relational_embeddings.npy',d+model2+'_taxonomy_relation_ids.npy',relations2))

    X_train, y_train = np.asarray([(me1[a],me2[b],float(x)) for a,b,x in X_train if a in entities1 and b in entities2]), np.asarray([float(x) for x,a in zip(y_train,X_train) if a[0] in entities1 and a[1] in entities2])

    X_test, y_test = np.asarray([(me1[a],me2[b],float(x)) for a,b,x in X_test if a in entities1 and b in entities2]), np.asarray([float(x) for x,a in zip(y_test, X_test) if a[0] in entities1 and a[1] in entities2])
    
    X_valid, y_valid = np.asarray([(me1[a],me2[b],float(x)) for a,b,x in X_valid if a in entities1 and b in entities2]), np.asarray([float(x) for x,a in zip(y_valid, X_valid) if a[0] in entities1 and a[1] in entities2])
        
    scores = []    
    k_best_predictions = []
    
    kg_lengths = list(map(len,[kg1,kg2]))
    output_lengths = len(X_train)
    
    hptuner.add_fixed_hp('num_entities1',len(entities1))
    hptuner.add_fixed_hp('num_entities2',len(entities2))
    hptuner.add_fixed_hp('num_relations1',len(relations1))
    hptuner.add_fixed_hp('num_relations2',len(relations2))

    hptuner.add_fixed_hp('embedding_model1',model1)
    hptuner.add_fixed_hp('embedding_model2',model2)
    hptuner.add_fixed_hp('use_pretrained',params['use_pretrained'])
    hptuner.add_fixed_hp('learning_rate',LEARNING_RATE)
    
    bs = 2048

    hptuner.add_fixed_hp('loss_weight1',len(y_train)/len(kg1))
    hptuner.add_fixed_hp('loss_weight2',len(y_train)/len(kg2))
    
    for k in hps:
        hptuner.add_fixed_hp(k,hps[k])
    
    if params['MAX_TRIALS'] > 0:
        hptuner.add_value_hp('loss_weight1',-2,2,sampling='log',default=0)
        hptuner.add_value_hp('loss_weight2',-2,2,sampling='log',default=0)
    
    hp = hptuner.get_default_config()
    tr_gen = DataGenerator(kg1,
                                kg2,
                                X_train,
                                y_train,
                                hp['negative_samples1'],
                                hp['negative_samples2'],
                                batch_size=bs)
        
    val_gen = DataGenerator(kg1,
                                kg2,
                                X_valid,
                                y_valid,
                                hp['negative_samples1'],
                                hp['negative_samples2'],
                                batch_size=bs)
    te_gen = DataGenerator(kg1,
                                kg2,
                                X_test,
                                y_test,
                                1,
                                1,
                                batch_size=bs,
                                shuffle=False)
        
    with tqdm(total=hptuner.runs) as pbar:
        while hptuner.is_active:
            hp = hptuner.next_hp_config()
            model = build_model(hp)
            hist = model.fit(tr_gen,
                        validation_data=val_gen,
                        epochs=params['SEARCH_MAX_EPOCHS'],
                        verbose=0,
                        class_weight=params['cw'],
                        callbacks=[EarlyStopping(monitor="val_bce",
                                                mode='min',
                                                patience=params['PATIENCE']),TerminateOnNaN()])
                
            model.trainable=True
            hp['learning_rate'] /= LR_REDUCTION
            compile_model(model,hp)
            hist = model.fit(tr_gen,
                  validation_data=val_gen,
                  epochs=params['SEARCH_MAX_EPOCHS'],
                  verbose=0, 
                  class_weight=params['cw'],
                  callbacks=[EarlyStopping(monitor="val_bce",                                                                                                     
                                          mode='min',
                                          patience=params['PATIENCE'],
                                          restore_best_weights=True)])
            
            hptuner.add_result(hist.history['val_auc'][-1])
            tf.keras.backend.clear_session()
            pbar.update(1)

    if params['MAX_TRIALS'] > 0:
        hp = hptuner.best_config().copy()
        out = {}
        for k in hp:
            if 'loss_weight' in k:
                out[k] = hp[k]
        with open('./sim_hp/' + results_file.split('/')[-1]+".json" , 'w') as fp:
            json.dump(out, fp)

    results = []
    out = []
    best_hps = hptuner.best_config() if params['MAX_TRIALS'] > 0 else hptuner.get_default_config()
    for k in tqdm(range(params['NUM_RUNS']), desc='RUNS'): 
        hp = best_hps.copy()
        model = build_model(hp)
        
        model.fit(tr_gen,
                  validation_data=val_gen,
                  epochs=params['MAX_EPOCHS'],
                  verbose=0, 
                  class_weight=params['cw'],
                  callbacks=[EarlyStopping(monitor="val_bce",                                                                                                     
                                          mode='min',
                                          patience=params['PATIENCE'])])
        #FINETUNE
        model.trainable=True
        hp['learning_rate'] /= LR_REDUCTION
        compile_model(model,hp)
        model.fit(tr_gen,
                  validation_data=val_gen,
                  epochs=params['MAX_EPOCHS'],
                  verbose=0, 
                  class_weight=params['cw'],
                  callbacks=[EarlyStopping(monitor="val_bce",                                                                                                     
                                          mode='min',
                                          patience=params['PATIENCE'],
                                          restore_best_weights=True)])
        #TEST
        results.append(model.evaluate(te_gen,verbose=0))
        out.append(model.predict(te_gen,verbose=0))
    
    if params['NUM_RUNS'] > 0:
        var = np.std(np.asarray(results),axis=0)
        results = np.mean(np.asarray(results),axis=0)
        
        df = pd.DataFrame(data={'metric':model.metrics_names,'value':list(results), 'std':list(var)})
        df.to_csv(results_file)
        
        out = np.reshape(np.asarray(out),(params['NUM_RUNS'],-1))
        out = np.concatenate([out,np.reshape(y_test,(1,-1))],axis=0)
        np.save(results_file.replace('/','/predictions_').replace('.csv','.npy'),out)
        
        for l in model.layers:
            if isinstance(l,models[hp['embedding_model1']]):
                m1 = l.name
            if isinstance(l,models[hp['embedding_model2']]):
                m2 = l.name
            
        embeddings1 = model.get_layer(m1).entity_embedding.get_weights()[0]
        embeddings2 = model.get_layer(m2).entity_embedding.get_weights()[0]
        
        for s, mn, W, ent in zip(['chemical','taxonomy'],[hp['embedding_model1'],hp['embedding_model2']],[embeddings1,embeddings2],[entities1,entities2]):
            f = './results/sim_embeddings/%s' % results_file.split('/')[-1][:-4]
            np.save(f+'_%s_embeddings.npy' % s, W)
            np.save(f+'_%s_ids.npy' % mn,np.asarray(list(zip(ent,range(len(ent))))))





























