
import sys
import os

from itertools import product
from KGEkeras import DistMult, HolE, TransE, HAKE, ConvE, ComplEx, ConvR, RotatE, pRotatE, ConvKB, CosinE

from kerastuner import RandomSearch, HyperParameters, Objective, Hyperband, BayesianOptimization

from random import choice
from collections import defaultdict

from tensorflow.keras.losses import binary_crossentropy,hinge,mean_squared_error
from tensorflow.keras import Input
from tensorflow.keras import Model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback, TerminateOnNaN, ReduceLROnPlateau
from sklearn.metrics.cluster import completeness_score

from tensorflow.keras.optimizers import Adam
import json

import tensorflow as tf

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from KGEkeras import loss_function_lookup
from utils import generate_negative, oversample_data, load_data
from tqdm import tqdm

import string
import random

from random import choices
from hptuner import HPTuner
import pickle

try:
    from tensorflow_addons.callbacks import TimeStopping
except:
    pass

SECONDS_PER_TRAIL = 600
SECONDS_TO_TERMINATE = 3600
SEARCH_MAX_EPOCHS = 10
MAX_EPOCHS = 200
MIN_EPOCHS = 50
MAX_TRIALS = 20
PATIENCE = 10

EPSILON = 10e-7

models = {
            'DistMult':DistMult,
            'TransE':TransE,
            'HolE':HolE,
            'ComplEx':ComplEx,
            'HAKE':HAKE,
            'pRotatE':pRotatE,
            'RotatE':RotatE,
            'ConvE':ConvE,
            'ConvKB':ConvKB,
         }

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, kg, ns=10, batch_size=32, shuffle=True):
        self.batch_size = min(batch_size,len(kg))
        self.kg = kg
        self.ns = ns
        self.num_e = len(set([s for s,_,_ in kg])|set([o for _,_,o in kg]))
        self.shuffle = shuffle
        self.indices = list(range(len(kg)))
        
        self.on_epoch_end()

    def __len__(self):
        return len(self.kg) // self.batch_size

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
        tmp_kg = np.asarray([self.kg[i] for i in batch])
        
        negative_kg = generate_negative(tmp_kg,N=self.num_e,negative=self.ns)
        X = oversample_data(kgs=[tmp_kg,negative_kg])
    
        return X, None 

def build_model(hp):
    
    params = hp.copy()
    params['e_dim'] = params['dim']
    params['r_dim'] = params['dim']
    params['name'] = 'embedding_model'
    
    embedding_model = models[params['embedding_model']]
    embedding_model = embedding_model(**params)
    triple = Input((3,))
    ftriple = Input((3,))
    
    inputs = [triple, ftriple]
    
    score = embedding_model(triple)
    fscore = embedding_model(ftriple)
    
    loss_function = loss_function_lookup(params['loss_function'])
    loss = loss_function(score,fscore,params['margin'] or 1, 1)
    
    model = Model(inputs=inputs, outputs=loss)
    model.add_loss(loss)
    
    model.compile(optimizer=Adam(learning_rate=ExponentialDecay(params['learning_rate'],decay_steps=100000,decay_rate=0.96)),
                  loss=None)
    
    return model


def optimize_model(model, kg, name, hp_file=None):
   
    bs = 256
    
    kg = np.asarray(kg)
    
    model_name = model
    
    N = len(set([s for s,_,_ in kg]) | set([o for _,_,o in kg]))
    M = len(set([p for _,p,_ in kg]))
        
    hptuner = HPTuner(min_runs=MAX_TRIALS, objectiv_direction='min')
    hptuner.add_value_hp('gamma',0,21)
    hptuner.add_value_hp('dim',100,401,dtype=int,step=4)
    hptuner.add_value_hp('negative_samples',10,101,dtype=int,step=10)
    hptuner.add_value_hp('margin',1,11,dtype=int)
    hptuner.add_list_hp('loss_function',['pairwize_hinge','pairwize_logistic','pointwize_hinge','pointwize_logistic'])
    
    hptuner.add_fixed_hp('embedding_model',model)
    hptuner.add_fixed_hp('dp',0.2)
    hptuner.add_fixed_hp('hidden_dp',0.2)
    hptuner.add_fixed_hp('num_entities',N)
    hptuner.add_fixed_hp('num_relations',M)
    
    if hp_file:
        with open(hp_file,'r') as f:
            for k,i in json.load(f).items():
                hptuner.add_fixed_hp(k,i)
        hptuner.add_fixed_hp('num_entities',N)
        hptuner.add_fixed_hp('num_relations',M)
    
    hptuner.add_fixed_hp('learning_rate',0.001)
    hptuner.add_fixed_hp('regularization',0.001)
    
    if hp_file:
        hptuner.next_hp_config()
        hptuner.add_result(0.0)
    
    with tqdm(total=MAX_TRIALS, desc='Trials') as pbar:
        while hptuner.is_active and not hp_file:
            hp = hptuner.next_hp_config()
            model = build_model(hp)
            tr_gen = DataGenerator(kg, batch_size=bs, shuffle=True, ns=hp['negative_samples'])
            hist = model.fit(tr_gen,epochs=SEARCH_MAX_EPOCHS,verbose=2, callbacks=[EarlyStopping('loss'),TerminateOnNaN()])
            score = hist.history['loss'][-1]/hist.history['loss'][0]
            hptuner.add_result(score)
            tf.keras.backend.clear_session()
            pbar.update(1)
        
    hp = hptuner.best_config()
    
    with open('./pretrained_hp/%s%s_kg.json' % (model_name,name), 'w') as fp:
        json.dump(hp, fp)
    
    model = build_model(hp)
    tr_gen = DataGenerator(kg, batch_size=bs, shuffle=True, ns=hp['negative_samples'])
    hist = model.fit(tr_gen,epochs=MAX_EPOCHS, verbose=2, callbacks=[EarlyStopping('loss',patience=PATIENCE), TerminateOnNaN()])
    if np.isnan(hist.history['loss'][-1]):
        print(model_name,'nan loss.')
        return optimize_model(model_name,kg,name,None)
    
    for l in model.layers:
        if isinstance(l,models[model_name]):
            m = l.name
    
    return model, model.get_layer(m).entity_embedding.get_weights()[0], model.get_layer(m).relational_embedding.get_weights()[0]
                

def main():
    d = './results/pretrained_embeddings/'
    
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
    
    best_models = {}
    for model,i in models.items():
        for ent,rel,kg,name in zip([entities1,entities2],[relations1,relations2],[kg1,kg2],['_chemical','_taxonomy']):
            print(model,name)
            hp_file = 'pretrained_hp/%s%s_kg.json' % (model,name)
            m,W1,W2 = optimize_model(model,kg,name,hp_file)
            m.save_weights('pretrained_models/'+model+name+'/model')
            f = d+model+name
            np.save(f+'_entity_embeddings.npy', W1)
            np.save(f+'_entity_ids.npy',np.asarray(list(zip(ent,range(len(ent))))))
            np.save(f+'_relational_embeddings.npy', W2)
            np.save(f+'_relation_ids.npy',np.asarray(list(zip(rel,range(len(rel))))))
            
            tf.keras.backend.clear_session()
        
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
