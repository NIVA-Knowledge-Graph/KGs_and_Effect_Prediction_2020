#utils 

import matplotlib.pyplot as plt
import numpy as np

import keras 
from tensorflow.keras.losses import cosine_similarity
import tensorflow.keras.backend as K
from random import choices, choice
import pandas as pd

import kerastuner
import numpy as np
from sklearn import model_selection
from itertools import product
import tensorflow as tf
import math
from random import shuffle

from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Concatenate
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import TrueNegatives,TruePositives,FalseNegatives,FalsePositives,AUC,BinaryCrossentropy

def sensitivity(tp,fn,tn,fp):
    return tp/(tp+fn)

def specificity(tp,fn,tn,fp):
    return tn/(tn+fp)

def balanced_accuracy(tp,fn,tn,fp):
    return (sensitivity(tp,fn,tn,fp)+specificity(tp,fn,tn,fp))/2
 
def youden(tp,fn,tn,fp):
    return sensitivity(tp,fn,tn,fp) + specificity(tp,fn,tn,fp) - 1

def dor(tp,fn,tn,fp):
    return (tp*tn)/(fp*fn)

def matthews_correlation_coefficient(tp,fn,tn,fp):
    return (tp*tn-fp*fn)/(tf.math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))

class ConfusionMatrixDerivedMetric(tf.keras.metrics.Metric):
    def __init__(self, f, name, **kwargs):
        super(ConfusionMatrixDerivedMetric, self).__init__(name=name, **kwargs)
        self.tp = TruePositives()
        self.tn = TrueNegatives()
        self.fn = FalseNegatives()
        self.fp = FalsePositives()
        self.f = f
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp.update_state(y_true,y_pred, sample_weight=sample_weight)
        self.tn.update_state(y_true,y_pred, sample_weight=sample_weight)
        self.fp.update_state(y_true,y_pred, sample_weight=sample_weight)
        self.fn.update_state(y_true,y_pred, sample_weight=sample_weight)
    
    def reset_states(self):
        self.tp.reset_states()
        self.tn.reset_states()
        self.fp.reset_states()
        self.fn.reset_states()
        
    def result(self):
        return tf.reduce_sum(self.f(self.tp.result(),self.fn.result(),self.tn.result(),self.fp.result()))

class DiagnosticOddsRatio(ConfusionMatrixDerivedMetric):
    def __init__(self, name="diagnostic_odds_ratio", **kwargs):
        super(DiagnosticOddsRatio, self).__init__(f=dor, name=name, **kwargs)
        
class YoudenIndex(ConfusionMatrixDerivedMetric):
    def __init__(self, name="youden_index", **kwargs):
        super(YoudenIndex, self).__init__(f=youden, name=name, **kwargs)
                                 
class BalancedAccuracy(ConfusionMatrixDerivedMetric):
    def __init__(self, name="balanced_accuracy", **kwargs):
        super(BalancedAccuracy, self).__init__(f=balanced_accuracy,name=name, **kwargs)
 
class MatthewsCorrelationCoefficient(ConfusionMatrixDerivedMetric):
    def __init__(self, name="Matthews correlation coefficient", **kwargs):
        super(MatthewsCorrelationCoefficient, self).__init__(f=matthews_correlation_coefficient,name=name, **kwargs)
        
class Sensitivity(ConfusionMatrixDerivedMetric):
    def __init__(self, name="Matthews correlation coefficient", **kwargs):
        super(Sensitivity, self).__init__(f=sensitivity,name=name, **kwargs)
        
class Specificity(ConfusionMatrixDerivedMetric):
    def __init__(self, name="Matthews correlation coefficient", **kwargs):
        super(Specificity, self).__init__(f=specificity,name=name, **kwargs)

def l3_reg(weight_matrix, w = 0.01):
    return w * tf.norm(weight_matrix,ord=3)**3

def compile_model(model,hp):
    optimizer = RMSprop(hp['learning_rate'])
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[
                    BinaryCrossentropy(name='bce'),
                    DiagnosticOddsRatio(name='dor'),
                    YoudenIndex(name='yi'),
                    MatthewsCorrelationCoefficient(name='mcc'),
                    BalancedAccuracy(name='ba'),
                    Sensitivity(name='sensitivity'),
                    Specificity(name='specificity'),
                    AUC(num_thresholds=10000,summation_method='interpolation',name='auc')])

def base_model(c,s,conc,params={}):
    for i,layer_num in enumerate(range(params.pop('branching_num_layers_chemical',0))):
        c = Dense(params.pop('branching_units_chemical_'+str(i+1),128),activation='relu')(c)
        c = LayerNormalization(axis=-1)(c)
        c = Dropout(0.2)(c)
    
    for i,layer_num in enumerate(range(params.pop('branching_num_layers_species',0))):
        s = Dense(params.pop('branching_units_species_'+str(i+1),128), activation='relu')(s)
        s = LayerNormalization(axis=-1)(s)
        s = Dropout(0.2)(s)
    
    for i,layer_num in enumerate(range(params.pop('branching_num_layers_conc',0))):
        conc = Dense(params.pop('branching_units_conc_'+str(i+1),4), activation='relu')(conc)
        conc = LayerNormalization(axis=-1)(conc)
        conc = Dropout(0.2)(conc)
    
    x = Concatenate(axis=-1)([c,s,conc])
    
    for i,layer_num in enumerate(range(params.pop('num_layers',1))):
        x = Dense(params.pop('units_'+str(i+1),128),activation='relu')(x)
        x = LayerNormalization(axis=-1)(x)
        x = Dropout(0.2)(x)
        
    x = Dense(1,activation='sigmoid',name='output')(x)
    return x


def oversample_data(kgs,x=None,y=None,testing=False):
    if testing:
        kgs = [list(kg)[:len(y)] for kg in kgs]
    else:
        kgs = [list(kg) for kg in kgs]
        
    if y is not None:
        m = max(max(map(len,kgs)),len(y))
    else:
        m = max(map(len,kgs))
    
    out = []
    for kg in kgs:
        out.append(choices(kg, k=m))
    
    if x is not None and y is not None:
        k = np.ceil(m/len(y))
        y = np.repeat(y,k,axis=0)[:m]
        x = np.repeat(x,k,axis=0)[:m,:]
        for s in np.split(x,3,axis=1):
            out.append(s.reshape((-1,)))
        return [np.squeeze(np.asarray(o)) for o in out], np.asarray(y)
    
    else:
        return [np.squeeze(np.asarray(o)) for o in out]

def undersample_data(kgs,x=None,y=None, testing=False):
    if testing:
        kgs = [list(kg)[:len(y)] for kg in kgs]
    else:
        kgs = [list(kg) for kg in kgs]
    
    if y is not None:
        m = min(min(map(len,kgs)),len(y))
    else:
        m = min(map(len,kgs))
    
    out = []
    for kg in kgs:
        out.append(choices(kg, k=m))
    
    if x is not None and y is not None:
        k = np.ceil(m/len(y))
        y = np.repeat(y,k,axis=0)[:m]
        x = np.repeat(x,k,axis=0)[:m,:]
        for s in np.split(x,3,axis=1):
            out.append(s.reshape((-1,)))
        return [np.squeeze(np.asarray(o)) for o in out], np.asarray(y)
    
    else:
        return [np.squeeze(np.asarray(o)) for o in out]


class CVTuner(kerastuner.engine.tuner.Tuner):
    def run_trial(self, trial, x, y, batch_size=32, epochs=1, callbacks=None, kfolds=5, class_weight=None):
        cv = model_selection.KFold(kfolds,shuffle=True,random_state=42)
        val_losses = []
        
        k = len(x) - 3
        m = max(map(len,x)) + (batch_size - max(map(len,x)) % batch_size)
        
        for train_indices, test_indices in cv.split(y):
            x_train, x_test = [a[train_indices] for a in x[k:]], [a[test_indices] for a in x[k:]]
            y_train, y_test = y[train_indices], y[test_indices]
            
            if k != 0:
                x_train, x_test = np.asarray(x_train).T, np.asarray(x_test).T
                x_train, y_train = prep_data_v2(x[0],x[1],x_train,y_train,max_length=m)
                x_test, y_test = prep_data_v2(x[0],x[1],x_test,y_test,max_length=m)
            
            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0, class_weight=class_weight)
            val_losses.append(model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size))
        m = np.mean(val_losses,axis=0)
        d = dict([('val_'+mn,vl) for mn,vl in zip(model.metrics_names,m)])
        self.oracle.update_trial(trial.trial_id, d)
        self.save_model(trial.trial_id, model)

def reset_weights(model):
    for layer in model.layers: 
        if isinstance(layer, tf.keras.Model):
            reset_weights(layer)
            continue
    for k, initializer in layer.__dict__.items():
        if "initializer" not in k:
            continue
        # find the corresponding variable
        var = getattr(layer, k.replace("_initializer", ""))
        var.assign(initializer(var.shape, var.dtype))
        
def create_class_weight(labels_dict,mu=0.15):
    total = np.sum([v for k,v in labels_dict.items()])
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/labels_dict[key])
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

def train_valid_test_split(X,Y,valid_size=0.15,test_size=0.15):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size)
    if valid_size > 0:
        x_val, x_test, y_val, y_test = model_selection.train_test_split(x_test, y_test, test_size=test_size/(test_size + valid_size)) 
    else:
        x_val = None
        y_val = None
    return (x_train,y_train), (x_val,y_val), (x_test,y_test)

def train_test_split_custom(X,Y,valid_size=0.15,test_size=0.15,sampling='none',random_state=42):
    assert sampling in ['none','both','species','chemical']
    assert test_size > 0 or valid_size > 0
    
    if sampling == 'none':
        return train_valid_test_split(X,Y,valid_size,test_size)
    
    entities1, entities2, _ = map(list, zip(*X))
    entities1, entities2 = set(entities1),set(entities2)
    
    if sampling == 'chemical' or sampling == 'both':
        train, valid, test = train_valid_test_split(list(entities1),list(entities1),valid_size,test_size)
        train_chemicals = train[0]
        valid_chemicals = valid[0] or set() #If valid_size = 0
        test_chemicals = test[0]
    else:
        train_chemicals = entities1
        valid_chemicals = entities1 or set()
        test_chemicals = entities1
        
    if sampling == 'species' or sampling == 'both':
        train, valid, test = train_valid_test_split(list(entities2),list(entities2),valid_size,test_size)
        train_species = train[0]
        valid_species = valid[0] or set()
        test_species = test[0]
    else:
        train_species = entities2
        valid_species = entities2 or set()
        test_species = entities2
    
    X_train_tmp = []
    X_valid_tmp = []
    X_test_tmp = []
    y_train_tmp = []
    y_valid_tmp = []
    y_test_tmp = []

    for x,y in zip(X,Y):
        x1,x2,x3 = x
        if x1 in train_chemicals and x2 in train_species:
            X_train_tmp.append(x)
            y_train_tmp.append(y)
        elif x1 in valid_chemicals and x2 in valid_species:
            X_valid_tmp.append(x)
            y_valid_tmp.append(y)
        elif x1 in test_chemicals and x2 in test_species:
            X_test_tmp.append(x)
            y_test_tmp.append(y)
    
    X_train, X_valid, X_test, y_train, y_valid, y_test = list(map(np.asarray, [X_train_tmp, X_valid_tmp, X_test_tmp, y_train_tmp, y_valid_tmp, y_test_tmp]))
    
    # for sanity
    if sampling == 'chemical':
        assert len(set((X_train[:,0])).intersection(set(X_test[:,0]))) == 0
            
    if sampling == 'species':
        assert len(set((X_train[:,1])).intersection(set(X_test[:,1]))) == 0
    
    if sampling == 'both':
        assert len(set((X_train[:,0])).intersection(set(X_test[:,0]))) == 0 and len(set((X_train[:,1])).intersection(set(X_test[:,1]))) == 0
    
    train = (X_train,y_train)
    test = (X_test,y_test)
    if valid_size > 0:
        valid = (X_valid,y_valid)
    else:
        valid = (None,None)
    
    return train, valid, test

def load_data(filename):
    df = pd.read_csv(filename).dropna()
    f = lambda x: 1 if x == 'https://cfpub.epa.gov/ecotox/effect/MOR' else 0
    df['effect'] = df['effect'].apply(f)
    df = df.groupby(['chemical','species'],as_index=False).median()
    df1 = df[df['effect'] == 1]
    df2 = df[df['effect'] == 0]
    df = pd.concat((df1,df2))
    X, y = list(zip(df['chemical'],df['species'],df['concentration'])),list(df['effect'])
    return X,y 

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    r = true_positives / (possible_positives + K.epsilon())
    return r

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    p = true_positives / (predicted_positives + K.epsilon())
    return p

def f1(y_true, y_pred):
    beta = 1
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2) * (p*r)/(beta**2*p+r + K.epsilon())

def f2(y_true, y_pred):
    beta = 2
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2) * (p*r)/(beta**2*p+r + K.epsilon())

def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels = (1-factor)*labels + factor/labels.shape[1]

    # returned the smoothed labels
    return labels

def generate_negative(kg, N, negative=2, check_kg=False, corrupt_head=True, corrupt_tail=True):
    # false triples:
    assert corrupt_head or corrupt_tail
    R = np.repeat(np.asarray([p for _,p,_ in kg]).reshape((-1,1)),negative,axis=0)
    fs = np.random.randint(0,N,size=(negative*len(kg),1))  
    fo = np.random.randint(0,N,size=(negative*len(kg),1))  
    negative_kg = np.stack([fs,R,fo],axis=1)
    return negative_kg

def lengths(inputs):
    return [len(i) for i in inputs]

def joint_cosine_loss(x):
    """
    x : dot(chemical,species)
    """
    def func(y_true, y_pred):
        return K.reduce_sum(y_true*x + (y_true-1)*x)
    return func

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    







