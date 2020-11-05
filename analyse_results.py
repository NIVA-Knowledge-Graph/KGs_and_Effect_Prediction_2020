# analyse results

import pandas as pd
import math

from functools import reduce 
import operator
from collections import defaultdict
import numpy as np
from itertools import product

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

EPSILON = 1e-6

def sensitivity(tp,fn,tn,fp):
    return tp/(tp+fn)

def specificity(tp,fn,tn,fp):
    return tn/(tn+fp)

def dor(tp,fn,tn,fp):
    return math.log((tp*tn)/(fp*fn))

def confidence(std,num_samples,Z=1.96):
    return Z*std/math.sqrt(num_samples)

def balanced_accuracy(tp,fn,tn,fp):
    return (sensitivity(tp,fn,tn,fp)+specificity(tp,fn,tn,fp))/2
 
def youden(tp,fn,tn,fp):
    return sensitivity(tp,fn,tn,fp) + specificity(tp,fn,tn,fp) - 1
    
def youden_max(y_true,y_pred):
    out = {}
    x, y, thresholds = roc_curve(y_true,y_pred,drop_intermediate=False)
    f = lambda x: x
    y = y - f(x)
    idx = np.argmax(y)
    
    return y[idx], thresholds[idx]

def metrics(y_true,y_pred):
    tn,fp,fn,tp = confusion_matrix(y_true,np.around(y_pred)).ravel()
    
    j = youden(tp,fn,tn,fp)
    jmax,th = youden_max(y_true,y_pred)
    
    assert jmax >= j - EPSILON
    
    return [sensitivity(tp,fn,tn,fp), specificity(tp,fn,tn,fp), j, jmax, th, balanced_accuracy(tp,fn,tn,fp)]

def load_results(f):
    arr = np.load(f)
    y_pred, y_true = arr[:-1,:], arr[-1,:]
    m = []
    for y_hat in y_pred:
        m.append(metrics(y_true,y_hat))
    
    m = np.asarray(m)
    mean = np.mean(m,axis=0)
    std = np.std(m,axis=0)
    
    mean,score = mean[:-1], mean[-1]
    
    return {'value':mean,'std':std,'score':score}

def main():

    sampling_methods = ['none','chemical','species','both']

    models = [
            'DistMult', 
            'ComplEx',
            'HolE',
            'TransE',
            'RotatE',
            'pRotatE',
            'HAKE',
            'ConvKB',
            'ConvE',
        ]

    d = 'results/'
    header = 'Model & '+' & '.join(['Sensitivity','Specificity','YI',r'YI$_{max}$',r't_{max}'])+r'\\ \hline'


    popular_models1 = defaultdict(int)
    popular_models2 = defaultdict(int)
    for sm in sampling_methods:
        print('\n',sm)
        pretrained_files = [d+'%s_simple_pretrained_%s_%s.csv' % (sm,m1,m2) for m1,m2 in product(models,models)]
        finetune_files = [d+'%s_simple_joint_finetune_%s_%s.csv' % (sm,m1,m2) for m1,m2 in product(models,models)]
        one_hot_files = [d+'%s_simple_one_hot.csv' % sm]
        pretrained_files += [d+'%s_complex_pretrained_%s_%s.csv' % (sm,m1,m2) for m1,m2 in product(models,models)]
        finetune_files += [d+'%s_complex_joint_finetune_%s_%s.csv' % (sm,m1,m2) for m1,m2 in product(models,models)]
        one_hot_files += [d+'%s_complex_one_hot.csv' % sm]
        
        best = {}
        for f in pretrained_files + one_hot_files + finetune_files:
            try:
                best[f] = load_results(f.replace('/','/predictions_').replace('csv','npy'))
            except FileNotFoundError or KeyError:
                pass
            
        print(header)
        for b,a in product(['simple','complex'],['one_hot','pretrained','joint_finetune']):
            tmp = best.copy()
            ks = filter(lambda x: '_'.join([b,a]) in x, tmp.keys())
            ks = sorted(ks,key=lambda x: tmp[x]['score'],reverse=True)
            
            prefix = ''
            if a == 'pretrained':
                prefix = 'PT ' 
            if a == 'joint_finetune':
                prefix = 'FT '
                
            if b == 'simple':
                prefix = 'Simple '+prefix
            else:
                prefix = 'Complex '+prefix
            
            if a == 'pretrained':
                for k in ks[:10]:
                    m1,m2 = k.split('_')[-2:]
                    m2=m2[:-4]
                    popular_models1[(sm,b,m1)] += 1
                    popular_models2[(sm,b,m2)] += 1
                ks = ks[:3] + [ks[len(ks)//2-1]] + [ks[-1]]
                pretrained_ks = ks
            
            if a=='joint_finetune':
                order = dict(filter(lambda x: '_'.join([b,'pretrained']) in x[0], tmp.items()))
                order = {k.replace('pretrained','joint_finetune'):i for k,i in order.items()}
                ks = sorted(ks,key=lambda x: order[x]['score'],reverse=True)
                
            
            for k in ks:
                m1,m2 = k.split('_')[-2:]
                m2=m2[:-4]
                if a == 'joint_finetune':
                    if not any(['_'.join([m1,m2]) in v for v in pretrained_ks]):
                        continue
                    
                s = prefix+m1+'-'+m2
                
                for v,std in zip(tmp[k]['value'],tmp[k]['std']):
                    s += ' & '+ '$'+str(round(v,3))
                    if std:
                        s+=' \pm '+ str(round(std,3))
                    s+='$'
                print(s + r' \\ \hline')
    print('\n')
    for m in models:
        s = m +' & ' + ' & '.join(['$(%s,%s)/(%s,%s)$' % (popular_models1[(sm,'simple',m)],
                                                popular_models2[(sm,'simple',m)],
                                                popular_models1[(sm,'complex',m)],
                                                popular_models2[(sm,'complex',m)]) for sm in sampling_methods])
        print(s,r'\\ \hline')
            
if __name__ == '__main__':
    main()
            
            
            
