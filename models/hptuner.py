
from collections import defaultdict
import numpy as np
import warnings
from random import choice, shuffle
from itertools import product

class HPTuner:
    def __init__(self, runs=10, objectiv_direction='max'):
        self._runs = runs
        self.experiments = {}
        self.hps = {}
        self.objectiv_direction = objectiv_direction
        self.best = 0.0 if objectiv_direction == 'max' else float('inf')
        self.history = [self.best]
    
    def add_result(self,res):
        if np.isnan(res): 
            warnings.warn('Added result is NaN.')
        else:
            self.experiments[res] = self.curr
            self.best = max(self.best,res) if self.objectiv_direction=='max' else min(self.best,res)
            self.history.append(self.best)
            self.__update_landscape()
        self._runs -= 1
    
    def _add_hp(self,name,d):
        if name in self.hps: 
            warnings.warn('%s allready in hyper-parameters, overwriting.' % name)
        self.hps[name] = d
    
    def add_value_hp(self,name,minimum,maximum,default=None,dtype=float,sampling=None,exhaustive=False):
        # sampling in {None, 'log'}
        
        if dtype==int:
            if sampling=='log':
                return self.add_list_hp(name,list(10**np.arange(minimum,maximum)), default=default, exhaustive=exhaustive)
            else:
                return self.add_list_hp(name,list(np.arange(minimum,maximum)), default=default, exhaustive=exhaustive)
            
        elif dtype==float and exhaustive:
            if sampling=='log':
                return self.add_list_hp(name,list(10**np.arange(minimum,maximum,0.1*(maximum-minimum))), default=default, exhaustive=exhaustive)
            else:
                return self.add_list_hp(name,list(np.arange(minimum,maximum,0.1*(maximum-minimum))), default=default, exhaustive=exhaustive)
            warnings.warn('Adding float HPs in exhaustive mode creates 10 evenly spaced HPs.')
            
            
        if sampling == 'log':
            f = lambda x: 10**dtype(x)
            f_inv = lambda x: np.log10(x)
        else:
            f = lambda x: dtype(x)
            f_inv = lambda x: x
            
        if default:
            df = lambda x: 10**dtype(default) if sampling=='log' else dtype(default)
        else:
            df = f
        
        self._add_hp(name,{'min':minimum,'max':maximum,'sampling':f,'default':df,'values':None,'exhaustive':exhaustive})
    
    def add_list_hp(self,name,choices,default=None,exhaustive=False):
        if default: assert default in choices
        f = lambda x: choices[int(x)]
        f_inv = lambda x: x
        if default:
            df = lambda x: default
        else:
            df = f
        self._add_hp(name,{'min':0,'max':len(choices)-1,'sampling':f,'default':df,'values':choices.copy(),'exhaustive':exhaustive})
    
    def add_fixed_hp(self,name,value):
        f = lambda x: value
        f_inv = lambda x: value
        df = lambda x: value
        self._add_hp(name,{'min':0,'max':1,'sampling':f,'inv_sampling':f_inv,'default':df,'values':[value],'exhaustive':False})
        
    def next_hp_config(self):
        if not hasattr(self,'all_configs'):
            self.all_configs = []
            
            idx_ex = [i for i,k in enumerate(self.hps) if self.hps[k]['exhaustive']]
            while len(self.all_configs) < self.runs:
                for v in product(*[self.hps[k]['values'] for k in self.hps if self.hps[k]['exhaustive']]):
                    v = list(v)
                    p = []
                    for i,k in enumerate(self.hps):
                        if i in idx_ex:
                            p.append(v.pop(0))
                        else:
                            p.append(self.hps[k]['sampling'](np.random.uniform(self.hps[k]['min'],self.hps[k]['max'])))
                    self.all_configs.append(p)
                    
            shuffle(self.all_configs)
            self.all_configs = self.all_configs[:self.runs]
       
        out = {k:i for k,i in zip(self.hps,self.all_configs.pop(0))}
        self.curr = out.copy()
        return out.copy()
    
    def get_default_config(self):
        out = self.hps.copy()
        for k in out:
            out[k] = out[k]['default'](np.random.uniform(out[k]['min'],out[k]['max']))
        
        self.curr = out
        return out
    
    def get(self,key):
        return self.hps[key]['default'](1)
    
    def __update_landscape(self):
        #TODO: implement baysian optimazation
        pass
    
    def best_config(self):
        try:
            return self.experiments[self.best]
        except KeyError:
            return self.get_default_config()
    
    def set_runs(self,num):
        self._runs = num
    
    @property
    def is_active(self):
        if hasattr(self,'all_configs'):
            return len(self.all_configs) > 0
        elif self._runs < 1:
            return False
        else:
            return True
    
    @property 
    def runs(self):
        return max(self._runs,len(list(product(*[self.hps[k]['values'] for k in self.hps if self.hps[k]['exhaustive']]))))
    
    
    
    
    
    
    
    
