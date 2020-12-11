#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
############################################################################################
# NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search, ICLR 2020 #
############################################################################################
# The history of benchmark files:
# [2020.02.25] NAS-Bench-201-v1_0-e61699.pth : 6219 architectures are trained once, 1621 architectures are trained twice, 7785 architectures are trained three times. `LESS` only supports CIFAR10-VALID.
# [2020.03.16] NAS-Bench-201-v1_1-096897.pth : 2225 architectures are trained once, 5439 archiitectures are trained twice, 7961 architectures are trained three times on all training sets. For the hyper-parameters with the total epochs of 12, each model is trained on CIFAR-10, CIFAR-100, ImageNet16-120 once, and is trained on CIFAR-10-VALID twice.
#
# I'm still actively enhancing this benchmark. Please feel free to contact me if you have any question w.r.t. NAS-Bench-201.
#

import os, copy, random, torch, numpy as np
from pathlib import Path
from typing import List, Text, Union, Dict, Optional
from collections import OrderedDict, defaultdict

from nasbench.lib import graph_util
from nas_201_api import model_spec as _model_spec

# Bring ModelSpec to top-level for convenience. See lib/model_spec.py.
ModelSpec = _model_spec.ModelSpec

class OutOfDomainError(Exception):
  """Indicates that the requested graph is outside of the search domain."""

class NASBench201API(object):

    def __init__(self, path: Optional[Union[Text, Dict]]=None,
                             verbose: bool=True):
        """Initialize dataset."""
        # load pth file    
        if isinstance(path, str) or isinstance(path, Path):
            path = str(path)
            if verbose: print('try to create the NAS-Bench-201 api from {:}'.format(path))
            assert os.path.isfile(path), 'invalid path : {:}'.format(path)
            
            path = torch.load(path, map_location='cpu')  
        elif isinstance(path, dict):
            path = copy.deepcopy(path)            
        else:
            raise ValueError('invalid type : {:} not in [str, dict]'.format(type(path)))
        assert isinstance(path, dict), 'It should be a dict instead of {:}'.format(type(path))
                
        keys = ('meta_archs', 'arch2infos', 'evaluated_indexes')
        for key in keys:
            assert key in path, 'Can not find key[{:}] in the dict'.format(key)
        
        # Stores the list of arch_str.
        # ex)'|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|'
        self.meta_archs = copy.deepcopy( path['meta_archs'] )

        # Stores the statistics that are computed via training and evaluating the model on cifar10, cifar100, ImageNet16. 
        # Statistics are computed for multiple repeats of each model at each max epoch length.
        # hash --> hash_value
        #      --> arch_str
        #      --> dataset_seed --> data name : [seed1, seed2, ... ]
        #      --> all_results  --> (data name, seed) --> metric name --> scalar
        # Stores architecture string
        # cf)   matrix, ops -> modelspec                     -> hash --> hash2archstr -> get arch str 
        # cf)   arch str    -> modelspec(get matrix and ops) -> hash --> hash2results -> get statistics
        self.hash2infos = path['arch2infos']

        self._avaliable_epochs = ['less': 20, 'full': 200]
        self._avaliable_datasets = ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120']
        self.allhashvalues = list(path['arch2infos'].keys())
        self.search_space = ['skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        self.total_time = 0
        self.total_epochs = 0 # not implement yet

        max_nodes=0
        max_edges=0
        for hash_val in sorted(list()):
            arch_str = path['arch2infos'][hash_val]['arch_str']
            assert arch_str in self.meta_archs, 'This [{:}]-th arch not in the meta_archs. This arch is {:}.'.format(archidx, arch_str)
            
            modelspec = ModelSpec(model_str=arch_str)
            max_nodes = max(max_nodes, modelspec.num_nodes)
            max_edges = max(max_edges, modelspec.num_edges)
        self.max_edges = max_edges
        self.max_nodes = max_nodes
    
    def hash_iterator(self):
        """Returns iterator over all unique model hashes."""
        return self.allhashvalues
    
    def get_modelspec(self, matrix, ops):
        """Return model spec."""        
        return ModelSpec(matrix=matrix, ops=ops)
    
    def get_modelspec_by_hash(self, hash_val):
        """Return modelspec by hash value"""
        archstr = self.hash2infos[hash_val]['arch_str']
        arch = ModelSpec(model_str=archstr)
        return arch
    
    def get_budget_counters(self):
        """Returns the time and budget counters."""
        return self.total_time, self.total_epochs

    def _check_spec(self, model_spec):
        """Checks that the model spec is within the dataset."""
        if not model_spec.valid_spec:
            raise OutOfDomainError('invalid spec, provided graph is disconnected.')
        
        model_hash = model_spec.hash_spec(self.search_space)
        if model_hash not in self.allhashvalues:
            raise OutOfDomainError('unsupported model hash %s ' % (model_hash))

        num_vertices = len(model_spec.ops)
        num_edges = np.sum(model_spec.matrix)
        if num_vertices > self.max_nodes:
            raise OutOfDomainError('too many vertices, got %d (max vertices = %d)' % (num_vertices, self.max_nodes))
        if num_edges > self.max_edges:
            raise OutOfDomainError('too many edges, got %d (max edges = %d)' % (num_edges, self.max_edges))

        if model_spec.ops[0] != 'input':
            raise OutOfDomainError('first operation should be \'input\'')
        if model_spec.ops[-1] != 'output':
            raise OutOfDomainError('last operation should be \'output\'')
        for op in model_spec.ops[1:-1]:
            if op not in self.search_space:
                raise OutOfDomainError('unsupported op %s (available ops = %s)' % (op, self.search_space))
    
    def is_valid(self, model_spec):
        """Checks the validity of the model_spec."""
        try:
            self._check_spec(model_spec)
        except OutOfDomainError:
            return False
        
        return True
    
    def query(self, modelspec, option, dataset='cifar10-valid', epochs='less'):
        """Fetch one of the evaluations for this model spec
        * option == 'valid'
        Fetch one of the evaluations for this model spec by sampling one seed from seed_set.
        increment the budget counters for benchmarking purposes.
        
        * option == 'test'
        Returns the average metrics of all repeats of this model spec.
        not be used for benchmarking. so not increment any of the budget counters.
        
        dataset            train, validation, test   
        ---------------------------------------------------------------------     
        [cifar10-valid] : 'train', 'x-valid', 'ori-test'
        [cifar10]       : 'train'(train+val), 'ori-test'
        [cifar100,
        ImageNet16-120] : 'train', 'x-valid', 'x-test', 'ori-test'(val+test)

        Raises error:
            invalid option, dataset, epochs
        """
        assert option == 'valid' or option == 'test'
        assert dataset in self._avaliable_datasets
        assert epochs in self._avaliable_epochs.keys()

        if option == 'valid':
            model_hash = modelspec.hash_spec(self.search_space)
            archresult = self.hash2infos[model_hash]['all_results'][epochs]
            seeds = self.hash2infos[model_hash]['dataset_seed'][epochs][dataset]
            picked_seed = random.choice(seeds)
            info = archresult[(dataset, picked_seed)]
            eval_name = [n for n in info['eval_names'] if 'valid' in n][0]
            self.total_time += (info['train_times'] + info['eval_times'][eval_name])
            self.total_epochs += self._available_epochs[epochs]            
            return info['eval_acc1es'][eval_name]        
        elif option == 'test':
            archresult = self.hash2infos[model_hash]['all_results'][epochs]
            seeds = self.hash2results[model_hash]['dataset_seed'][epochs][dataset]
            avg_test_acc = 0
            for seed in seeds:
                info = archresult[(dataset, seed)]
                test_name = [n for n in info['eval_names'] if 'test' in n][0]
                avg_test_acc += info['eval_acc1es'][test_name]
            avg_test_acc /= len(seeds)            
            return avg_test_acc
