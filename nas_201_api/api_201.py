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
from libs.nasbench201.nas_201_api import model_spec_201 as _model_spec

# Bring ModelSpec to top-level for convenience. See lib/model_spec.py.
ModelSpec = _model_spec.ModelSpec


class NASBench201API(object):

    def __init__(self, path: Optional[Union[Text, Dict]]=None,
                             verbose: bool=True):
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
        
        # Data Stored in the class
        self.meta_archs = copy.deepcopy( path['meta_archs'] )
        self.arch2infos_dict = {}
        self.archstr2index = {}
        self.hash2archstr = {}
        self._avaliable_hps = set(['12', '200'])
        self.evaluated_indexes = sorted(list(path['evaluated_indexes']))
        self.search_space = ['skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        self.total_time = 0
        self.total_epochs = 0 # not implement yet
    
        for archidx in list(path['arch2infos'].keys()):
            self.arch2infos_dict[archidx] = path['arch2infos'][archidx]
        
        for idx, arch in enumerate(self.meta_archs):
            assert arch not in self.archstr2index, 'This [{:}]-th arch {:} already in the dict ({:}).'.format(idx, arch, self.archstr2index[arch])
            self.archstr2index[ arch ] = idx
            modelspec = ModelSpec(model_str=arch, index=idx)
            max_nodes = max(max_nodes, modelspec.num_nodes)
            max_edges = max(max_edges, modelspec.num_edges)
            hash_val = modelspec.hash_spec(self.search_space)
            self.hash2archstr[hash_val] = arch

        self.max_edges = max_edges
        self.max_nodes = max_nodes
    
    def hash_iterator(self):
        return self.hash2archstr.keys()
    
    # hash -> ModelSpec (idx, str, matrix, ops)
    def get_model_spec_by_hash(self, hash):
        archstr = self.hash2archstr[hash]
        arch = ModelSpec(model_str=archstr, index=self.archstr2index[archstr])
        return arch
        
    def get_modelspec(self, matrix, ops):
        model_spec = ModelSpec(matrix=matrix, ops=ops)
        
        hash_val = model_spec.hash_spec()
        if hash_val not in self.hash2archstr.keys():
            return False
        model_str = self.hash2archstr[hash_val]
        if model_str not in self.archstr2index.keys():
            return False
        index = self.archstr2index[model_str]
        
        model_spec.index = index
        model_spec.model_str = model_str
        
        return model_spec
    
    def is_valid(self, model_spec):
        try:
            self._check_spec(model_spec)
        except OutOfDomainError:
            return False

        return True
    
    def _check_spec(self, model_spec):
        """Checks that the model spec is within the dataset."""
        if not model_spec.valid_spec:
            raise OutOfDomainError('invalid spec, provided graph is disconnected.')

        num_vertices = len(model_spec.ops)
        num_edges = np.sum(model_spec.matrix)

        if num_vertices > self.max_nodes:
            raise OutOfDomainError('too many vertices, got %d (max vertices = %d)'
                                                         % (num_vertices, self.max_nodes))

        if num_edges > self.max_edges:
            raise OutOfDomainError('too many edges, got %d (max edges = %d)'
                                                         % (num_edges, self.max_edges))

        if model_spec.ops[0] != 'input':
            raise OutOfDomainError('first operation should be \'input\'')
        if model_spec.ops[-1] != 'output':
            raise OutOfDomainError('last operation should be \'output\'')
        for op in model_spec.ops[1:-1]:
            if op not in self.search_space:
                raise OutOfDomainError('unsupported op %s (available ops = %s)'
                                                             % (op, self.search_space))
                
    def get_budget_counters(self):
        return self.total_time, self.total_epochs
    
    def query(self, modelspec, option, dataset='cifar10-valid'):
        """
        dataset         :  train, validation, test
        
        [cifar10-valid] : 'train', 'x-valid', 'ori-test'
        [cifar10]       : 'train'(train+val), 'ori-test'
        [cifar100
        ImageNet16-120] : 'train', 'x-valid', 'x-test', 'ori-test'(val+test)
        """
        assert option == 'valid' or option == 'test'

        if option == 'valid':
            archresult = self.arch2infos_dict[modelspec.index]['less']['all_results']
            seeds = self.arch2infos_dict[modelspec.index]['less']['dataset_seed'][dataset]
            picked_seed = random.choice(seeds)
            
            info = archresult[(dataset, picked_seed)]
            
            eval_name = [n for n in info['eval_names'] if 'valid' in n][0]
            self.total_time += (info['train_times'] + info['eval_times'][eval_name])
            
            return info['eval_acc1es'][eval_name]
        
        elif option == 'test':
            archresult = self.arch2infos_dict[modelspec.index]['less']['all_results']
            seeds = self.arch2infos_dict[modelspec.index]['less']['dataset_seed'][dataset]
            avg_test_acc = 0
            
            for seed in seeds:
                info = archresult[(dataset, seed)]
                test_name = [n for n in info['eval_names'] if 'test' in n][0]
                avg_test_acc += info['eval_acc1es'][test_name]
                
            avg_test_acc /= len(seeds)
            
            return avg_test_acc