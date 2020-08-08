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
import model_spec_201 as _model_spec

# Bring ModelSpec to top-level for convenience. See lib/model_spec.py.
ModelSpec = _model_spec.ModelSpec


class NASBench201API(object):

    def __init__(self, file_path_or_dict: Optional[Union[Text, Dict]]=None,
                             verbose: bool=True):
        self.filename = None
        
        # load pth file    
        if isinstance(file_path_or_dict, str) or isinstance(file_path_or_dict, Path):
            file_path_or_dict = str(file_path_or_dict)
            if verbose: print('try to create the NAS-Bench-201 api from {:}'.format(file_path_or_dict))
            assert os.path.isfile(file_path_or_dict), 'invalid path : {:}'.format(file_path_or_dict)
            
            self.filename = Path(file_path_or_dict).name
            file_path_or_dict = torch.load(file_path_or_dict, map_location='cpu')
            
        elif isinstance(file_path_or_dict, dict):
            file_path_or_dict = copy.deepcopy(file_path_or_dict)
            
        else: raise ValueError('invalid type : {:} not in [str, dict]'.format(type(file_path_or_dict)))
            
        assert isinstance(file_path_or_dict, dict), 'It should be a dict instead of {:}'.format(type(file_path_or_dict))
        
        
        self.verbose = verbose # [TODO] a flag indicating whether to print more logs
        keys = ('meta_archs', 'arch2infos', 'evaluated_indexes')
        for key in keys: assert key in file_path_or_dict, 'Can not find key[{:}] in the dict'.format(key)
        
        # Data Stored in the class
        self.meta_archs = copy.deepcopy( file_path_or_dict['meta_archs'] )
        self.arch2infos_dict = OrderedDict()
        self.archstr2index = {}
        self.hash2archstr = {}
        self._avaliable_hps = set(['12', '200'])
        self.evaluated_indexes = sorted(list(file_path_or_dict['evaluated_indexes']))
        self.search_space = ['skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        self.total_time = 0
        self.total_epochs = 0 # not implement yet
    
        for archidx in sorted(list(file_path_or_dict['arch2infos'].keys())):
            self.arch2infos_dict[archidx] = file_path_or_dict['arch2infos'][archidx]
        
        for idx, arch in enumerate(self.meta_archs):
            assert arch not in self.archstr2index, 'This [{:}]-th arch {:} already in the dict ({:}).'.format(idx, arch, self.archstr2index[arch])
            self.archstr2index[ arch ] = idx
            modelspec = ModelSpec(model_str=arch, index=idx)
            hash_val = modelspec.hash_spec()
            self.hash2archstr[hash_val] = arch
    
    def hash_iterator(self):
        return self.hash2archstr.keys()
    
    # hash -> ModelSpec (idx, str, matrix, ops)
    def get_model_spec_by_hash(self, hash):
        archstr = self.hash2archstr[hash]
        arch = ModelSpec(model_str=archstr, index=self.archstr2index[archstr])
        return arch
        
    # matrix, ops -> looking into all archs and find isomorphic network.
    def get_modelspec(self, matrix, ops):
        for key in self.hash_iterator():
            arch = self.get_model_spec_by_hash(key)
            arch_matrix = arch.matrix.tolist()
            arch_labeling = [-1] + [self.search_space.index(op) for op in arch.ops[1:-1]] + [-2]
            graph1 = (arch_matrix, arch_labeling)
            
            labeling = [-1] + [self.search_space.index(op) for op in ops[1:-1]] + [-2]
            graph2 = (matrix.tolist(), labeling)
            
            if is_isomorphic(graph1, graph2):
                return arch
        # Error
        return False
    
    def get_budget_counters(self):
        return self.total_time, self.total_epochs
    
    """
    dataset         :  train, validation, test
    
    [cifar10-valid] : 'train', 'x-valid', 'ori-test'
    [cifar10]       : 'train'(train+val), 'ori-test'
    [cifar100
    ImageNet16-120] : 'train', 'x-valid', 'x-test', 'ori-test'(val+test)
    """
    def query_option(self, modelspec, option, dataset='cifar10-valid'):
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