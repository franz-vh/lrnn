#!/usr/bin/env python3

'''Functionality to optimize HLRNN hyperparameters gradually.

This functionality consists mainly of three classes:
A class that is able to generate, at each iteration, the search parameters
required, as well as to map those search parameters to the corresponding model
initialization and optimization parameters.
A class that implements the training and test functions of an HLRNN block,
keeping its state.
A class that implements the training and test functions of an HLRNN search
iteration trial, keeping its state.
While this code can work both with batched and non-batched sequences, variable
batch_size refers to different things in each case. When working with
non-batched sequences, variable batch_size refers to the sequence length. When
working with batched sequences, variable batch_size refers to the number of
sequences in the batch (and variable seq_size to the sequence length).
'''

import lrnn
import attention
import torch
import math
from functools import partial
from collections import OrderedDict
import typing
from typing import Any, List, Tuple, Dict, Callable


class BlockTypeParams:
    '''Model, optimization and search parameter names of an HLRNN block type.

     Atributes:
        name: A string with the HLRNN block type name.
        block_class: A class with the HLRNN block type.
        search_params: A list of strings with the names of the search
            parameters.
        get_model_params: A function that obtains the model initialization
            parameters from the input size(s), the search parameters and
            possibly the output size.
        get_opt_params: A function that obtains the optimization parameters
            from the search parameters.
        get_out_features: A function that obtains the output size from the
            input size(s) and the search parameters.
        hiddens: A list of strings with the names of the hidden states.
    '''
    def __init__(self, name: str, block_class: Callable[[Any],
        torch.nn.Module], search_params: List[str], get_model_params:
        Callable[[List[int], Dict[str, Any], int], Dict[str, Any]],
        get_opt_params: Callable[[Dict[str, Any]], Dict[str, Any]],
        get_out_features: Callable[[List[int], Dict[str, Any]], List[int]] =
        None, hiddens: List[str] = []) -> None:
        '''Inits BlockTypeParams.

        Arguments:
            name: The HLRNN block type name.
            block_class: The HLRNN block type.
            search_params: The names of the search parameters.
            get_model_params: Obtains the model initialization parameters from
                the input size(s), the search parameters and possibly the
                output size.
            get_opt_params: Obtains the optimization parameters from the search
                parameters.
            get_out_features: Obtains the output size from the input size(s)
                and the search parameters.
            hiddens: A list of strings with the names of the hidden states.
        '''
        self.name, self.block_class = name, block_class
        self.search_params = search_params
        self.get_model_params, self.get_opt_params, self.get_out_features = \
            get_model_params, get_opt_params, get_out_features
        self.hiddens = hiddens

def _get_slice_out_features(in_features, search_params):
    start = search_params['start']
    stop = search_params['stop']
    if start < 0:
        start += in_features
    if stop < 0:
        stop += in_features
    return max(min(stop, in_features), 0) - max(min(start, in_features), 0)

temp_max_pool_params = BlockTypeParams('TempMaxPool', lrnn.TempMaxPool,
    ['tmpk_sz', 'tmpk_st', 'tmpk_ct'],
    lambda in_features, search_params: {
        'kernel_size': search_params['tmpk_sz'],
        'stride': search_params['tmpk_st'],
        'centered': search_params['tmpk_ct']},
    lambda search_params: {},
    lambda in_features, search_params: in_features,
    ['hidden'])
temp_window_params = BlockTypeParams('TempWindow', lrnn.TempWindow,
    ['w_sz'],
    lambda in_features, search_params: {
        'window_size': search_params['w_sz']},
    lambda search_params: {},
    lambda in_features, search_params: in_features * search_params['w_sz'],
    ['hidden'])
downsample_params = BlockTypeParams('Downsample', lrnn.Downsample,
    ['tmpk_st'],
    lambda in_features, search_params: {
        'stride': search_params['tmpk_st']},
    lambda search_params: {},
    lambda in_features, search_params: in_features)
upsample_params = BlockTypeParams('Upsample', lrnn.Upsample,
    ['tmpk_st', 'up_mod'],
    lambda in_features, search_params: {
        'scale_factor': search_params['tmpk_st'],
        'mode': search_params['up_mod']},
    lambda search_params: {},
    lambda in_features, search_params: in_features)
multi_upsample_params = BlockTypeParams('MultiUpsample', lrnn.MultiUpsample,
    ['tmpk_st', 'up_mod'],
    lambda in_features, search_params: {
        'scale_factor': search_params['tmpk_st'],
        'mode': search_params['up_mod']},
    lambda search_params: {},
    lambda in_features, search_params: in_features)
rect_gauss_mix_params = BlockTypeParams('RectGaussMix', lrnn.RectGaussMix,
    [],
    lambda in_features, search_params: {},
    lambda search_params: {},
    lambda in_features, search_params: in_features[0] // in_features[2])
rect_gauss_mix_peak_params = BlockTypeParams('RectGaussMixPeak',
    lrnn.RectGaussMixPeak,
    [],
    lambda in_features, search_params: {},
    lambda search_params: {},
    lambda in_features, search_params: in_features[0] // in_features[2])
cat_params = BlockTypeParams('Cat', lrnn.Cat,
    [],
    lambda in_features, search_params: {},
    lambda search_params: {},
    lambda in_features, search_params: sum(in_features))
slice_params = BlockTypeParams('Slice', lrnn.Slice,
    ['start', 'stop'],
    lambda in_features, search_params: {
        'start': search_params['start'],
        'stop': search_params['stop']},
    lambda search_params: {},
    _get_slice_out_features)
trim_like_params = BlockTypeParams('TrimLike', lrnn.TrimLike,
    [],
    lambda in_features, search_params: {},
    lambda search_params: {},
    lambda in_features, search_params: in_features[1:])
emulate_nonlinearity_params = BlockTypeParams('EmulateNonlinearity',
        lrnn.EmulateNonlinearity,
    ['act', 'des'],
    lambda in_features, search_params: {
        'actual': search_params['act'],
        'desired': search_params['des']},
    lambda search_params: {},
    lambda in_features, search_params: in_features)
function_params = BlockTypeParams('Function', lrnn.Function,
    ['func'],
    lambda in_features, search_params: {
        'function': search_params['func']},
    lambda search_params: {},
    lambda in_features, search_params: in_features)
attention_params = BlockTypeParams('Attention', attention.Attention,
    ['nhead', 'gru_sz', 'hid_sz', 'gru_sz2', 'hid_sz2', 'out_ch', 'dpt_p',
        'n_std', 'btch_sz', 'lr', 'err', 'lmbd', 'nbeta'],
    lambda in_features, search_params: {
        'fix_in_features': in_features[0],
        'att_in_features': in_features[1],
        'nhead': search_params['nhead'],
        'query_gru_features': search_params['gru_sz'],
        'query_hidden_features': search_params['hid_sz'],
        'dec_gru_features': search_params['gru_sz2'],
        'dec_hidden_features': search_params['hid_sz2'],
        'train_horizon': search_params['out_ch'],
        'dropout_p': search_params['dpt_p'],
        'noise_std': search_params['n_std']},
    lambda search_params: {
        'batch_size': search_params['btch_sz'],
        'lr': search_params['lr'],
        'err': search_params['err'],
        'lmbd': search_params['lmbd'],
        'nbeta': search_params['nbeta']},
    lambda in_features, search_params: in_features[0] + search_params['nhead']
        * in_features[1],
    ['q_gru', 'dec_gru', 'query'])
nilrnn_params = BlockTypeParams('NILRNN', lrnn.NILRNN,
    ['loc_sd', 'lock_n', 'plk_n', 'plk_st', 'out_ch', 'dpt_p', 'n_std',
        'loc_nl', 'out_nl', 'k_sp', 'out_ft', 'btch_sz', 'lr', 'err', 'lmbd',
        'beta', 'rho', 'gamma', 'delta'],
    lambda in_features, search_params: {
        'in_features': in_features,
        'rec_shape': (search_params['loc_sd'],) * 2,
        'rec_kernel_numel': search_params['lock_n'],
        'pool_kernel_numel': search_params['plk_n'],
        'pool_stride': search_params['plk_st'],
        'train_out_channels': search_params['out_ch'],
        'dropout_p': search_params['dpt_p'],
        'noise_std': search_params['n_std'],
        'rec_nonlinearity': search_params['loc_nl'],
        'train_out_nonlinearity': search_params['out_nl'],
        'kernel_shape': search_params['k_sp'],
        'return_contr': search_params['out_ft']},
    lambda search_params: {
        'batch_size': search_params['btch_sz'],
        'lr': search_params['lr'],
        'err': search_params['err'],
        'out_channels': search_params['out_ch'],
        'lmbd': search_params['lmbd'],
        'beta': search_params['beta'],
        'rho': search_params['rho'],
        'contr': search_params['out_ft'],
        'gamma': search_params['gamma'],
        'delta': search_params['delta']},
    lambda in_features, search_params: math.ceil(search_params['loc_sd'] /
        search_params['plk_st']) ** 2,
    ['hidden'])
lrb_params = BlockTypeParams('LRB', lrnn.LRB,
    ['loc_sd', 'lock_n', 'plk_n', 'plk_st', 'tmpk_sz', 'tmpk_st', 'out_ch',
        'dpt_p', 'n_std', 'loc_nl', 'out_nl', 'k_sp', 't_pos', 'tmpk_ct',
        'out_ft', 'btch_sz', 'lr', 'err', 'lmbd', 'beta', 'rho', 'gamma',
        'delta'],
    lambda in_features, search_params: {
        'in_features': in_features,
        'rec_shape': (search_params['loc_sd'],) * 2,
        'rec_kernel_numel': search_params['lock_n'],
        'pool_kernel_numel': search_params['plk_n'],
        'pool_stride': search_params['plk_st'],
        'temp_kernel_size': search_params['tmpk_sz'],
        'temp_stride': search_params['tmpk_st'],
        'train_out_channels': search_params['out_ch'],
        'dropout_p': search_params['dpt_p'],
        'noise_std': search_params['n_std'],
        'rec_nonlinearity': search_params['loc_nl'],
        'train_out_nonlinearity': search_params['out_nl'],
        'kernel_shape': search_params['k_sp'],
        'temp_pos': search_params['t_pos'],
        'temp_kernel_centered': search_params['tmpk_ct'],
        'return_contr': search_params['out_ft']},
    lambda search_params: {
        'batch_size': search_params['btch_sz'],
        'lr': search_params['lr'],
        'err': search_params['err'],
        'out_channels': search_params['out_ch'],
        'lmbd': search_params['lmbd'],
        'beta': search_params['beta'],
        'rho': search_params['rho'],
        'contr': search_params['out_ft'],
        'gamma': search_params['gamma'],
        'delta': search_params['delta']},
    lambda in_features, search_params: math.ceil(search_params['loc_sd'] /
        search_params['plk_st']) ** 2,
    ['rec', 'temp'])
deep_lrb_params = BlockTypeParams('DeepLRB', lrnn.DeepLRB,
    ['hid_sz', 'loc_sd', 'lock_n', 'plk_n', 'plk_st', 'tmpk_sz', 'tmpk_st',
        'out_ch', 'dpt_p', 'n_std', 'hid_nl', 'loc_nl', 'out_nl', 'k_sp',
        't_pos', 'tmpk_ct', 'out_ft', 'btch_sz', 'lr', 'err', 'lmbd', 'beta',
        'rho', 'gamma', 'delta'],
    lambda in_features, search_params: {
        'in_features': in_features,
        'hidden_features': search_params['hid_sz'],
        'rec_shape': (search_params['loc_sd'],) * 2,
        'rec_kernel_numel': search_params['lock_n'],
        'pool_kernel_numel': search_params['plk_n'],
        'pool_stride': search_params['plk_st'],
        'temp_kernel_size': search_params['tmpk_sz'],
        'temp_stride': search_params['tmpk_st'],
        'train_out_channels': search_params['out_ch'],
        'dropout_p': search_params['dpt_p'],
        'noise_std': search_params['n_std'],
        'hidden_nonlinearity': search_params['hid_nl'],
        'rec_nonlinearity': search_params['loc_nl'],
        'train_out_nonlinearity': search_params['out_nl'],
        'kernel_shape': search_params['k_sp'],
        'temp_pos': search_params['t_pos'],
        'temp_kernel_centered': search_params['tmpk_ct'],
        'return_contr': search_params['out_ft']},
    lambda search_params: {
        'batch_size': search_params['btch_sz'],
        'lr': search_params['lr'],
        'err': search_params['err'],
        'out_channels': search_params['out_ch'],
        'lmbd': search_params['lmbd'],
        'beta': search_params['beta'],
        'rho': search_params['rho'],
        'contr': search_params['out_ft'],
        'gamma': search_params['gamma'],
        'delta': search_params['delta']},
    lambda in_features, search_params: math.ceil(search_params['loc_sd'] /
        search_params['plk_st']) ** 2,
    ['rec', 'temp'])
lrb_predictor_params = BlockTypeParams('LRBPredictor', lrnn.LRBPredictor,
    ['n_cmp', 'hid_sz', 'ctxt_h', 'out_ch', 'rho', 'btch_sz', 'lr', 'lmbd'],
    lambda in_features, search_params: {
        'main_features': in_features[0],
        'main_n_components': search_params['n_cmp'],
        'hidden_features': search_params['hid_sz'],
        'ctxt_features': in_features[1],
        'ctxt_n_components': None if len(in_features) == 3 else in_features[4],
        'ctxt_horizon': search_params['ctxt_h'],
        'horizon': search_params['out_ch'],
        'rho': search_params['rho']},
    lambda search_params: {
        'batch_size': search_params['btch_sz'],
        'lr': search_params['lr'],
        'err': 'nll',
        'lmbd': search_params['lmbd']},
    lambda in_features, search_params: [search_params['n_cmp'] * in_features[
        0], search_params['n_cmp'] * in_features[0], search_params['n_cmp']])
gru_lrb_predictor_params = BlockTypeParams('GRULRBPredictor',
    lrnn.GRULRBPredictor,
    ['n_cmp', 'gru_sz', 'hid_sz', 'out_ch', 'rho', 'btch_sz', 'lr', 'lmbd'],
    lambda in_features, search_params: {
        'main_features': in_features if isinstance(in_features, int) else
            in_features[0],
        'main_n_components': search_params['n_cmp'],
        'gru_features': search_params['gru_sz'],
        'hidden_features': search_params['hid_sz'],
        'ctxt_features': None if isinstance(in_features, int) else
            in_features[1],
        'horizon': search_params['out_ch'],
        'rho': search_params['rho']},
    lambda search_params: {
        'batch_size': search_params['btch_sz'],
        'lr': search_params['lr'],
        'err': 'nll',
        'lmbd': search_params['lmbd']},
    lambda in_features, search_params: [search_params['n_cmp'] * in_features,
        search_params['n_cmp'] * in_features, search_params['n_cmp']] if
        isinstance(in_features, int) else [search_params['n_cmp'] *
        in_features[0], search_params['n_cmp'] * in_features[0], search_params[
        'n_cmp']],
    ['hidden'])
linear_params = BlockTypeParams('Linear', torch.nn.Linear,
    ['btch_sz', 'lr', 'lmbd'],
    lambda in_features, search_params, out_features: {
        'in_features': in_features,
        'out_features': out_features},
    lambda search_params: {
        'batch_size': search_params['btch_sz'],
        'lr': search_params['lr'],
        'err': 'ce',
        'lmbd': search_params['lmbd']})
upsample_linear_params = BlockTypeParams('UpsampleLinear', lrnn.UpsampleLinear,
    ['tmpk_st', 'up_pos', 'up_mod', 'btch_sz', 'lr', 'lmbd'],
    lambda in_features, search_params, out_features: {
        'scale_factor': search_params['tmpk_st'],
        'in_features': in_features,
        'out_features': out_features,
        'upsample_pos': search_params['up_pos'],
        'upsample_mode': search_params['up_mod']},
    lambda search_params: {
        'batch_size': search_params['btch_sz'],
        'lr': search_params['lr'],
        'err': 'ce',
        'lmbd': search_params['lmbd']})
shallow_rnn_params = BlockTypeParams('ShallowRNN', lrnn.ShallowRNN,
    ['hid_sz', 'btch_sz', 'lr', 'lmbd'],
    lambda in_features, search_params, out_features: {
        'in_features': in_features,
        'hidden_features': search_params['hid_sz'],
        'out_features': out_features},
    lambda search_params: {
        'batch_size': search_params['btch_sz'],
        'lr': search_params['lr'],
        'err': 'ce',
        'lmbd': search_params['lmbd']},
    None,
    ['hidden'])
upsample_rnn_params = BlockTypeParams('UpsampleRNN', lrnn.UpsampleRNN,
    ['tmpk_st', 'hid_sz', 'up_pos', 'up_mod', 'btch_sz', 'lr', 'lmbd'],
    lambda in_features, search_params, out_features: {
        'scale_factor': search_params['tmpk_st'],
        'in_features': in_features,
        'hidden_features': search_params['hid_sz'],
        'out_features': out_features,
        'upsample_pos': search_params['up_pos'],
        'upsample_mode': search_params['up_mod']},
    lambda search_params: {
        'batch_size': search_params['btch_sz'],
        'lr': search_params['lr'],
        'err': 'ce',
        'lmbd': search_params['lmbd']},
    None,
    ['hidden'])


block_type_params_list = [temp_max_pool_params, temp_window_params,
    downsample_params, upsample_params, multi_upsample_params,
    rect_gauss_mix_params, rect_gauss_mix_peak_params, cat_params,
    slice_params, trim_like_params, emulate_nonlinearity_params,
    function_params, attention_params, nilrnn_params, lrb_params,
    deep_lrb_params, lrb_predictor_params, gru_lrb_predictor_params,
    linear_params, upsample_linear_params, shallow_rnn_params,
    upsample_rnn_params]

class HLRNNParams:
    '''Model, optimization and search parameters of an HLRNN system.

    Atributes:
        in_features: A dictionary of strings to integers with the dataset input
            names and sizes.
        out_features: A dictionary of strings to integers with the dataset
            labeling names and sizes.
        block_types: A dictionary of strings to strings with the HLRNN block
            names and types forming the hierarchy.
        block_inputs: A dictionary of strings to strings or lists of strings
            with the HLRNN block names and input names.
        block_classifs: A dictionary of strings to strings with the supervised
            HLRNN block names and dataset labeling names.
        search_params: A dictionary of strings to search parameter space
            definitions with the names and space definitions (search library
            dependent) of the search parameters.
        param_constrs: A dictionary of tuples of strings to lists of tuples of
            strings with the search parameter identifying tuples that are
            constraint to the values of other parameters and the values, lists
            of multiplying model initialization parameter identifying tuples or
            functions defining the constraint parameters.
        search_blocks: A list of strings with the HLRNN block names with the
            last block of each search iteration (defaults to the block names in
            block_classifs).
        block_type_params: A list of the considered block type parameter names.
        block_extra_params: A list of the block type parameter names specific
            to an HLRNN block besides those corresponding to its type.
        block_model_params: A dictionary of strings to dictionaries of strings
            to anything with the HLRNN block names and model initialization
            parameter names and values.
        block_opt_params: A dictionary of strings to dictionaries of strings to
            anything with the HLRNN block names and optimization parameter
            names and values.
        block_out_features: A dictionary of strings to integers or lists of
            integers with the HLRNN block names and output sizes.
        next_block_names: A list of strings with the HLRNN block names of the
            next iteration.
        next_search_space: A dictionary of strings to search parameter space
            definition with the search parameter names and space definitions of
            the next iteration.
        _block_search_params: A dictionary of strings to dictionaries of
            strings to anything with the HLRNN block names and search parameter
            names and values.
        _unavail_block_names: A list of strings with the HLRNN block names
            whose model initialization and optimization parameters have not yet
            been set.
        _block_type_params: A dictionary of strings to block type parameter
            names with the HLRNN block type names and parameter names.
        _block_extra_params: A dictionary of strings to block type parameter
            names with the specific HLRNN block names and specific parameter
            names besides those corresponding to its type.
    '''
    def __init__(self, in_features: typing.OrderedDict[str, int], out_features:
        Dict[str, int], block_types: typing.OrderedDict[str, str],
        block_inputs: Dict[str, List[str]], block_classifs: Dict[str, str],
        search_params: Dict[str, Any], param_constrs: Dict[Tuple[str, str],
        Any] = {}, search_blocks: List[str] = None, block_type_params: List[
        BlockTypeParams] = block_type_params_list, block_extra_params: List[
        BlockTypeParams] = []) -> None:
        '''Inits HLRNNParams.

        Arguments:
            in_features: The dataset input names and sizes.
            out_features: The dataset labeling names and sizes.
            block_types: The HLRNN block names and types forming the hierarchy.
            block_inputs: The HLRNN block names and input names.
            block_classifs: The supervised HLRNN block names and dataset
                labeling names.
            search_params: The names and space definitions (search library
                dependent) of the search parameters.
            param_constrs: The search parameter identifying tuples that are
                constraint to the values of other parameters and the values,
                lists of multiplying model initialization parameter identifying
                tuples or functions defining the constraint parameters.
            search_blocks: The HLRNN block names with the last block of each
                search iteration (defaults to the block names in
                block_classifs).
            block_type_params: The considered HLRNN block type parameter names.
            block_extra_params: The block type parameter names specific to an
                HLRNN block besides those corresponding to its type.
        '''
        self.in_features, self.out_features = in_features, out_features
        self.block_types, self.block_inputs, self.block_classifs = \
            block_types, block_inputs, block_classifs
        self.search_params, self.param_constrs = search_params, param_constrs
        if search_blocks is None:
            search_blocks = [name for name in block_types if name in
                block_classifs]
        self.search_blocks = search_blocks
        self.block_type_params, self.block_extra_params = block_type_params, \
            block_extra_params
        self._block_type_params = {params.name: params for params in
            block_type_params}
        self._block_extra_params = {params.name: params for params in
            block_extra_params}
        self._block_search_params = {}
        self._init_block_params()

    def _init_block_params(self):
        '''Initializes the HLRNN block parameter dictionaries.'''
        self.block_out_features = {name: features for name, features in
            self.in_features.items()}
        for name, classif in self.block_classifs.items():
            self.block_out_features[name] = self.out_features[classif]
        self.block_model_params, self.block_opt_params, block_out_features = \
            self._get_block_params(self._block_search_params)
        self.block_out_features.update(block_out_features)
        self._unavail_block_names = list(self.block_types)[len(
            self.block_model_params):]
        self._set_next_iteration()

    def _get_block_params(self, block_search_params: Dict[str, Any]) -> \
        Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str,
        int]]:
        '''Returns the next iteration parameters given test search parameters.

        Arguments:
            block_search_params: The HLRNN block names and search parameter
                names and values.

        Returns:
            block_model_params: The HLRNN block names and model initialization
                parameter names and test values.
            block_opt_params: The HLRNN block names and optimization parameter
                names and test values.
            block_out_features: The HLRNN block names and test output sizes.
        '''
        block_model_params = {}
        block_opt_params = {}
        block_out_features = self.block_out_features.copy()
        for name in self.block_types:
            if name in block_search_params:
                block_input = self.block_inputs[name]
                if isinstance(block_input, str):
                    in_features = block_out_features[block_input]
                else:
                    in_feats_per_block = [block_out_features[block_in] for
                        block_in in block_input]
                    in_features = [in_feats for in_tuples in [(block_in_feats,)
                        if isinstance(block_in_feats, int) else block_in_feats
                        for block_in_feats in in_feats_per_block] for in_feats
                        in in_tuples]
                block_type_params = self._block_type_params[self.block_types[
                    name]]
                if name in block_out_features:
                    block_model_params[name] = \
                        block_type_params.get_model_params(in_features,
                        block_search_params[name], block_out_features[name])
                else:
                    block_model_params[name] = \
                        block_type_params.get_model_params(in_features,
                        block_search_params[name])
                    block_out_features[name] = \
                        block_type_params.get_out_features(in_features,
                        block_search_params[name])
                block_opt_params[name] = block_type_params.get_opt_params(
                    block_search_params[name])
                if name in self._block_extra_params:
                    block_extra_params = self._block_extra_params[name]
                    block_model_params[name].update(
                        block_extra_params.get_model_params(
                            block_search_params[name]))
                    block_opt_params[name].update(
                        block_extra_params.get_opt_params(block_search_params[
                        name]))
        block_out_features = {name: features for name, features in
            block_out_features.items() if name in block_search_params}
        return block_model_params, block_opt_params, block_out_features

    def _set_next_iteration(self) -> None:
        '''Sets the next block names, labeling name and search space.'''
        next_block_names = []
        next_params = set()
        for name in self._unavail_block_names:
            next_block_names.append(name)
            for param in self._block_type_params[self.block_types[
                name]].search_params:
                next_params.add((name, param))
            if name in self._block_extra_params:
                for param in self._block_extra_params[name].search_params:
                    next_params.add((name, param))
            if name in self.search_blocks:
                for param in self.param_constrs:
                    if param in next_params:
                        next_params.remove(param)
                self.next_block_names = next_block_names
                self.next_search_space = {name + '_' + param:
                    self.search_params[param] for name, param in next_params}
                break
        else:
            self.next_block_names = None
            self.next_search_space = None

    def _get_test_block_search_params(self, test_search_params: Dict[str,
        float]) -> Dict[str, Dict[str, Any]]:
        '''Returns the next iteration given test search parameters per block.

        Arguments:
            test_search_params: The search parameter names and test values for
                the next iteration.

        Returns:
            test_block_search_params: The HLRNN block names and search
                parameter names and values for the next iteration.
        '''
        test_search_params = test_search_params.copy()
        block_search_params = self._block_search_params.copy()
        for name in self.next_block_names:
            for param, constr in self.param_constrs.items():
                if name == param[0]:
                    if callable(constr):
                        value = constr(block_search_params)
                    elif isinstance(constr, list):
                        value = 1
                        for mult in constr:
                            if isinstance(mult, (int, float)):
                                value *= mult
                            else:
                                if name == mult[0]:
                                    value *= test_search_params['_'.join(mult)]
                                else:
                                    value *= block_search_params[mult[0]][mult[
                                        1]]
                    else:
                        value = constr
                    test_search_params['_'.join(param)] = value
            block_search_params[name] = {param: test_search_params[
                f'{name}_{param}'] for param in self._block_type_params[
                self.block_types[name]].search_params}
            if name in self._block_extra_params:
                block_search_params[name].update({param: test_search_params[
                    f'{name}_{param}'] for param in self._block_extra_params[
                    name].search_params})
        return {name: param for name, param in block_search_params.items() if
            name in self.next_block_names}

    def get_test_block_params(self, test_search_params: Dict[str, float]) -> \
        Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        '''Returns the next iteration parameters given test search parameters.

        Arguments:
            test_search_params: The search parameter names and test values for
                the next iteration.

        Returns:
            test_block_model_params: The HLRNN block names and model
                initialization parameter names and test values for the next
                iteration.
            test_block_opt_params: The HLRNN block names and optimization
                parameter names and test values for the next iteration.
        '''
        test_block_search_params = self._get_test_block_search_params(
            test_search_params)
        test_block_model_params, test_block_opt_params, _ = \
            self._get_block_params(test_block_search_params)
        return test_block_model_params, test_block_opt_params

    def get_test_blocks(self, test_search_params: Dict[str, float]) -> Tuple[
        typing.OrderedDict[str, torch.nn.Module], Dict[str, List[str]], List[
        str], Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
        '''Returns the next iteration blocks given the test search parameters.

        Arguments:
            test_search_params: The search parameter names and test values for
                the next iteration.

        Returns:
            test_blocks: The HLRNN block names and test blocks for the next
                iteration.
            test_block_inputs: The HLRNN block names and input names of the
                next iteration.
            test_hidden_names: The names of the HLRNN hidden states of the next
                iteration.
            test_block_hiddens: The HLRNN block names and hidden state names of
                the next iteration.
            test_block_opt_params: The HLRNN block names and optimization
                parameter names and test values for the next iteration.
        '''
        test_block_model_params, test_block_opt_params = \
            self.get_test_block_params(test_search_params)
        test_blocks = OrderedDict([(name, self._block_type_params[
            self.block_types[name]].block_class(**test_block_model_params[
            name])) for name in self.next_block_names])
        test_block_inputs = {name: self.block_inputs[name] for name in
            self.next_block_names}
        test_hidden_names, test_block_hiddens = [], {}
        for name in self.next_block_names:
            hiddens = [f'{name}_{hidden}' for hidden in
                self._block_type_params[self.block_types[name]].hiddens]
            test_hidden_names.extend(hiddens)
            test_block_hiddens[name] = hiddens
        return test_blocks, test_block_inputs, test_hidden_names, \
            test_block_hiddens, test_block_opt_params

    def update_block_params(self, search_params: Dict[str, float]) -> None:
        '''Updates the parameters given the next iteration search parameters.

        Arguments:
            search_params: The search parameter names and values of the next
                iteration.
        '''
        next_block_search_params = self._get_test_block_search_params(
            search_params)
        self._block_search_params.update(next_block_search_params)
        next_block_model_params, next_block_opt_params, \
            next_block_out_features = self._get_block_params(
            next_block_search_params)
        self.block_model_params.update(next_block_model_params)
        self.block_opt_params.update(next_block_opt_params)
        self.block_out_features.update(next_block_out_features)
        del self._unavail_block_names[:len(next_block_model_params)]
        self._set_next_iteration()

    def update_last_block_params(self) -> None:
        '''Updates the last block parameters that not require a search.'''
        self.next_block_names = self._unavail_block_names
        self.update_block_params({name: {} for name in
            self._unavail_block_names})

    def extend(self, block_types: typing.OrderedDict[str, str], block_inputs:
        Dict[str, List[str]], block_classifs: Dict[str, str], param_constrs:
        Dict[Tuple[str, str], Any] = {}, search_blocks: List[str] = None,
        block_extra_params: List[BlockTypeParams] = []) -> None:
        '''Extends the hierarchy with the given blocks.

        Arguments:
            block_types: The HLRNN block names and types forming the hierarchy.
            block_inputs: The HLRNN block names and input names.
            block_classifs: The supervised HLRNN block names and dataset
                labeling names.
            param_constrs: The search parameter identifying tuples that are
                constraint to the values of other parameters and the values,
                lists of multiplying model initialization parameter identifying
                tuples or functions defining the constraint parameters.
            search_blocks: The HLRNN block names with the last block of each
                search iteration (defaults to the block names in
                block_classifs).
            block_extra_params: The block type parameter names specific to an
                HLRNN block besides those corresponding to its type.
        '''
        self.block_types.update(block_types)
        self.block_inputs.update(block_inputs)
        self.block_classifs.update(block_classifs)
        self.param_constrs.update(param_constrs)
        if search_blocks is None:
            search_blocks = [name for name in block_types if name in
                block_classifs]
        self.search_blocks.extend(search_blocks)
        self.block_extra_params.extend(block_extra_params)
        self._block_extra_params.update({params.name: params for params in
            block_extra_params})
        self._init_block_params()

    def extend_input(self, in_features: typing.OrderedDict[str, int]) -> None:
        '''Extends the inputs to the hierarchy with the given inputs.

        Arguments:
            in_features: The new dataset input names and sizes.
        '''
        self.in_features.update(in_features)
        self.block_out_features.update(in_features)

    def state_dict(self) -> Dict[str, Any]:
        '''Returns a dictionary containing a whole state of the module.

        Returns:
            state_dict: The state of the module.
        '''
        return {'block_search_params': self._block_search_params}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        '''Copies parameters and buffers from state_dict into this module.

        Arguments:
            state_dict: The state of the module.
        '''
        self._block_search_params = state_dict['block_search_params']
        self._init_block_params()

    def get_hlrnn(self, model_state: Dict[str, Any] = None) -> lrnn.HLRNN:
        '''Returns the HLRNN corresponding to the available parameters.

        Arguments:
            model_state: The model state.

        Returns:
            hlrnn: The stack of neural network blocks with the available model
                parameters.
        '''
        block_names = list(self.block_types)
        avail_block_names = block_names[:len(block_names) - len(
            self._unavail_block_names)]
        blocks = OrderedDict([(name, self._block_type_params[self.block_types[
            name]].block_class(**self.block_model_params[name])) for name in
            avail_block_names])
        block_inputs = {name: self.block_inputs[name] for name in
            avail_block_names}
        hidden_names, block_hiddens = [], {}
        for name in avail_block_names:
            hiddens = [f'{name}_{hidden}' for hidden in
                self._block_type_params[self.block_types[name]].hiddens]
            hidden_names.extend(hiddens)
            block_hiddens[name] = hiddens
        model = lrnn.HLRNN(blocks, list(self.in_features), block_inputs,
            hidden_names, block_hiddens)
        if model_state is not None:
            model.load_state_dict(model_state)
        return model


class BlockTypeEval:
    '''Initialization, train and evaluation functions for an HLRNN block type.

     Atributes:
        name: A string with the HLRNN block type name.
        get_loss_fns: A function that obtains the loss function and error
            function from the HLRNN block and the optimization parameters.
        train: A function that trains the HLRNN block and obtains the running
            average loss from the dataloader, the HLRNN model, the loss
            function, the optimizer, the number of batches and a function to
            arrange the model input.
        test: A function that tests the HLRNN block and obtains the estimated
            metrics from the dataloader, the HLRNN model, the loss function,
            the number of batches and a function to arrange the model input.
    '''
    def __init__(self, name: str, get_loss_fns: Callable[[torch.nn.Module,
        Dict[str, Any]], Tuple[Callable[[Any], torch.Tensor]]] = None, train:
        Callable[[torch.utils.data.DataLoader, lrnn.HLRNN, Callable[[Any],
        torch.Tensor], torch.optim.Optimizer, int, Callable[[List[
        torch.Tensor]], List[torch.Tensor]]], Tuple[float, int]] = None, test:
        Callable[[torch.utils.data.DataLoader, lrnn.HLRNN, Callable[[Any],
        torch.Tensor], Callable[[Any], torch.Tensor], int, Callable[[List[
        torch.Tensor]], List[torch.Tensor]]], Tuple[float, int, Dict[str,
        float]]] = None) -> None:
        '''Inits BlockTypeEval.

        Arguments:
            name: The HLRNN block type name.
            get_loss_fns: Obtains the loss function and error function from the
                HLRNN block and the optimization parameters.
            train: Trains the HLRNN block and obtains the running average loss
                from the dataloader, the HLRNN model, the loss function, the
                optimizer, the number of batches and a function to arrange the
                model input.
            test: Tests the HLRNN block and obtains the estimated metrics from
                the dataloader, the HLRNN model, the loss function, the number
                of batches and a function to arrange the model input.
        '''
        self.name = name
        self.get_loss_fns, self.train, self.test = get_loss_fns, train, test

def _get_loss_fns(block, opt_params):
    weight = None
    if 'err_w' in opt_params:
        weights = []
        sum_features = sum(opt_params['errw_ft'])
        for w, features in zip(opt_params['err_w'], opt_params['errw_ft']):
            weights.append(w * sum_features / features * torch.ones(features))
        weight = torch.cat(weights)
        if 'out_channels' in opt_params:
            weight = weight.repeat(opt_params['out_channels'])
    if hasattr(block, 'get_mask'):
        if weight is None:
            get_weight_fn = block.get_mask
        else:
            get_weight_fn = lambda batch_size: weight * model.get_mask(
                batch_size)
    else:
        if weight is None:
            get_weight_fn = None
        else:
            get_weight_fn = lambda batch_size: weight
    if opt_params['err'] == 'mse':
        err_fn = lrnn.MSELoss(get_weight_fn)
    elif opt_params['err'] == 'l1':
        err_fn = lrnn.L1Loss(get_weight_fn)
    elif opt_params['err'] == 'bce':
        err_fn = lambda pred, y: \
            torch.nn.functional.binary_cross_entropy_with_logits(pred, y,
            weight=get_weight_fn(pred.shape[0]), reduction = 'sum') / \
            pred.shape[0]
    elif opt_params['err'] == 'nll':
        err_fn = lrnn.RectGaussMixNLLLoss(get_weight_fn)
    elif opt_params['err'] == 'ce':
        if get_weight_fn is None:
            err_fn = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError('The functionality of weighting the '
                'samples in the loss function has been only implemented for '
                'unsupervised blocks')
    else:
        raise ValueError(f'{opt_params["err"]} is not a valid cost error term')
    loss_fns = []
    loss_ws = []
    if 'lmbd' in opt_params and opt_params['lmbd'] != 0:
        loss_fns.append(lrnn.WeightDecayLoss(block))
        loss_ws.append(opt_params['lmbd'])
    if 'beta' in opt_params and opt_params['beta'] != 0:
        loss_fns.append(lrnn.SparsityLoss(opt_params['rho'],
        block.sparse_vbles, block.sparse_nonlinearity))
        loss_ws.append(opt_params['beta'])
    if 'nbeta' in opt_params and opt_params['nbeta'] != 0:
        loss_fns.append(lrnn.NormSparsityLoss(block.norm_sparse_vbles))
        loss_ws.append(opt_params['nbeta'])
    if opt_params['err'] == 'nll':
        loss_fn = lambda mn, vr, p, y: err_fn(mn, vr, p, y) + sum(w * fn() for
            fn, w in zip(loss_fns, loss_ws))
        error_fn = err_fn
    elif 'contr' in opt_params and opt_params['contr']:
        slowness_loss_fn = lrnn.SlownessLoss(opt_params['delta'])
        loss_fn = lambda pred, pred2, y: err_fn(pred, y) + opt_params[
            'gamma'] * slowness_loss_fn(pred2) + sum(w * fn() for fn, w in zip(
            loss_fns, loss_ws))
        error_fn = lambda pred, _, y: err_fn(pred, y)
    else:
        loss_fn = lambda pred, y: err_fn(pred, y) + sum(w * fn() for fn, w in
            zip(loss_fns, loss_ws))
        error_fn = err_fn
    return loss_fn, error_fn

def _train_unsupervised_block(dataloader, model, loss_fn, opt, num_samples,
    preprocess_x, keep_hiddens):
    num_end_samples = num_samples / 10
    #cur_h_names = model.block_hiddens[model.cur_block_name]
    #cur_h_is = [model.hidden_names.index(cur_h_names)] if cur_h_names is str \
    #    else [model.hidden_names.index(h) for h in cur_h_names]
    model.train()
    hs = [None] * len(model.hidden_names)
    sample, end_loss, end_batches = 0, 0, 0
    for *x, _ in dataloader:
        #x = tuple(c.to(device) for c in x)
        output = model(*preprocess_x(x), *hs)
        if keep_hiddens:
            hs = [h.detach() if h is not None else None for h in output[len(
                output) - len(hs):]]
            #if sample < num_samples / 2:
            #    hs = [None if i in cur_h_is else h for i, h in enumerate(hs)]
        loss = loss_fn(*output[:len(output) - len(hs)])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sample += x[0].shape[0]
        if sample >= num_samples - num_end_samples:
            end_loss += loss.item()
            end_batches += 1
        if sample >= num_samples:
            break
    end_loss /= end_batches
    return end_loss, sample

def _train_supervised_block(dataloader, model, loss_fn, opt, num_samples,
    preprocess_x, keep_hiddens):
    num_end_samples = num_samples / 10
    model.train()
    hs = [None] * len(model.hidden_names)
    sample, end_loss, end_batches = 0, 0, 0
    for *x, y in dataloader:
        #x, y = tuple(c.to(device) for c in x), y.to(device)
        pred = model(*preprocess_x(x), *hs)
        if len(hs) != 0:
            if keep_hiddens:
                hs = [h.detach() if h is not None else None for h in pred[1:]]
            pred = pred[0]
        pred = pred[:y.shape[0]]
        labeled = y >= 0
        if not labeled.all():
            if not labeled.any():
                continue
            pred = pred[labeled]
            y = y[labeled]
        loss = loss_fn(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sample += y.shape[0]
        if sample >= num_samples - num_end_samples:
            end_loss += loss.item()
            end_batches += 1
        if sample >= num_samples:
            break
    end_loss /= end_batches
    return end_loss, sample

def _test_unsupervised_block(dataloader, model, loss_fn, error_fn, num_samples,
    preprocess_x, keep_hiddens):
    model.train()
    hs = [None] * len(model.hidden_names)
    sample, batch, loss, error = 0, 0, 0, 0
    with torch.no_grad():
        for *x, _ in dataloader:
            #x = tuple(c.to(device) for c in x)
            output = model(*preprocess_x(x), *hs)
            if keep_hiddens:
                hs = [h.detach() if h is not None else None for h in output[
                    len(output) - len(hs):]]
            sample += x[0].shape[0]
            batch += 1
            loss += loss_fn(*output[:len(output) - len(hs)]).item()
            error += error_fn(*output[:len(output) - len(hs)]).item()
            if sample >= num_samples:
                break
    loss /= batch
    metrics = {'error': error / batch}
    return loss, sample, metrics

def _test_supervised_block(dataloader, model, loss_fn, error_fn, num_samples,
    preprocess_x, keep_hiddens):
    model.eval()
    hs = [None] * len(model.hidden_names)
    sample, batch, loss, error, acc = 0, 0, 0, 0, 0
    out_features = model.cur_block.out_features
    tp, fp_fn_2tp = torch.zeros(out_features), torch.zeros(out_features)
    with torch.no_grad():
        for *x, y in dataloader:
            #x, y = tuple(c.to(device) for c in x), y.to(device)
            pred = model(*preprocess_x(x), *hs)
            if len(hs) != 0:
                if keep_hiddens:
                    hs = [h.detach() if h is not None else None for h in pred[
                        1:]]
                pred = pred[0]
            pred = pred[:y.shape[0]]
            labeled = y >= 0
            if not labeled.all():
                if not labeled.any():
                    continue
                pred = pred[labeled]
                y = y[labeled]
            sample += y.shape[0]
            batch += 1
            loss += loss_fn(pred, y).item()
            error += error_fn(pred, y).item()
            y_ = pred.argmax(1)
            correct = y_ == y
            acc += correct.type(torch.float).sum().item()
            tp += torch.bincount(y[correct], minlength=out_features)
            fp_fn_2tp += torch.bincount(y, minlength=out_features) + \
                torch.bincount(y_, minlength=out_features)
            if sample >= num_samples:
                break
    loss /= batch
    metrics = {'error': error / batch, 'acc': acc / sample, 'f1': 2 * (tp[
        fp_fn_2tp > 0] / fp_fn_2tp[fp_fn_2tp > 0]).mean().item()}
    return loss, sample, metrics

temp_max_pool_eval = BlockTypeEval('TempMaxPool')
temp_window_eval = BlockTypeEval('TempWindow')
downsample_eval = BlockTypeEval('Downsample')
upsample_eval = BlockTypeEval('Upsample')
multi_upsample_eval = BlockTypeEval('MultiUpsample')
rect_gauss_mix_eval = BlockTypeEval('RectGaussMix')
rect_gauss_mix_peak_eval = BlockTypeEval('RectGaussMixPeak')
cat_eval = BlockTypeEval('Cat')
slice_eval = BlockTypeEval('Slice')
trim_like_eval = BlockTypeEval('TrimLike')
emulate_nonlinearity_eval = BlockTypeEval('EmulateNonlinearity')
function_eval = BlockTypeEval('Function')
attention_eval = BlockTypeEval('Attention', _get_loss_fns,
    _train_unsupervised_block, _test_unsupervised_block)
nilrnn_eval = BlockTypeEval('NILRNN', _get_loss_fns, _train_unsupervised_block,
    _test_unsupervised_block)
lrb_eval = BlockTypeEval('LRB', _get_loss_fns, _train_unsupervised_block,
    _test_unsupervised_block)
deep_lrb_eval = BlockTypeEval('DeepLRB', _get_loss_fns,
    _train_unsupervised_block, _test_unsupervised_block)
lrb_predictor_eval = BlockTypeEval('LRBPredictor', _get_loss_fns,
    _train_unsupervised_block, _test_unsupervised_block)
gru_lrb_predictor_eval = BlockTypeEval('GRULRBPredictor', _get_loss_fns,
    _train_unsupervised_block, _test_unsupervised_block)
linear_eval = BlockTypeEval('Linear', _get_loss_fns, _train_supervised_block,
    _test_supervised_block)
upsample_linear_eval = BlockTypeEval('UpsampleLinear', _get_loss_fns,
    _train_supervised_block, _test_supervised_block)
shallow_rnn_eval = BlockTypeEval('ShallowRNN', _get_loss_fns,
    _train_supervised_block, _test_supervised_block)
upsample_rnn_eval = BlockTypeEval('UpsampleRNN', _get_loss_fns,
    _train_supervised_block, _test_supervised_block)


block_type_eval_list = [temp_max_pool_eval, temp_window_eval, downsample_eval,
    upsample_eval, multi_upsample_eval, rect_gauss_mix_eval,
    rect_gauss_mix_peak_eval, cat_eval, slice_eval, trim_like_eval,
    emulate_nonlinearity_eval, function_eval, attention_eval, nilrnn_eval,
    lrb_eval, deep_lrb_eval, lrb_predictor_eval, gru_lrb_predictor_eval,
    linear_eval, upsample_linear_eval, shallow_rnn_eval, upsample_rnn_eval]

class HLRNNBlockEval:
    '''Initialization, train and evaluation functions for an HLRNN block.

    Atributes:
        model: An HLRNN with the stack of neural network blocks.
        block_name: A string with the name of the HLRNN block.
        block_type: A string with the type of the HLRNN block.
        dataset: A dataset with the input data.
        classif: A string with the dataset labeling name (if supervised).
        opt_params: A dictionary of strings to anything with the optimization
            parameter names and values.
        train_num_samples: An integer with the number of samples with which the
            HLRNN block has been trained.
        preprocess_x: A function that preprocesses and arranges the input from
            the dataset to be ready for the model.
        keep_hiddens: A boolean indicating whether the hidden states are kept
            accross successive batches.
        must_trial_lbls: A dictionary of strings to lists of strings with the
            trial classifications and mandatory labels.
        block_type_eval_list: A list of the considered block type evaluation
            functions.
        train_loss: A float with the running average loss obtained during the
            last training.
        _dataloader: A dataloader with the input data for the current HLRNN
            block.
        _loss_fn: A function that obtains the loss from the predicted and the
            target output.
        _error_fn: A function that obtains the loss from the predicted and the
            target output (the first term of the loss function).
        _opt: An optimizer for the current HLRNN block weights.
        _train: A function that trains the HLRNN block and obtains the running
            average loss from the dataloader, the HLRNN model, the loss
            function, the optimizer, the number of batches and a function to
            arrange the model input.
        _test: A function that tests the HLRNN block and obtains the estimated
            accuracy and loss from the dataloader, the HLRNN model, the loss
            function, the number of batches and a function to arrange the model
            input.
    '''
    def __init__(self, model: lrnn.HLRNN, block_name: str, block_type: str,
        dataset: torch.utils.data.Dataset, classif: str, opt_params: Dict[str,
        Any], train_num_samples: int = 0, opt_state: Dict[str, Any] = None,
        preprocess_x: Callable[[List[torch.Tensor]], List[torch.Tensor]] =
        None, keep_hiddens: bool = True, must_trial_lbls: Dict[str, List[str]]
        = None, block_type_eval_list: List[BlockTypeEval] =
        block_type_eval_list) -> None:
        '''Inits HLRNNBlockEval.

        Arguments:
            model: The stack of neural network blocks.
            block_name: The name of the HLRNN block.
            block_type: The type of the HLRNN block.
            dataset: The input data.
            classif: The dataset labeling name (if supervised).
            opt_params: The optimization parameter names and values.
            train_num_samples: The number of samples with which the HLRNN block
                has been trained.
            opt_state: The optimizer state.
            preprocess_x: Preprocesses and arranges the input from the dataset
                to be ready for the model.
            keep_hiddens: If true, the hidden states of all blocks are kept
                accross successive batches.
            must_trial_lbls: The mandatory labels per trial classification.
            block_type_eval_list: The considered HLRNN block type evaluation
                functions.
        '''
        self.model, self.block_name, self.block_type = model, block_name, \
            block_type
        self.dataset, self.classif = dataset, classif
        self.opt_params = opt_params
        self.train_num_samples = train_num_samples
        self.preprocess_x = preprocess_x
        self.keep_hiddens = keep_hiddens
        self.must_trial_lbls = must_trial_lbls
        self.block_type_eval_list = block_type_eval_list
        self.train_loss = None
        self.model.set_cur_block(block_name)
        for type_eval in block_type_eval_list:
            if block_type == type_eval.name:
                block_type_eval = type_eval
                break
        else:
            raise TypeError(f'{type(model.cur_block)} is not among the '
                'considered block types')
        self._dataloader = None
        self._loss_fn, self._error_fn = None, None
        self._opt = None
        if block_type_eval.get_loss_fns is not None:
            if 'seq_size' in opt_params:
                seq_size = round(math.ceil(opt_params['seq_size'] *
                    model.cur_min_scale_factor / model.train_scale_factor) /
                    model.cur_min_scale_factor)
                if preprocess_x is None:
                    self.preprocess_x = lambda x: tuple(c.reshape(opt_params[
                    'batch_size'], seq_size, *c.shape[1:]) for c in x)
                else:
                    self.preprocess_x = partial(preprocess_x, seq_size=
                        seq_size)
                batch_size = opt_params['batch_size'] * seq_size
            else:
                if preprocess_x is None:
                    self.preprocess_x = lambda x: x
                batch_size = round(math.ceil(opt_params['batch_size'] *
                    model.cur_min_scale_factor / model.train_scale_factor) /
                    model.cur_min_scale_factor)
            if hasattr(dataset, 'set_batch_iter'):
                dataset.set_batch_iter(batch_size)
                batch_size = None
            self._dataloader = torch.utils.data.DataLoader(dataset,
                    batch_size=batch_size)
            self._loss_fn, self._error_fn = block_type_eval.get_loss_fns(
                model.cur_block, opt_params)
            self.init_opt()
            if opt_state is not None:
                self._opt.load_state_dict(opt_state)
        self._train = block_type_eval.train
        self._test = block_type_eval.test

    def init_opt(self):
        '''Initializes the optimizer.'''
        self._opt = torch.optim.Adam(self.model.blocks[
            self.block_name].parameters(), lr=self.opt_params['lr'])

    def restart(self):
        '''Restarts the weights and the optimizer.'''
        self.model.blocks[self.block_name].reset_parameters()
        self.init_opt()

    def train(self, num_samples: int):
        '''Trains the HLRNN block for the indicated number of samples.

        Arguments:
            num_samples: The number of training samples.
        '''
        if self._train is not None:
            self.model.set_cur_block(self.block_name)
            if self.classif is not None:
                self.dataset.set_classif(self.classif)
            if hasattr(self.dataset, 'set_must_classifs'):
                if self.classif is None:
                    self.dataset.set_must_classifs([])
                else:
                    self.dataset.set_must_classifs([self.classif])
            if self.must_trial_lbls is not None:
                self.dataset.set_must_trial_lbls(self.must_trial_lbls)
            self.train_loss, train_num_samples = self._train(self._dataloader,
                self.model, self._loss_fn, self._opt, num_samples,
                self.preprocess_x, self.keep_hiddens)
            self.train_num_samples += train_num_samples

    def test(self, num_samples: int) -> Tuple[float, int, Dict[str, float]]:
        '''Tests the HLRNN block for the indicated number of samples.

        Arguments:
            num_samples: The number of testing samples.

        Returns:
            loss: The estimated loss.
            num_samples: The actual number of samples used.
            metrics: The estimated metrics.
        '''
        self.model.set_cur_block(self.block_name)
        if self.classif is not None:
            self.dataset.set_classif(self.classif)
        if hasattr(self.dataset, 'set_must_classifs'):
            if self.classif is None:
                self.dataset.set_must_classifs([])
            else:
                self.dataset.set_must_classifs([self.classif])
        if self.must_trial_lbls is not None:
            self.dataset.set_must_trial_lbls(self.must_trial_lbls)
        return self._test(self._dataloader, self.model, self._loss_fn,
            self._error_fn, num_samples, self.preprocess_x, self.keep_hiddens)

    def get_opt_state(self) -> Dict[str, Any]:
        '''Returns the optimizer state dictionary.

        Returns:
            opt_state: the optimizer state.
        '''
        if self._opt is not None:
            return self._opt.state_dict()
        return None


class HLRNNTrialEval:
    '''Initialization and evaluation functions for an HLRNN search trial.

    Atributes:
        model_params: HLRNN parameters with the model initialization and
            optimization parameters.
        search_params: A dictionary of strings to anything with the search
            parameter names and values of this trial.
        dataset: A dataset with the input data.
        preprocess_x: A function that preprocesses and arranges the input from
            the dataset to be ready for the model.
        keep_hiddens: A boolean indicating whether the hidden states are kept
            accross successive batches.
        block_type_eval_list: A list of the considered block type evaluation
            functions.
        model: An HLRNN with the stack of neural network blocks.
        block_names: A list of the HLRNN block names of this trial.
        block_evals: A list of the HLRNN block evaluation functions of this
            trial.
    '''
    def __init__(self, model_params: HLRNNParams, search_params: Dict[str,
        Any], dataset: torch.utils.data.Dataset, prev_model_state: Dict[str,
        Any] = None, model_state: Dict[str, Any] = None, opt_states: Dict[str,
        Dict[str, Any]] = None, train_num_samples: Dict[str, int] = None,
        preprocess_x: Callable[[List[torch.Tensor]], List[torch.Tensor]] =
        None, keep_hiddens: bool = True, must_trial_lbls: Dict[str, Dict[str,
        List[str]]] = None, block_type_eval_list: List[BlockTypeEval] =
        block_type_eval_list) -> None:
        '''Inits HLRNNTrialEval.

        Arguments:
            model_params: The model initialization and optimization parameters.
            search_params: The search parameter names and values of this trial.
            dataset: The input data.
            prev_model_state: The model state at the previous iteration.
            model_state: The model state.
            opt_states: The optimizer states for each HLRNN block.
            train_num_samples: The number of samples with which each HLRNN
                block has been trained.
            preprocess_x: Preprocesses and arranges the input from the dataset
                to be ready for the model.
            keep_hiddens: If true, the hidden states are kept accross
                successive batches.
            must_trial_lbls: The mandatory labels per trial classification for
                each HLRNN block.
            block_type_eval_list: The considered block type evaluation
                functions.
        '''
        self.model_params, self.search_params = model_params, search_params
        self.dataset = dataset
        self.preprocess_x = preprocess_x
        self.keep_hiddens = keep_hiddens
        self.block_type_eval_list = block_type_eval_list
        self.model = model_params.get_hlrnn(prev_model_state)#.to(device)
        blocks, block_inputs, hidden_names, block_hiddens, opt_params = \
            model_params.get_test_blocks(search_params)
        self.model.extend(blocks, block_inputs, hidden_names, block_hiddens)
        if model_state is not None:
            self.model.load_state_dict(model_state)
        self.block_names = list(blocks)
        classifs = {name: model_params.block_classifs[name] if name in
            model_params.block_classifs else None for name in self.block_names}
        if train_num_samples is None:
            train_num_samples = {name: 0 for name in self.block_names}
        if opt_states is None:
            opt_states = {name: None for name in self.block_names}
        if must_trial_lbls is None:
            must_trial_lbls = {}
        must_trial_lbls = {name: must_trial_lbls[name] if name in
            must_trial_lbls else None for name in self.block_names}
        self.block_evals = [HLRNNBlockEval(self.model, name,
            model_params.block_types[name], dataset, classifs[name],
            opt_params[name], train_num_samples[name], opt_states[name],
            preprocess_x, keep_hiddens, must_trial_lbls[name],
            block_type_eval_list) for name in self.block_names]

    def train(self, num_samples: Dict[str, int]):
        '''Trains the HLRNN blocks for the indicated number of samples.

        Arguments:
            num_samples: The HLRNN block names and corresponding number of
                training samples.
        '''
        self.dataset.restart({'splitset': 0})
        for block_eval in self.block_evals:
            block_eval.train(num_samples[block_eval.block_name])

    def test(self, num_samples: Dict[str, int]) -> Dict[str, Tuple[float, int,
        Dict[str, float]]]:
        '''Tests the last HLRNN block for the indicated number of samples.

        Arguments:
            num_samples: The HLRNN block names and corresponding number of
                testing samples.

        Returns:
            result: The HLRNN block names and corresponding estimated loss,
                actual number of samples used and estimated metrics.
        '''
        self.dataset.restart({'splitset': 1})
        result = {}
        for name, n_samples in num_samples.items():
            loss, n_samples, metrics = self.block_evals[self.block_names.index(
                name)].test(n_samples)
            result[name] = {'loss': loss, 'num_samples': n_samples, 'metrics':
                metrics}
        return result

    def get_state(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]],
        Dict[str, int]]:
        '''Returns the current blocks and optimizers state dictionary.

        Returns:
            blocks_state: The part of the model state corresponding to the
                current blocks.
            opt_states: The optimizer states for each HLRNN block.
            train_num_samples: The number of samples with which each HLRNN
                block has been trained.
        '''
        model_state, opt_states, train_num_samples = self.get_full_state()
        blocks_state = {key: model_state[key] for key in model_state if
            key.startswith(tuple(name + '.' for name in self.block_names))}
        return blocks_state, opt_states, train_num_samples

    def get_full_state(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str,
        Any]], Dict[str, int]]:
        '''Returns the current model and optimizers state dictionary.

        Returns:
            model_state: The model state.
            opt_states: The optimizer states for each HLRNN block.
            train_num_samples: The number of samples with which each HLRNN
                block has been trained.
        '''
        self.model.unset_cur_block()
        model_state = self.model.state_dict()
        opt_states = {name: block_eval.get_opt_state() for name, block_eval in
            zip(self.block_names, self.block_evals)}
        train_num_samples = {name: block_eval.train_num_samples for name,
            block_eval in zip(self.block_names, self.block_evals)}
        return model_state, opt_states, train_num_samples


def main(dataset_name: str = 'SynthPlanInput') -> None:
    '''Gradually builds and optimizes an HLRNN.

    Arguments:
        dataset_name: The name of the dataset.
    '''
    dataset_class = getattr(inputdata, dataset_name)
    dataset = dataset_class({'splitprops': [0.6, 0.2, 0.2], 'splitset': 0})
    #device = "cuda" if torch.cuda.is_available() else "cpu"

    search_config = {'num_trials': 4, 'metric': 'acc', 'metric_mode': 'max'}
    trial_config = {'dataset_class': dataset_class, 'num_unsup_samples': 3000,
        'num_sup_samples': 2000, 'num_test_samples': 5000}
    test_config = {'num_trials': 2}
    test_trial_config = {'dataset_class': dataset_class, 'block_name': 'rec3',
        'num_samples': 5000, 'num_steps': 5, 'num_test_samples': 5000}
    in_features = OrderedDict([('input', dataset.size)])
    out_features = {classif: len(labels) for classif, labels in zip(
        dataset.classifs, dataset.all_lbls)}
    block_types = OrderedDict([#('lrb1', 'LRB'),
                               ('tmpl0', 'TempMaxPool'),
                               ('lrb1', 'NILRNN'),
                               ('upsmp1', 'Upsample'),
                               ('rec1', 'ShallowRNN'),
                               #('lrb2', 'LRB'),
                               ('tmpl1', 'TempMaxPool'),
                               ('lrb2', 'NILRNN'),
                               ('upsmp2', 'Upsample'),
                               ('rec2', 'ShallowRNN'),
                               ('tmpl01', 'TempMaxPool'),
                               ('cat2', 'Cat'),
                               #('lrb3', 'LRB'),
                               ('tmpl2', 'TempMaxPool'),
                               ('lrb3', 'NILRNN'),
                               ('upsmp3', 'Upsample'),
                               ('rec3', 'ShallowRNN')])
    block_inputs = {#'lrb1': 'input',
                    'tmpl0': 'input',
                    'lrb1': 'tmpl0',
                    'upsmp1': 'lrb1',
                    'rec1': 'upsmp1',
                    #'lrb2': 'lrb1',
                    'tmpl1': 'lrb1',
                    'lrb2': 'tmpl1',
                    'upsmp2': 'lrb2',
                    'rec2': 'upsmp2',
                    'tmpl01': 'input',
                    'cat2': ['lrb2', 'tmpl01'],
                    #'lrb3': 'cat2',
                    'tmpl2': 'cat2',
                    'lrb3': 'tmpl2',
                    'upsmp3': 'lrb3',
                    'rec3': 'upsmp3'}
    block_classifs = {'rec1': dataset.classifs[0],
                      'rec2': dataset.classifs[0],
                      'rec3': dataset.classifs[0]}
    param_constrs = {('lrb1', 'k_sp'): 'circular',
                     ('lrb2', 'k_sp'): 'circular',
                     ('lrb3', 'k_sp'): 'circular',
                     ('lrb1', 'loc_nl'): 'sigm',
                     ('lrb2', 'loc_nl'): 'sigm',
                     ('lrb3', 'loc_nl'): 'sigm',
                     ('lrb1', 'out_nl'): 'sigm',
                     ('lrb2', 'out_nl'): 'sigm',
                     ('lrb3', 'out_nl'): 'sigm',
                     ('lrb1', 'err'): 'mse',
                     ('lrb2', 'err'): 'mse',
                     ('lrb3', 'err'): 'mse',
                     #('lrb1', 't_pos'): 'first',
                     #('lrb2', 't_pos'): 'first',
                     #('lrb3', 't_pos'): 'first',
                     #('lrb1', 'tmpk_ct'): False,
                     ('tmpl0', 'tmpk_ct'): False,
                     #('lrb2', 'tmpk_ct'): False,
                     ('tmpl1', 'tmpk_ct'): False,
                     ('tmpl01', 'tmpk_ct'): False,
                     #('lrb3', 'tmpk_ct'): False,
                     ('tmpl2', 'tmpk_ct'): False,
                     ('lrb1', 'out_ft'): True,
                     ('lrb2', 'out_ft'): True,
                     ('lrb3', 'out_ft'): True,
                     #('tmpl2', 'tmpk_st'): [('lrb1', 'tmpk_st'), ('lrb2',
                     #    'tmpk_st')],
                     ('tmpl01', 'tmpk_st'): [('tmpl0', 'tmpk_st'), ('tmpl1',
                         'tmpk_st')],
                     #('upsmp1', 'tmpk_st'): [('lrb1', 'tmpk_st')],
                     ('upsmp1', 'tmpk_st'): [('tmpl0', 'tmpk_st')],
                     #('upsmp2', 'tmpk_st'): [('upsmp1', 'tmpk_st'), ('lrb2',
                     #   'tmpk_st')],
                     ('upsmp2', 'tmpk_st'): [('upsmp1', 'tmpk_st'), ('tmpl1',
                        'tmpk_st')],
                     #('upsmp3', 'tmpk_st'): [('upsmp2', 'tmpk_st'), ('lrb3',
                     #   'tmpk_st')],
                     ('upsmp3', 'tmpk_st'): [('upsmp2', 'tmpk_st'), ('tmpl2',
                        'tmpk_st')],
                     ('upsmp1', 'up_mod'): 'constant',
                     ('upsmp2', 'up_mod'): 'constant',
                     ('upsmp3', 'up_mod'): 'constant'}
    search_params = {'loc_sd': lambda: random.randint(4, 16),
                     'lock_n': lambda: random.randint(1, 64),
                     'plk_n': lambda: random.randint(1, 32),
                     'plk_st': lambda: random.randint(1, 4),
                     'tmpk_sz': lambda: random.randint(1, 8),
                     'tmpk_st': lambda: random.randint(1, 4),
                     'gru_sz': lambda: random.randint(8, 128),
                     'out_ch': lambda: random.randint(1, 8),
                     'dpt_p': lambda: random.uniform(0, .5),
                     'n_std': lambda: random.uniform(0, .5),
                     'hid_sz': lambda: random.randint(8, 128),
                     'n_cmp': lambda: random.randint(1, 4),
                     'btch_sz': lambda: random.randint(8, 64),
                     'lr': lambda: random.uniform(1e-5, 1),
                     'lmbd': lambda: random.uniform(1e-5, 1),
                     'beta': lambda: random.uniform(1e-5, 1),
                     'rho': lambda: random.uniform(1e-5, .5),
                     'gamma': lambda: random.uniform(1e-5, 1),
                     'delta': lambda: random.randint(1, 8)}
    model_params = HLRNNParams(in_features, out_features, block_types,
        block_inputs, block_classifs, search_params, param_constrs)

    def hlrnn_search_trial(trial_config, model_params, search_params):
        inputdata._matlab_eng = None
        dataset = trial_config['dataset_class']({'splitprops': [0.6, 0.2, 0.2],
            'splitset': 0})
        inputdata._matlab_eng.rng(random.randint(0, 2 ** 20))
        trial_eval = HLRNNTrialEval(model_params, search_params, dataset,
            trial_config['model_state'])
        num_samples = {name: trial_config['num_sup_samples'] if name in
            model_params.block_classifs else round(trial_config[
            'num_unsup_samples'] / trial_eval.model.train_scale_factors[name])
            for name in trial_eval.block_names}
        trial_eval.train(num_samples)
        result = trial_eval.test({trial_eval.block_names[-1]: trial_config[
            'num_test_samples']})
        blocks_state, _, _ = trial_eval.get_state()
        return result[trial_eval.block_names[-1]]['metrics'], blocks_state

    def hlrnn_test_trial(trial_config, model_params, search_params):
        block_name = trial_config['block_name']
        inputdata._matlab_eng = None
        dataset = trial_config['dataset_class']({'splitprops': [.6, .2, .2],
            'splitset': 0})
        inputdata._matlab_eng.rng(random.randint(0, 2 ** 20))
        dataset.hard_restart({'splitprops': [.8, .0, .2], 'resplit': [True,
            True, False], 'splitset': 2})
        model = model_params.get_hlrnn(trial_config['model_state'])#.to(device)
        block_eval = HLRNNBlockEval(model, block_name,
            model_params.block_types[block_name], dataset,
            model_params.block_classifs[block_name],
            model_params.block_opt_params[block_name])
        block_eval.restart()
        _, _, metrics = block_eval.test(trial_config['num_test_samples'])
        metric_lists = {metric: [value] for metric, value in metrics.items()}
        num_samples = [round(step * trial_config['num_samples'] /
            trial_config['num_steps']) for step in range(trial_config[
            'num_steps'] + 1)]
        for step in range(trial_config['num_steps']):
            dataset.restart({'splitset': 0})
            block_eval.train(num_samples[step + 1] - num_samples[step])
            num_samples[step + 1] = block_eval.train_num_samples
            dataset.restart({'splitset': 2})
            _, _, metrics = block_eval.test(trial_config['num_test_samples'])
            for metric, metric_list in metric_lists.items():
                metric_list.append(metrics[metric])
        return num_samples, metric_lists

    def hlrnn_search_iter(search_config, trial_config, model_params):
        samples = [{param: func() for param, func in
            model_params.next_search_space.items()} for i in range(
            search_config['num_trials'])]
        fitness = []
        blocks_states = []
        for i, search_params in enumerate(samples):
            print(f'\nTrial {i}\n-------------------------------')
            print(f'search parameters: {search_params}')
            test_model_params, test_opt_params = \
                model_params.get_test_block_params(search_params)
            print(f'model initialization parameters: {test_model_params}')
            print(f'optimization parameters: {test_opt_params}')
            metrics, sample_blocks_state = hlrnn_search_trial(trial_config,
                model_params, search_params)
            for name, value in metrics.items():
                print(f'{name}: {value}')
            fitness.append(metrics[search_config['metric']])
            blocks_states.append(sample_blocks_state)
        if search_config['metric_mode'] == 'max':
            best_index = fitness.index(max(fitness))
        elif search_config['metric_mode'] == 'min':
            best_index = fitness.index(min(fitness))
        else:
            raise ValueError(f'{search_config["metric_mode"]} is not a valid '
                             f'metric optimization mode')
        return samples[best_index], blocks_states[best_index]

    def hlrnn_search(search_config, trial_config, model_params):
        if 'model_state' not in trial_config:
            trial_config['model_state'] = {}
        while model_params.next_block_names is not None:
            print(f'\n\nBlock {model_params.next_block_names[-1]}:\n'
                '-------------------------------')
            search_params, blocks_state = hlrnn_search_iter(search_config,
                trial_config, model_params)
            model_params.update_block_params(search_params)
            trial_config['model_state'].update(blocks_state)
            print('\nNew model parameters\n-------------------------------')
            print(f'model initialization parameters: '
                  f'{model_params.block_model_params}')
            print(f'optimization_parameters: {model_params.block_opt_params}')

    def hlrnn_test(test_config, trial_config, model_params):
        print(f'\n\nTest:\n'
              f'-------------------------------')
        results = []
        for i in range(test_config['num_trials']):
            print(f'\nTrial {i}\n-------------------------------')
            num_samples, metrics = hlrnn_test_trial(trial_config,
                model_params, search_params)
            for name, value in metrics.items():
                print(f'{name}: {value}')
            results.append({'num_samples': num_samples, 'metrics': metrics})
        return results

    hlrnn_search(search_config, trial_config, model_params)
    test_trial_config['model_state'] = trial_config['model_state']
    results = hlrnn_test(test_config, test_trial_config, model_params)
    avg_metrics = {name: [0] * len(value) for name, value in results[0][
        'metrics'].items()}
    for result in results:
        for metric, values in avg_metrics.items():
            for i, res_value in enumerate(result['metrics'][metric]):
                values[i] += res_value
    for name, values in avg_metrics.items():
        avg_metrics[name] = [value / len(results) for value in values]
    print(f'\nAverage metrics:\n-------------------------------')
    for name, value in avg_metrics.items():
        print(f'{name}: {value}')

    search_config['metric'] = 'error'
    search_config['metric_mode'] = 'min'
    block_types = OrderedDict([('tmpl3', 'TempMaxPool'),
                               ('lrbpred3', 'GRULRBPredictor'),
                               ('upsmp3p', 'MultiUpsample'),
                               ('lrbpred2', 'LRBPredictor'),
                               ('upsmp2p', 'MultiUpsample'),
                               ('lrbpred1', 'LRBPredictor'),
                               ('upsmp1p', 'MultiUpsample'),
                               ('lrbpred0', 'LRBPredictor')])
    block_inputs = {'tmpl3': 'lrb3',
                    'lrbpred3': 'tmpl3',
                    'upsmp3p': ['tmpl3', 'lrbpred3'],
                    'lrbpred2': ['tmpl2', 'upsmp3p'],
                    'upsmp2p': ['tmpl2', 'lrbpred2'],
                    'lrbpred1': ['tmpl1', 'upsmp2p'],
                    'upsmp1p': ['tmpl1', 'lrbpred1'],
                    'lrbpred0': ['tmpl0', 'upsmp1p']}
    block_classifs = {}
    param_constrs = {('tmpl3', 'tmpk_sz'): 3,
                     ('tmpl3', 'tmpk_st'): 2,
                     ('tmpl3', 'tmpk_ct'): False,
                     ('lrbpred3', 'out_ch'): 1,
                     ('lrbpred3', 'rho'): lambda params: 1 - (1 - params[
                         'lrb3']['rho']) ** params['lrb3']['plk_n'],
                     ('upsmp3p', 'tmpk_st'): [('tmpl3', 'tmpk_st')],
                     ('upsmp3p', 'up_mod'): 'constant',
                     ('lrbpred2', 'ctxt_h'): None,
                     ('lrbpred2', 'out_ch'): 1,
                     ('lrbpred2', 'rho'): lambda params: 1 - (1 - params[
                         'lrb2']['rho']) ** params['lrb2']['plk_n'],
                     ('upsmp2p', 'tmpk_st'): [('tmpl2', 'tmpk_st')],
                     ('upsmp2p', 'up_mod'): 'constant',
                     ('lrbpred1', 'ctxt_h'): None,
                     ('lrbpred1', 'out_ch'): 1,
                     ('lrbpred1', 'rho'): lambda params: 1 - (1 - params[
                         'lrb1']['rho']) ** params['lrb1']['plk_n'],
                     ('upsmp1p', 'tmpk_st'): [('tmpl1', 'tmpk_st')],
                     ('upsmp1p', 'up_mod'): 'constant',
                     ('lrbpred0', 'ctxt_h'): None,
                     ('lrbpred0', 'out_ch'): 1,
                     ('lrbpred0', 'rho'): .07}
    search_blocks = ['lrbpred3', 'lrbpred2', 'lrbpred1', 'lrbpred0']
    model_params.extend(block_types, block_inputs, block_classifs,
        param_constrs, search_blocks)

    hlrnn_search(search_config, trial_config, model_params)

    search_config['metric'] = 'acc'
    search_config['metric_mode'] = 'max'
    block_types = OrderedDict([('rec2p', 'UpsampleRNN'),
                               ('samp2', 'RectGaussMix')])
    block_inputs = {'rec2p': 'tmpl2',
                    'samp2': 'lrbpred2'}
    block_classifs = {'rec2p': dataset.classifs[0]}
    param_constrs = {('rec2p', 'tmpk_st'): [('tmpl0', 'tmpk_st'), ('tmpl1',
                         'tmpk_st'), ('tmpl2', 'tmpk_st')],
                     ('rec2p', 'up_pos'): 'last',
                     ('rec2p', 'up_mod'): 'constant'}
    model_params.extend(block_types, block_inputs, block_classifs,
        param_constrs)

    hlrnn_search(search_config, trial_config, model_params)

    def pred_test(config, model_params, preprocess_x=lambda x: x, keep_hiddens=
        True):
        print(f'\n\nPrediction test:\n'
              f'-------------------------------')
        block_name = config['block_name']
        inputdata._matlab_eng = None
        dataset = config['dataset_class']({'splitprops': [0.6, 0.2, 0.2],
            'splitset': 2})
        inputdata._matlab_eng.rng(random.randint(0, 2 ** 20))
        dataset.set_classif(model_params.block_classifs[block_name])
        model_params = copy.copy(model_params)
        model_params_state = model_params.state_dict()
        model_params_state['block_search_params'][block_name]['tmpk_st'] = 1
        model_params.load_state_dict(model_params_state)
        model = model_params.get_hlrnn(config['model_state'])#.to(device)
        model.set_cur_block(block_name)
        batch_size = round(math.ceil(model_params.block_opt_params[config[
            'block_name']]['batch_size'] * model.cur_min_scale_factor /
            model.train_scale_factor) / model.cur_min_scale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=
            batch_size)
        in_block_name = model.block_inputs[block_name]
        submodel = model.get_sub_hlrnn([block_name, config[
            'closed_loop_block_name']], [in_block_name])
        model.set_cur_block(in_block_name)
        submodel.set_cur_block(block_name)
        submodel.set_closed_loop_block(config['closed_loop_block_name'])
        horizon = config['horizon']
        closed_loop_horizon = math.ceil(horizon * model.scale_factor) + 1
        floor_horizon = math.floor(horizon * model.scale_factor)
        closed_loop_stride = round(1 / model.scale_factor)
        pred_shift = horizon % closed_loop_stride
        model.eval()
        submodel.eval()
        hs = [None] * len(model.hidden_names)
        subhs = []
        sample, acc, candidate, n_candidates = 0, 0, 0, 0
        with torch.no_grad():
            for *x, y in dataloader:
                #x, y = tuple(c.to(device) for c in x), y.to(device)
                subinput = model(*preprocess_x(x), *hs)
                if len(hs) != 0:
                    if keep_hiddens:
                        hs = [h.detach() if h is not None else None for h in
                            subinput[1:]]
                    subinput = subinput[0]
                pred, *subhs_ = submodel.closed_loop_forward(subinput, *subhs,
                    horizon=closed_loop_horizon, repetitions=config[
                    'repetitions'])
                if keep_hiddens:
                    subhs = [h.detach() if h is not None else None for h in
                        subhs_[:-3]] + subhs_[-3:]
                pred = pred[:, :, [floor_horizon] * (closed_loop_stride -
                    pred_shift) + [floor_horizon + 1] * pred_shift].permute(0,
                    2, 1, 3).reshape(-1, config['repetitions'], pred.shape[-1])
                sample += y.shape[0]
                ys_ = pred.argmax(2)
                acc += (ys_.mode(1)[0] == y).type(torch.float).sum().item()
                candidate += (ys_ == y.unsqueeze(1)).any(1).type(
                    torch.float).sum().item()
                n_candidates += (ys_.sort(1)[0].diff(dim=1) != 0).type(
                    torch.float).sum().item()
                if sample >= config['num_samples']:
                    break
        return sample, {'acc': acc / sample, 'candidate': candidate / sample,
            'n_candidates': n_candidates / sample + 1}

    pred_test_config = {'dataset_class': dataset_class, 'block_name': 'rec2p',
        'closed_loop_block_name': 'samp2', 'num_samples': 500, 'horizon':
        15, 'repetitions': 8, 'model_state': trial_config['model_state']}

    model_params.update_last_block_params()
    _, metrics = pred_test(pred_test_config, model_params)
    print(f'\nMetrics:\n-------------------------------')
    for name, value in metrics.items():
        print(f'{name}: {value}')

    classif = dataset.classifs[0]
    if dataset_name == 'SynthPlanInput':
        classif = 'plans'
    class LabelAtInputDataset(dataset_class):
        def __init__(self, params):
            super().__init__(params)
            self.set_classif(classif)
        def __next__(self):
            sample, label = super().__next__()
            return sample, torch.nn.functional.one_hot(label, len(
                self.all_lbls[self.classif])), torch.tensor([], dtype=
                torch.int64)

    trial_config['dataset_class'] = LabelAtInputDataset
    search_config['metric'] = 'error'
    search_config['metric_mode'] = 'min'
    in_features = OrderedDict([('label', out_features[classif])])
    block_types = OrderedDict([('downsample', 'Downsample'),
                               ('lrbpred3p', 'GRULRBPredictor'),
                               ('peak3', 'RectGaussMixPeak'),
                               ('upsmp3pp', 'MultiUpsample'),
                               ('lrbpred2p', 'LRBPredictor'),
                               ('peak2', 'RectGaussMixPeak'),
                               ('upsmp2pp', 'MultiUpsample'),
                               ('lrbpred1p', 'LRBPredictor'),
                               ('peak1', 'RectGaussMixPeak'),
                               ('upsmp1pp', 'MultiUpsample'),
                               ('lrbpred0p', 'LRBPredictor'),
                               ('peak0', 'RectGaussMixPeak'),
                               ('upsmp0pp', 'MultiUpsample')])
    block_inputs = {'downsample': 'label',
                    'lrbpred3p': ['tmpl3', 'downsample'],
                    'peak3': 'lrbpred3p',
                    'upsmp3pp': ['tmpl3', 'peak3'],
                    'lrbpred2p': ['tmpl2', 'upsmp3pp'],
                    'peak2': 'lrbpred2p',
                    'upsmp2pp': ['tmpl2', 'peak2'],
                    'lrbpred1p': ['tmpl1', 'upsmp2pp'],
                    'peak1': 'lrbpred1p',
                    'upsmp1pp': ['tmpl1', 'peak1'],
                    'lrbpred0p': ['tmpl0', 'upsmp1pp'],
                    'peak0': 'lrbpred0p',
                    'upsmp0pp': 'peak0'}
    block_classifs = {}
    param_constrs = {('downsample', 'tmpk_st'): [('tmpl0', 'tmpk_st'), (
                         'tmpl1', 'tmpk_st'), ('tmpl2', 'tmpk_st'), ('tmpl3',
                         'tmpk_st')],
                     ('lrbpred3p', 'out_ch'): 1,
                     ('lrbpred3p', 'rho'): lambda params: 1 - (1 - params[
                         'lrb3']['rho']) ** params['lrb3']['plk_n'],
                     ('upsmp3pp', 'tmpk_st'): [('tmpl3', 'tmpk_st')],
                     ('upsmp3pp', 'up_mod'): 'constant',
                     ('lrbpred2p', 'ctxt_h'): [('tmpl3', 'tmpk_st')],
                     ('lrbpred2p', 'out_ch'): 1,
                     ('lrbpred2p', 'rho'): lambda params: 1 - (1 - params[
                         'lrb2']['rho']) ** params['lrb2']['plk_n'],
                     ('upsmp2pp', 'tmpk_st'): [('tmpl2', 'tmpk_st')],
                     ('upsmp2pp', 'up_mod'): 'constant',
                     ('lrbpred1p', 'ctxt_h'): [('tmpl2', 'tmpk_st')],
                     ('lrbpred1p', 'out_ch'): 1,
                     ('lrbpred1p', 'rho'): lambda params: 1 - (1 - params[
                         'lrb1']['rho']) ** params['lrb1']['plk_n'],
                     ('upsmp1pp', 'tmpk_st'): [('tmpl1', 'tmpk_st')],
                     ('upsmp1pp', 'up_mod'): 'constant',
                     ('lrbpred0p', 'ctxt_h'): [('tmpl1', 'tmpk_st')],
                     ('lrbpred0p', 'out_ch'): 1,
                     ('lrbpred0p', 'rho'): .07,
                     ('upsmp0pp', 'tmpk_st'): [('tmpl0', 'tmpk_st')],
                     ('upsmp0pp', 'up_mod'): 'constant'}
    search_blocks = ['lrbpred3p', 'lrbpred2p', 'lrbpred1p', 'lrbpred0p']
    model_params.extend_input(in_features)
    model_params.extend(block_types, block_inputs, block_classifs,
        param_constrs, search_blocks)

    hlrnn_search(search_config, trial_config, model_params)

    def action_selection_test(config, model_params):
        print(f'\n\nAction selection test:\n'
              f'-------------------------------')
        inputdata._matlab_eng = None
        dataset = config['dataset_class']({'splitprops': [0.6, 0.2, 0.2],
            'splitset': 2})
        inputdata._matlab_eng.rng(random.randint(0, 2 ** 20))
        model = model_params.get_hlrnn(config['model_state'])#.to(device)
        model.set_cur_block(config['block_name'])
        model.set_closed_loop_block(config['block_name'])
        for block in model.blocks.values():
            if type(block) == lrnn.LRBPredictor:
                block.use_actual_ctxt = False
        achieved, avg_num_samples = actionselection.test(config, dataset,
            model)
        return {'achieved': achieved, 'avg_num_samples': avg_num_samples}

    if dataset_name == 'SynthPlanInput':
        action_selection_config = {'dataset_class': LabelAtInputDataset,
            'block_name': 'upsmp0pp', 'num_trials': 16, 'num_samples_factor':
            2, 'model_state': trial_config['model_state']}
        model_params.update_last_block_params()
        metrics = action_selection_test(action_selection_config, model_params)
        print(f'\nMetrics:\n-------------------------------')
        for name, value in metrics.items():
            print(f'{name}: {value}')


if __name__ == '__main__':
    import actionselection
    import inputdata
    import random
    import copy
    import sys
    
    main(*sys.argv[1:])
