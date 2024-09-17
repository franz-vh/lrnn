#!/usr/bin/env python3

'''A set of sequential PyTorch Datasets.

These datasets include wrappers for the MATLAB classes inheriting from
InputData, located in directory ./matlab/InputData. The actual data should be
located in directory ./datasets.
'''

import torch
from torch.utils.data import IterableDataset
import numpy as np
import matlab.engine
import os
from typing import Any, List, Tuple, Dict, Iterator


_project_dir = os.path.dirname(os.path.abspath(__file__))
_matlab_eng = None


class MatlabInputWrapper(IterableDataset):
    '''A PyTorch Dataset wrapper for MATLAB classes inheriting from InputData.

    Attributes:
        matlab_input: The wrapped MATLAB object.
        matlab_batch_size: An integer with the number of samples retrieved from
            the MATLAB object each time new samples are needed.
        matlab_name: A string with the name of the MATLAB object within the
            MATLAB Engine Workspace (eng).
        classifs: A list of strings with the names of all the classifications.
        all_lbls: A list of lists of strings with the class names per
            classification.
        range: A list of floats with the sample minimum and maximum limits.
        dims: A list of integers with the sample dimensions.
        size: An integer with the sample number of elements.
        is_compound: A boolean indicating if the MATLAB object class inherits
            from CompoundInput.
        sub_inputs: If is_compound, a list of strings with the names of the
            sample components.
        sub_dims: If is_compound, a list of lists of integers with the sample
            component dimensions.
        sub_sizes: If is_compound, a list of integers with the sample component
            number of elements.
        classif: An integer with the current classification identifier (which
            is used in method __next__).
        _nxt_smpls: A 2D tensor of floats with the retrieved samples from
            MATLAB that have not been yet returned through method __next__.
        _nxt_lbls: A 2D tensor of integers with the retrieved labels from
            MATLAB that have not been yet returned through method __next__.
        _nxt_stts: A list of dictionaries from string to 2D tensors of floats
            with the retrieved internal states from MATLAB that have not been
            yet returned.
    '''
    def __init__(self, input_name: str, params: Dict[str, Any] = {},
        matlab_batch_size: int = 16384) -> None:
        '''Inits MatlabInputWrapper.

        Arguments:
            input_name: The name of the MATLAB class to be wrapped.
            params: The parameters for the MATLAB class constructor (they
                depend on the specific class).
            matlab_batch_size: The number of samples retrieved from the MATLAB
                object each time new samples are needed.
        '''
        super(MatlabInputWrapper, self).__init__()
        global _matlab_eng
        if _matlab_eng is None:
            _matlab_eng = matlab.engine.start_matlab()
            _matlab_eng.addpath(os.path.join(_project_dir, 'matlab',
                'General'), nargout=0)
            _matlab_eng.addpath(os.path.join(_project_dir, 'matlab',
                'InputData'), nargout=0)
        self.matlab_input = getattr(_matlab_eng, input_name)(
            self.to_matlab_params(params))
        self.matlab_batch_size = matlab_batch_size
        matlab_vars = _matlab_eng.who()
        cnt = 0
        while 'input' + str(cnt) in matlab_vars:
            cnt += 1
        self.matlab_name = 'input' + str(cnt)
        _matlab_eng.workspace[self.matlab_name] = self.matlab_input
        self.classifs = [name for name in _matlab_eng.eval(self.matlab_name +
            '.Classifs')]
        self.all_lbls = [[name for name in lbls] for lbls in
            _matlab_eng.eval(self.matlab_name + '.AllLbls')]
        self.range = list(_matlab_eng.eval(self.matlab_name + '.Range')[0])
        self.dims = [int(dim) for dim in matlab.double(_matlab_eng.eval(
            self.matlab_name + '.Dims'))[0]]
        self.size = int(_matlab_eng.eval(self.matlab_name + '.Size'))
        self.is_compound = False
        if _matlab_eng.isa(self.matlab_input,'CompoundInput'):
            self.is_compound = True
            self.sub_inputs = [name for name in _matlab_eng.eval(
                self.matlab_name + '.SubInputs')]
            self.sub_dims = [[int(dim) for dim in matlab.double(dims)[0]] for
                dims in _matlab_eng.eval(self.matlab_name + '.SubDims')]
            self.sub_sizes = [int(size) for size in _matlab_eng.eval(
                self.matlab_name + '.SubSizes')[0]]
        self._nxt_smpls = torch.tensor([])
        self._nxt_lbls = torch.tensor([])
        self._nxt_stts = []
        self.classif = slice(0, 0)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        '''Returns the dataset as an iterator.

        Returns:
            iter: The iterator returning the samples and associated labels.
        '''
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Returns the next sample and label.

        Returns:
            sample: The next sample.
            label: The label associated to sample.
        '''
        if self._nxt_smpls.numel() == 0:
            nxt_smpls, nxt_lbls = _matlab_eng.getBatch(self.matlab_input,
                self.matlab_batch_size, nargout = 2)
            self._nxt_smpls = torch.from_numpy(np.array(nxt_smpls,
                dtype=np.float32)).T
            self._nxt_lbls = torch.from_numpy(np.array(nxt_lbls,
                dtype=np.int64)).T - 1
            if len(self._nxt_lbls) == 0:
                self._nxt_lbls = torch.empty(len(self._nxt_smpls), 0)
        sample = self._nxt_smpls[0]
        label = self._nxt_lbls[0][self.classif]
        self._nxt_smpls = self._nxt_smpls[1:]
        self._nxt_lbls = self._nxt_lbls[1:]
        return sample, label

    def set_classif(self, classif: 'str') -> None:
        '''Sets the current classification.

        Arguments:
            classif: The current classification name (which is used in method
                __next__).
        '''
        self.classif = self.classifs.index(classif)

    def restart(self, params: Dict[str, Any] = {}) -> None:
        '''Restarts the input.

        Arguments:
            params: The parameters for the MATLAB restart function (they
                depend on the specific class).
        '''
        _matlab_eng.restart(self.matlab_input, self.to_matlab_params(params),
            nargout = 0)
        self._nxt_smpls = torch.tensor([])
        self._nxt_lbls = torch.tensor([])

    def hard_restart(self, params: Dict[str, Any] = {}) -> None:
        '''Restarts the input as if the MATLAB object were new.

        Arguments:
            params: The parameters for the MATLAB hardRestart function (they
                depend on the specific class).
        '''
        _matlab_eng.hardRestart(self.matlab_input, self.to_matlab_params(
            params), nargout = 0)
        self._nxt_smpls = torch.tensor([])
        self._nxt_lbls = torch.tensor([])

    def get_state_batch(self, batch_size) -> Tuple[torch.Tensor, torch.Tensor,
        List[Any]]:
        '''Returns the next batch of samples, labels and internal states.

        Returns:
            batch: The next batch of samples.
            labels: The labels associated to samples in the batch.
            states: The internal states associated to samples in the batch.
        '''
        batch_list = []
        labels_list = []
        states = []
        cur_batch_size = 0
        nxt_smpls = self._nxt_smpls
        nxt_lbls = self._nxt_lbls
        nxt_stts = self._nxt_stts
        while batch_size > cur_batch_size + len(nxt_stts):
            batch_list.append(nxt_smpls)
            labels_list.append(nxt_lbls)
            states.extend(nxt_stts)
            cur_batch_size += len(nxt_stts)
            nxt_smpls, nxt_lbls, nxt_stts = _matlab_eng.getStateBatch(
                self.matlab_input, self.matlab_batch_size, nargout = 3)
            nxt_smpls = torch.from_numpy(np.array(nxt_smpls,
                dtype=np.float32)).T
            nxt_lbls = torch.from_numpy(np.array(nxt_lbls,
                dtype=np.int64)).T - 1
            if len(nxt_lbls) == 0:
                nxt_lbls = torch.empty(len(nxt_smpls), 0)
            if isinstance(nxt_stts[0], dict):
                nxt_stts = [{key: torch.from_numpy(np.array(value,
                    dtype=np.float32)).T for key, value in stt.items()} for stt
                    in nxt_stts]
        batch_list.append(nxt_smpls[:batch_size - cur_batch_size])
        labels_list.append(nxt_lbls[:batch_size - cur_batch_size])
        states.extend(nxt_stts[:batch_size - cur_batch_size])
        batch = torch.cat(batch_list)
        labels = torch.cat(labels_list)[:, self.classif]
        self._nxt_smpls = nxt_smpls[batch_size - cur_batch_size:]
        self._nxt_lbls = nxt_lbls[batch_size - cur_batch_size:]
        self._nxt_stts = nxt_stts[batch_size - cur_batch_size:]
        return batch, labels, states

    def get_sub_input(self, batch: torch.Tensor, sub_input) -> torch.Tensor:
        '''If is_compound, returns the batch of indicated sample components.

        Arguments:
            batch: The batch of samples.
            sub_input: The sample component identifier.
            
        Returns:
            sub_batch: The batch of sample components.
        '''
        return torch.from_numpy(np.array(_matlab_eng.getSubInput(
            self.matlab_input, matlab.double(np.array(batch.T,
            dtype=np.float64)), sub_input + 1), dtype=np.float32)).T

    def get_input_plot(self, sample: torch.Tensor) -> torch.Tensor:
        '''Returns the visualization of the given sample.

        Arguments:
            sample: The input sample to be visualized.

        Returns:
            sample_vis: The visualization of the given sample.
        '''
        return torch.from_numpy(np.array(_matlab_eng.getInputPlot(
            self.matlab_input, matlab.double(np.array(sample.unsqueeze(1),
            dtype=np.float64))), dtype=np.float32))

    def get_state_plot(self, sample: torch.Tensor, state: Any) -> torch.Tensor:
        '''Returns the visualization of the state of the given sample.

        Arguments:
            sample: The input sample to be visualized.
            state: The internal state associated to sample.

        Returns:
            state_vis: The visualization of the internal state of the given
                sample.
        '''
        return self.get_input_plot(sample)

    @staticmethod
    def to_matlab_params(params: Dict[str, Any]) -> Dict[str, Any]:
        '''Converts the constructor parameters to their MATLAB form.

        Arguments:
            params: The parameters for the MATLAB class function (they depend
                on the specific class).

        Returns:
            matlab_params: The parameters converted to their MATLAB form.
        '''
        matlab_params = params.copy()
        if 'splitprops' in params:
            matlab_params['splitprops'] = matlab.double(params['splitprops'])
        if 'splitset' in params:
            matlab_params['splitset'] += 1
        return matlab_params


class SynthActionsWrapper(MatlabInputWrapper):
    '''A PyTorch Dataset wrapper for classes inheriting from SynthActions.

    Attributes:
        _data: A dictionary with the internal input data.
        _vis_data: A dictionary with the internal input visualization data.
    '''
    def __init__(self, input_name: str, params: Dict[str, Any] = {},
        matlab_batch_size: int = 16384) -> None:
        '''Inits SynthActionsWrapper.

        Arguments:
            input_name: The name of the MATLAB class to be wrapped.
            params: The parameters for the MATLAB class constructor (they
                depend on the specific class).
            matlab_batch_size: The number of samples retrieved from the MATLAB
                object each time new samples are needed.
        '''
        super(SynthActionsWrapper, self).__init__(input_name, params,
            matlab_batch_size)
        self._data = _matlab_eng.eval(self.matlab_name + '.Data')
        self._vis_data = _matlab_eng.eval(self.matlab_name + '.VisData')
        self._data['in_p']['objid_sz'] = int(self._data['in_p']['objid_sz'])
        self._data['n_objects'] = int(self._data['n_objects'])
        self._data['dims'] = {key: int(value) for key, value in
            self._data['dims'].items()}
        self._vis_data['hand_w'] = int(self._vis_data['hand_w'])
        self._vis_data['margin'] = int(self._vis_data['margin'])
        self._vis_data['colors'] = torch.from_numpy(np.array(self._vis_data[
            'colors'], dtype=np.float32))

    def get_scenario_plot(self) -> torch.Tensor:
        '''Returns the visualization of the current scenario.

        Returns:
            scenario_vis: The visualization of the current scenario.
        '''
        return torch.from_numpy(np.array(_matlab_eng.getScenarioPlot(
            self.matlab_input), dtype=np.float32))

    def get_state_plot(self, sample: torch.Tensor, state: Dict[str,
        torch.Tensor]) -> torch.Tensor:
        '''Returns the visualization of the state of the given sample.

        Arguments:
            sample: The input sample to be visualized.
            state: The internal state associated to sample.

        Returns:
            state_vis: The visualization of the internal state of the given
                sample.
        '''
        objtypes = state['objtypes'].squeeze().to(torch.int64) - 1
        x = (state['x'].squeeze() + self._vis_data['margin'] - self._data[
            'dims']['obj_w'] / 2 - 1).round().to(torch.int64)
        y = (state['y'].squeeze() + self._vis_data['margin'] - self._data[
            'dims']['obj_h'] / 2 - 1).round().to(torch.int64)
        x_hand = (state['x_hand'] + self._vis_data['margin'] - self._vis_data[
            'hand_w'] / 2 - 1).round().to(torch.int64)
        y_hand = (state['y_hand'] + self._vis_data['margin'] - state[
            'hand_h'] / 2 - 1).round().to(torch.int64)
        state_vis = torch.zeros([self._data['dims']['scen_h'] + 2 *
            self._vis_data['margin'], self._data['dims']['scen_w'] + 2 *
            self._vis_data['margin'], 3])
        for x_i, y_i, objtype in zip(x, y, objtypes):
            state_vis[y_i:y_i + self._data['dims']['obj_h'], x_i:x_i +
                self._data['dims']['obj_w'], :] = self._vis_data['colors'][
                objtype].reshape([1, 1, 3]) * torch.ones(self._data['dims'][
                'obj_h'], self._data['dims']['obj_w'], 3)
        state_vis[y_hand:y_hand + state['hand_h'].round().to(torch.int64),
            x_hand:x_hand + self._vis_data['hand_w'], :] = 1
        return state_vis


class SynthPlanInput(SynthActionsWrapper):
    '''A PyTorch Dataset wrapper for MATLAB SynthPlanInput class.'''
    def __init__(self, params: Dict[str, Any] = {}, matlab_batch_size: int =
        16384) -> None:
        '''Inits SynthPlanInput.

        Arguments:
            params: The parameters for the MATLAB class constructor (they are
                ignored for this class).
            matlab_batch_size: The number of samples retrieved from the MATLAB
                object each time new samples are needed.
        '''
        super(SynthPlanInput, self).__init__('SynthPlanInput', params,
            matlab_batch_size)


class SynthActInput(SynthActionsWrapper):
    '''A PyTorch Dataset wrapper for MATLAB SynthActInput class.'''
    def __init__(self, params: Dict[str, Any] = {}, matlab_batch_size: int =
        16384) -> None:
        '''Inits SynthActInput.

        Arguments:
            params: The parameters for the MATLAB class constructor (they are
                ignored for this class).
            matlab_batch_size: The number of samples retrieved from the MATLAB
                object each time new samples are needed.
        '''
        super(SynthActInput, self).__init__('SynthActInput', params,
            matlab_batch_size)


class WardInput(MatlabInputWrapper):
    '''A PyTorch Dataset wrapper for MATLAB WardInput class.'''
    def __init__(self, params: Dict[str, Any] = {}, matlab_batch_size: int =
        16384) -> None:
        '''Inits WardInput.

        Arguments:
            params: The parameters for the MATLAB class constructor. These
                parameters are mandatory and include:
                splitprops: MATLAB double with the desired proportion of
                    samples per split set (requires hard restart to be
                    changed).
                splitset: Integer with the identifier of the current split set
                    (requires restart to be changed).
            matlab_batch_size: The number of samples retrieved from the MATLAB
                object each time new samples are needed.
        '''
        super(WardInput, self).__init__('WardInput', params, matlab_batch_size)


class FsddInput(MatlabInputWrapper):
    '''A PyTorch Dataset wrapper for MATLAB FsddInput class.'''
    def __init__(self, params: Dict[str, Any] = {}, matlab_batch_size: int =
        16384) -> None:
        '''Inits FsddInput.

        Arguments:
            params: The parameters for the MATLAB class constructor. These
                parameters are mandatory and include:
                splitprops: MATLAB double with the desired proportion of
                    samples per split set (requires hard restart to be
                    changed).
                splitset: Integer with the identifier of the current split set
                    (requires restart to be changed).
            matlab_batch_size: The number of samples retrieved from the MATLAB
                object each time new samples are needed.
        '''
        super(FsddInput, self).__init__('FsddInput', params, matlab_batch_size)


class ShiftedImgInput(MatlabInputWrapper):
    '''A PyTorch Dataset wrapper for MATLAB ShiftedImgInput class.'''
    def __init__(self, params: Dict[str, Any] = {}, matlab_batch_size: int =
        16384) -> None:
        '''Inits ShiftedImgInput.

        Arguments:
            params: The parameters for the MATLAB class constructor. These
                parameters include:
                splitprops: (mandatory) MATLAB double with the desired
                    proportion of samples per split set (requires hard restart
                    to be changed).
                splitset: (mandatory) Integer with the identifier of the
                    current split set (requires restart to be changed).
                dims: (optional) MATLAB double with the sample dimensions.
                    Defaults to [8, 8].
            matlab_batch_size: The number of samples retrieved from the MATLAB
                object each time new samples are needed.
        '''
        super(ShiftedImgInput, self).__init__('ShiftedImgInput', params,
            matlab_batch_size)


def main(dataset_name: str = 'SynthPlanInput') -> None:
    '''Visualizes sequential samples of the given dataset.

    Arguments:
        dataset_name: The name of the PyTorch Dataset.
    '''
    dataset = getattr(sys.modules[__name__], dataset_name)({'splitprops':
        [0.6, 0.2, 0.2], 'splitset': 0})
    if len(dataset.classifs) > 0:
        dataset.set_classif(dataset.classifs[0])
    sample, label = next(dataset)
    sample_vis = dataset.get_input_plot(sample)
    if sample_vis.dim() == 2:
        sampleplot = plt.imshow(sample_vis, cmap='gray')
    else:
        sampleplot = plt.imshow(sample_vis)
    while True:
        if isinstance(dataset.classif, int):
            plt.title(dataset.all_lbls[dataset.classif][label])
        plt.show(block=False)
        plt.pause(0.033)
        sample, label = next(dataset)
        sample_vis = dataset.get_input_plot(sample)
        sampleplot.set_data(sample_vis)


if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    
    main(*sys.argv[1:])
