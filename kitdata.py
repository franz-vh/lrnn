#!/usr/bin/env python3

'''The KIT Whole-Body Human Motion Database.

This module includes three PyTorch datasets: one with the original data (with
few corrections, normalization and clamping), another one with processed data,
which contains more consistent coordinate frames accross objects, coordinates
expressed with respect to the torso, and rotations expressed using rotation
matrices, and another one that makes use of those more consistent coordinates
and ego perspective and represents positions and orientations in a sparse way
using grids. The actual data should be located in directory ./datasets.
These three datasets include a function to recover the original data from the
processed data (which can be useful, e.g., for visualization of batches). This
recovery process is however not perfect and has some limitations. In
particular, in the second dataset (class KitInput):
 - The data only contains information about the yaw of the torso and pitch and
yaw of the head.
 - The clamping over the differential torso position and yaw is not corrected,
making the whole environment move when these movements are fast enough.
 - The clamping over the object positions is corrected assuming that when the
objects are far enough they don't move. However, if a coordinate is clamped for
the whole batch (common for short enough batches), it is not possible to
recover the correct position.
 - There are few objects that have the same identifier (apple_juice_lid and
milk_small_lid, mixing_bowl_big and mixing_bowl_green). These objects are
always recovered as the object that appears first in the list.
In the third dataset (class KitSparseInput), besides those of the second
dataset not related to clamping:
- Due to symmetries, the orientations of some objects are represented through
the sum of gaussians, making data recovery not straightforward. This has been
approached by estimating the vectors as weighted averages of the vectors
pointing to the centers of the grid cells. This gives quite good results for
objects without symmetries, and also for objects with circular or 90°
symmetries with the affected vector being obtained as the orthogonal of the
other vector. However, for vectors with 180° symmetries, for which the
representation is the sum of two opposed gaussians, the approach has consisted
of finding the maximum and estimating the vector as the weighted average of the
grid vectors around that maximum. This leads to sudden jumps when the maximum
shifts from one grid to the contiguous one.
- When using 16-bit floats to store the trials (option mem_opt set to True),
the lower values of the position grids fall to 0, which brings some issues when
then recovering the original values. First, because, when inverting the
exponential, this leads to -infinity values (avoided easily by just setting
those 0s to a very small float). Second, because these wrongly set values
contribute equally to the final value as the correct ones if we just do an
average of the multiple pair-wise estimations, which has been approached by
weighting the averaged values by the product of the involved grid values. And
third, because these wrong values can also lead to values that are out of the
range (-1, 1), which leads to nans when applying the atanh (approached by
making those values -.9 or +.9).
'''

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
import random
import math
import copy
import glob
import csv
import re
from collections import OrderedDict
import os
import warnings
import typing
from typing import Any, List, Tuple, Dict, Iterator


_datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    'datasets')


class KitRawInput(IterableDataset):
    '''A PyTorch Dataset for the raw KIT Whole-Body Human Motion Database.

    Attributes:
        splitprops: A list of floats with the desired proportion of samples per
            split set (requires hard restart to be changed).
        splitset: An integer with the identifier of the current split set
            (requires restart to be changed).
        cat_x: A boolean indicating whether the fixed and variable part of the
            samples are given concatenated in the same tensor.
        makeup_objs: A boolean indicating whether the trials are augmented with
            random objects.
        random_mirror: A boolean indicating whether the trials are randomly
            augmented through mirroring of the scene.
        random_starts: A boolean indicating whether the trials are returned
            starting from some random sample at the beginning of the trial.
        makeup_actions: A boolean indicating whether the labeled long sequences
            should deduce the action labels from the subaction labels when
            possible.
        mem_opt: A boolean indicating whether the dataset is memory-optimized
            by storing the data as float16 instead of float32.
        keep_orig_trials: A boolean indicating whether the original trials
            should be kept.
        classifs: A list of strings with the names of all the classifications.
        all_lbls: A list of lists of strings with the class names per
            classification.
        fix_size: An integer with the number of elements of the fixed part of
            the sample.
        var_size: An integer with the number of elements of each component of
            the variable part of the sample.
        classif: An integer with the current classification identifier (which
            is used in method __next__).
        must_classifs: A list of integers with the current mandatory classification
            identifiers (method __iter__ will only pool trials with the indicated
            classifications).
        must_trial_lbls: A dictionary of strings to lists of strings with the
            trial classification names and current mandatory labels (method
            __iter__ will only pool trials with the indicated labels).
        batch_size: An integer with the batch size when the batch iterator is
            used, or None.
        _trials: A list of dictionaries with the samples and the labels
            corresponding to each trial.
        _orig_trials: A list of dictionaries with the original samples and
            labels (before normalization) corresponding to each trial.
        _splitsets: A list of lists of dictionaries with the split sets
            containing their corresponding trials.
        _splitset: A list of dictionaries with the samples and the labels
            corresponding to each trial of the current split set.
        _fix_obj_names: A list of strings with the objects that are always
            there and only once at each sample, and whose data forms the fixed
            part of the samples.
        _max_n_objs: An integer with the maximum number of objects in a trial.
        _madeup_objs: A dictionary of strings to dictionaries of strings to
            lists of tensors with the objects and possible positions and
            orientations that they can take when they are used to augment
            trials.
        _mirror_params: A dictionary of strings to anything with the keys and
            corresponding parameters used for the mirroring augmentation.
        _coord_corrections: A dictionary with the position and orientation
            corrections applied to the different objects.
        _norm_types: A dictionary of strings to lists of strings with the
            normalization types and corresponding lists of data types.
        _norm_fns: A dictionary of strings to functions with the normalization
            types and corresponding normalization functions.
        _unnorm_fns: A dictionary of strings to functions with the
            normalization types and corresponding unnormalization functions.
        _norm_params: A dictionary with the keys and normalization parameters
            of the different types of data.
        _object_ids: A dictionary with the object names and identifiers.
        _x_field_sizes: A dictionary with the components and sizes of the fixed
            and variable parts of the sample.
        #_trial: A dictionary with the samples and the labels corresponding to
        #    the current trial.
        #_i: An integer with the index of the last retrieved sample in the
        #    current trial.
        #_new_trial: A boolean indicating whether a new trial should be selected
        #    on the next call to method __next__.
    '''
    def __init__(self, params: Dict[str, Any] = {}) -> None:
        '''Inits KitRawInput.

        Arguments:
            params: The constructor parameters. These parameters include:
                splitprops (mandatory): A float with the desired proportion of
                    samples per split set (requires hard restart to be
                    changed).
                splitset (mandatory): An integer with the identifier of the
                    current split set (requires restart to be changed).
                cat_x (optional): A boolean indicating whether the fixed and
                    variable part of the samples are given concatenated in the
                    same tensor (defaults to False).
                makeup_objs (optional): A boolean indicating whether the trials
                    are augmented with random objects (defaults to True).
                random_mirror (optional): A boolean indicating whether the
                    trials are randomly augmented through mirroring of the
                    scene (defaults to True).
                random_starts (optional): A boolean indicating whether the
                    trials are returned starting from some random sample at the
                    beginning of the trial (defaults to True).
                makeup_actions (optional): A boolean indicating whether the
                    labeled long sequences should deduce the action labels from
                    the subaction labels when possible (defaults to True).
                mem_opt (optional): A boolean indicating whether the dataset is
                    memory-optimized (or compute-optimized), by, e.g., keeping
                    in memory the non-sparse representations (defaults to
                    True).
                keep_orig_trials (optional): A boolean indicating whether the
                    original trials should be kept (defaults to False).
        '''
        super(KitRawInput, self).__init__()
        self.splitprops = params['splitprops']
        self.cat_x = False
        if 'cat_x' in params:
            self.cat_x = params['cat_x']
        self.makeup_objs = True
        if 'makeup_objs' in params:
            self.makeup_objs = params['makeup_objs']
        if self.makeup_objs:
            self._madeup_objs = {'xy': {}, 'zq': {}}
        self.random_mirror = True
        if 'random_mirror' in params:
            self.random_mirror = params['random_mirror']
        if self.random_mirror:
            self._mirror_params = {}
        self.random_starts = True
        if 'random_starts' in params:
            self.random_starts = params['random_starts']
        self.makeup_actions = True
        if 'makeup_actions' in params:
            self.makeup_actions = params['makeup_actions']
        self.mem_opt = True
        if 'mem_opt' in params:
            self.mem_opt = params['mem_opt']
        self.keep_orig_trials = False
        if 'keep_orig_trials' in params:
            self.keep_orig_trials = params['keep_orig_trials']
        self._init_data()
        self.restart(params)
        self.set_sample_iter()
        self.classif = slice(0, 0)
        self.must_classifs = []
        self.must_trial_lbls = {}

    def _init_data(self) -> None:
        '''Inits the dataset data.'''
        self._fix_obj_names = ['left_hand', 'right_hand', 'head', 'torso']
        self._read_dataset()
        if self.random_mirror:
            self._init_mirror_lbls()
        #self._remove_nans()
        if self.keep_orig_trials:
            self._orig_trials = copy.deepcopy(self._trials)
        self._process_data()
        if self.makeup_objs:
            self._init_madeup_objs()
        self._norm_data()
        self._arrange_data()
        if self.mem_opt:
            for trial in self._trials:
                trial['x'].clamp_(0, 1)
                trial['x'][trial['x'] < torch.finfo(torch.float16).tiny] *= \
                    torch.finfo(torch.float16).min / torch.finfo(
                    torch.float16).tiny
                trial['x'][trial['x'] >= torch.finfo(torch.float16).tiny] *= \
                    torch.finfo(torch.float16).max
                trial['x'] = trial['x'].to(torch.float16)
        self._split_in_sets()

    def _read_dataset(self) -> None:
        '''Loads the dataset file data.'''
        classif_fieldnames = OrderedDict([
            ('leftprimitives', 'left_subaction'),
            ('rightprimitives', 'right_subaction'),
            ('leftsubactions', 'left_action'),
            ('rightsubactions', 'right_action'),
            ('leftmainobjects', 'left_main_object'),
            ('leftmainobjects2', 'left_main_object_2'),
            ('leftsourceobjects', 'left_source_object'),
            ('lefttargetobjects', 'left_target_object'),
            ('rightmainobjects', 'right_main_object'),
            ('rightmainobjects2', 'right_main_object_2'),
            ('rightsourceobjects', 'right_source_object'),
            ('righttargetobjects', 'right_target_object')])
        other_fieldnames = ['left_failure', 'right_failure']
        classif_types = {'leftprimitives': 'primitives',
                         'rightprimitives': 'primitives',
                         'leftsubactions': 'subactions',
                         'rightsubactions': 'subactions',
                         'actions': 'actions',
                         'leftmainobjects': 'objects',
                         'leftmainobjects2': 'objects',
                         'leftsourceobjects': 'objects',
                         'lefttargetobjects': 'objects',
                         'rightmainobjects': 'objects',
                         'rightmainobjects2': 'objects',
                         'rightsourceobjects': 'objects',
                         'righttargetobjects': 'objects'}
        type_lbls = {type: OrderedDict() for type in list(
            classif_types.values()) + ['actions', 'plans', 'subjects',
                'subsets', 'labelprops']}
        subaction_actions = {'Open': 'Open',
                             'Pour': 'Pour',
                             'Scoop': 'Scoop',
                             'Stir': 'Stir',
                             'Close': 'Close',
                             'RollOut': 'Roll',
                             'Peel': 'Peel',
                             'Cut': 'Cut',
                             'Transfer': 'Transfer',
                             'Mix': 'Mix',
                             'Wipe': 'Wipe',
                             'Sweep': 'Sweep'}
        filenames = glob.glob(os.path.join(_datasets_dir, 'kit', '*.csv'))
        trials = []
        self._max_n_objs = 0
        for filename in filenames:
            subject, rest = os.path.basename(filename).split(sep='_-_',
                maxsplit=1)
            task, rest = re.split('[0-9X]', rest, maxsplit=1)
            task = task.strip('_')
            if 'Stir' in task:
                task = 'Stir'
            with open(filename, newline='') as csvfile:
                csvreader = csv.reader(csvfile)
                fieldnames = next(iter(csvreader))
                lbl_is = {classif: fieldnames.index(fieldname) for classif,
                    fieldname in classif_fieldnames.items() if fieldname in
                    fieldnames}
                first_x_i = max([0, *lbl_is.values(), *[fieldnames.index(
                    fieldname) for fieldname in other_fieldnames if fieldname
                    in fieldnames]]) + 1
                x_is = {key: i for i, key in enumerate(fieldnames[first_x_i:])}
                rows = list(csvreader)
                x_raw = torch.tensor([[float(elem) for elem in row[first_x_i:]]
                    for row in rows])
                x = {'fix': [], 'var': []}
                obj_names = {fieldname.split(sep='[', maxsplit=1)[0] for
                    fieldname in x_is}
                for obj_name in obj_names:
                    coords = {}
                    coords['name'] = obj_name
                    coords['position'] = torch.stack([x_raw[:, x_is[
                        f'{obj_name}[{coordname}]']] for coordname in ['x',
                        'y', 'z']]).T
                    coords['orientation'] = torch.stack([x_raw[:, x_is[
                        f'{obj_name}[{coordname}]']] for coordname in ['qw',
                        'qx', 'qy', 'qz']]).T
                    if task == 'Cut' and obj_name == 'cucumber_attachment':
                        if rest.startswith('_e_'):
                            coords['name'] = 'eggplant_attachment'
                        if rest.startswith('_c_'):
                            coords['orientation'] = _rot_quaternion_axis(
                                coords['orientation'], 'local_x', torch.pi / 2)
                    if obj_name in self._fix_obj_names:
                        x['fix'].append(coords)
                    else:
                        x['var'].append(coords)
                        type_lbls['objects'][coords['name']] = None
                self._max_n_objs = max(self._max_n_objs, len(x['var']))
                lbl_strs = {classif: [row[index] for row in rows] for classif,
                    index in lbl_is.items()}
                if 'leftsubactions' in lbl_strs and any(lbl_str == 'Unknown'
                    for lbl_str in lbl_strs['leftsubactions']):
                    lbl_strs.pop('leftprimitives', None)
                    del lbl_strs['leftmainobjects']
                    del lbl_strs['leftmainobjects2']
                    del lbl_strs['leftsourceobjects']
                    del lbl_strs['lefttargetobjects']
                if 'rightsubactions' in lbl_strs and any(lbl_str == 'Unknown'
                    for lbl_str in lbl_strs['rightsubactions']):
                    lbl_strs.pop('rightprimitives', None)
                    del lbl_strs['rightmainobjects']
                    del lbl_strs['rightmainobjects2']
                    del lbl_strs['rightsourceobjects']
                    del lbl_strs['righttargetobjects']
                for classif in list(lbl_strs):
                    if all(lbl_str == 'Unknown' for lbl_str in lbl_strs[
                        classif]):
                        del lbl_strs[classif]
                if self.makeup_actions and task[0].islower() and \
                    'leftsubactions' in lbl_strs and 'rightsubactions' in \
                    lbl_strs:
                    actions = ['Unknown'] * len(rows)
                    for i, (left, right) in enumerate(zip(lbl_strs[
                        'leftsubactions'], lbl_strs['rightsubactions'])):
                        if left in subaction_actions:
                            actions[i] = subaction_actions[left]
                        elif right in subaction_actions:
                            actions[i] = subaction_actions[right]
                    lbl_strs['actions'] = actions
                for classif in lbl_strs:
                    if classif_types[classif] == 'objects':
                        lbl_strs[classif] = ['kitchen_sideboard' if lbl_str in
                            ('kinect_azure_right', 'kinect_azure_left',
                            'kitchen_sideboard_long') else '' if lbl_str ==
                            '1723' else lbl_str for lbl_str in lbl_strs[
                            classif]]
                        if task == 'Cut' and rest.startswith('_e_'):
                            lbl_strs[classif] = ['eggplant_attachment' if
                                lbl_str == 'cucumber_attachment' else lbl_str
                                for lbl_str in lbl_strs[classif]]
                    for lbl_str in lbl_strs[classif]:
                        type_lbls[classif_types[classif]][lbl_str] = None
                trial_lbl_strs = {'subjects': subject}
                if task[0].isupper():
                    trial_lbl_strs['actions'] = task
                    trial_lbl_strs['subsets'] = 'short'
                else:
                    trial_lbl_strs['plans'] = task
                    trial_lbl_strs['subsets'] = 'long'
                trial_lbl_strs['labelprops'] = 'full'
                for classif in lbl_strs:
                    if any(lbl_str == 'Unknown' for lbl_str in lbl_strs[
                        classif]):
                        trial_lbl_strs['labelprops'] = 'partial'
                for classif, lbl_str in trial_lbl_strs.items():
                    type_lbls[classif][lbl_str] = None
                for strs in type_lbls.values():
                    strs.pop('Unknown', None)
                    for lbl_i, lbl_str in enumerate(strs):
                        strs[lbl_str] = lbl_i
                    strs['Unknown'] = -1
                lbls = {classif: torch.tensor([type_lbls[classif_types[
                    classif]][lbl_str] for lbl_str in strs]) for classif, strs
                    in lbl_strs.items()}
                if 'leftmainobjects' in lbl_strs and 'rightmainobjects' in \
                    lbl_strs:
                    obj_is = {'Unknown': -3, '': -2, 'kitchen_sideboard': -1,
                        **{coords['name']: i for i, coords in enumerate(x[
                        'var'])}}
                    lbls['leftmainobjectindices'] = torch.tensor([obj_is[
                        lbl_str] for lbl_str in lbl_strs['leftmainobjects']])
                    lbls['rightmainobjectindices'] = torch.tensor([obj_is[
                        lbl_str] for lbl_str in lbl_strs['rightmainobjects']])
                trial_lbls = {classif: type_lbls[classif][lbl_str] for classif,
                    lbl_str in trial_lbl_strs.items()}
                for classif, lbl in trial_lbls.items():
                    lbls[classif] = torch.tensor([lbl] * len(rows))
                trials.append({'x': x, 'lbls': lbls, 'trial_lbls': trial_lbls,
                    'extra_info': {}})
        self.classifs = list(classif_fieldnames)
        self.classifs.insert(4, 'actions')
        self.classifs.insert(5, 'plans')
        self.classifs += ['leftmainobjectindices', 'rightmainobjectindices',
            'subjects', 'subsets', 'labelprops']
        for strs in type_lbls.values():
            strs.pop('Unknown', None)
        type_lbls['indices'] = range(self._max_n_objs)
        self.all_lbls = [list(type_lbls[classif_types[classif]]) for classif in
            classif_fieldnames]
        self.all_lbls.insert(4, list(type_lbls['actions']))
        self.all_lbls.insert(5, list(type_lbls['plans']))
        self.all_lbls += [list(type_lbls[type]) for type in ['indices',
            'indices', 'subjects', 'subsets', 'labelprops']]
        self._trials = trials

    def _init_mirror_lbls(self) -> None:
        '''Initializes the label mapping for mirrored trials.'''
        mapping = {'leftprimitives': 'rightprimitives',
                   'leftsubactions': 'rightsubactions',
                   'leftmainobjects': 'rightmainobjects',
                   'leftmainobjects2': 'rightmainobjects2',
                   'leftsourceobjects': 'rightsourceobjects',
                   'lefttargetobjects': 'righttargetobjects'}
        mapping.update({value: key for key, value in mapping.items()})
        classif_is = {classif: i for i, classif in enumerate(self.classifs)}
        self._mirror_params['lbls'] = [classif_is[mapping[classif]] if classif
            in mapping else classif_is[classif] for classif in self.classifs]

    def _process_data(self) -> None:
        '''Removes the outliers using a median filter.'''
        self._coord_corrections = {}
        for trial in self._trials:
            x = trial['x']
            self._filter_outliers(x, 5)
            if self.makeup_objs:
                self._update_madeup_objs(x, trial['trial_lbls'])
                trial['extra_info']['trial_size'] = x['fix'][0][
                    'position'].shape[0]
                trial['extra_info']['static_pos'] = self._get_static_positions(
                    x)
            trial['x'] = x

    @staticmethod
    def _filter_outliers(x: Dict[str, List[Dict[str, Any]]], span: int) -> \
        None:
        '''Removes the outliers using a median filter.

        Arguments:
            x: The names and coordinates (position and orientation) of the
                objects of a trial.
        '''
        for part in x.values():
            for coords in part:
                for key, coord in coords.items():
                    if key != 'name':
                        coords[key] = F.pad(coord.T, (span // 2, span // 2),
                            mode='replicate').contiguous().as_strided((
                            coord.shape[1], coord.shape[0], span), (
                            coord.shape[0] + span // 2 * 2, 1, 1)).median(2)[
                            0].T

    def _update_madeup_objs(self, x: Dict[str, List[Dict[str, Any]]],
        trial_lbls: Dict[str, int]) -> None:
        '''Updates the possible coordinates that made up objects can take.

        Arguments:
            x: The names and coordinates (position and orientation) of the
                objects of a trial.
        '''
        for coords in x['var']:
            subset = trial_lbls['subsets']
            if subset not in self._madeup_objs['xy']:
                self._madeup_objs['xy'][subset] = []
            self._madeup_objs['xy'][subset] += [coords['position'][0, :2],
                coords['position'][-1, :2]]
            obj_name = coords['name']
            if obj_name not in self._madeup_objs['zq']:
                self._madeup_objs['zq'][obj_name] = []
            self._madeup_objs['zq'][obj_name] += [torch.cat([coords[
                'position'][0, 2:], coords['orientation'][0]], 0), torch.cat([
                coords['position'][0, 2:], coords['orientation'][0]], 0)]

    @staticmethod
    def _get_static_positions(x: Dict[str, List[Dict[str, Any]]]) -> \
        torch.Tensor:
        '''Returns the object positions at the beginning and end of the trial.

        Arguments:
            x: The names and coordinates (position and orientation) of the
                objects of a trial.

        Returns:
            positions: The x and y coordinates of the positions at the
                beginning and end of the trial.
        '''
        pos0 = [coords['position'][0, :2] for coords in x['var']]
        posf = [coords['position'][-1, :2] for coords in x['var']]
        return torch.stack(pos0 + posf).unsqueeze(0)

    def _init_madeup_objs(self) -> None:
        '''Inits the collected coordinates that made up objects can take.'''
        for key, coords in self._madeup_objs.items():
            for name, coord in coords.items():
                coords[name] = torch.stack(coord).unsqueeze(1)
                if key == 'zq':
                    keep_factor = 1 / 5 if name == 'milk_small_lid' else 1 / 2
                    coords[name] = coords[name][coords[name][:, 0, 0].sort()[
                        1][:math.ceil(coords[name].shape[0] * keep_factor)]]

    def _norm_data(self) -> None:
        '''Truncates and rescales the data to the range (0, 1).'''
        self._define_norm_criteria()
        self._norm_params = {}
        for type, keys in self._norm_types.items():
            self._norm_fns[type](keys)

    def _define_norm_criteria(self) -> None:
        '''Defines criteria to normalize the different types of data.'''
        self._norm_types = {'positions': ['position'], 'orientations':
            ['orientation']}
        self._norm_fns = {'positions': self._norm_positions, 'orientations':
            self._norm_orientations}
        self._coord_norm_fns = {'positions': self._coord_norm_positions,
            'orientations': self._coord_norm_orientations}
        self._unnorm_fns = {'positions': self._unnorm_positions,
            'orientations': self._unnorm_orientations}

    def _norm_positions(self, keys: List[str]) -> None:
        '''Truncates and rescales position data to range [.1, .9].

        Arguments:
            keys: The types of data to be normalized.
        '''
        n = 0
        s = torch.zeros(3)
        s2 = torch.zeros(3)
        for trial in self._trials:
            for part in trial['x'].values():
                for coords in part:
                    for key in keys:
                        if key in coords:
                            n += coords[key].shape[0]
                            s += coords[key].sum(0)
                            s2 += (coords[key] ** 2).sum(0)
        mn = s / n
        pstd = 3 * ((n * s2 - s ** 2) / (n * (n - 1))).sqrt()
        for trial in self._trials:
            for part in trial['x'].values():
                for coords in part:
                    for key in keys:
                        if key in coords:
                            coords[key] = (coords[key] - mn).clamp(min=-pstd,
                                max=pstd) * .4 / pstd + .5
        self._norm_params['positions'] = {'mn': mn, 'pstd': pstd}

    def _norm_orientations(self, keys: List[str]) -> None:
        '''Truncates and rescales orientation data to range [.1, .9].

        Arguments:
            keys: The types of data to be normalized.
        '''
        for trial in self._trials:
            for part in trial['x'].values():
                for coords in part:
                    for key in keys:
                        if key in coords:
                            coords[key] = (coords[key] / 2 + .5).clamp(min=.1,
                                max=.9)
    
    def _arrange_data(self) -> None:
        '''Rearranges the data to make it appropriate for a dataloader.'''
        self._init_object_ids()
        self._init_x_field_sizes()
        for trial in self._trials:
            trial['x'] = self._arrange_x(trial['x'])
            trial['classifs'] = [True if classif in trial['lbls'] else False
                for classif in self.classifs]
            no_lbls = -1 * torch.ones(trial['x'].shape[0], dtype=torch.int64)
            trial['lbls'] = torch.stack([trial['lbls'][classif] if classif in
                trial['lbls'] else no_lbls for classif in self.classifs]).T
        self._init_sizes()

    def _init_object_ids(self) -> None:
        '''Generates the member dictionary with the object identifiers.'''
        objects = self.all_lbls[self.classifs.index('leftmainobjects')]
        self._object_ids = {obj: id for obj, id in zip(objects, .8 * torch.eye(
            len(objects)) + .1)}

    def _init_x_field_sizes(self) -> None:
        '''Generates the member dictionary with the sample field sizes.'''
        fix_names, var_names = self._get_x_field_names()
        x = self._trials[0]['x']
        fix = OrderedDict([(name, None) for name in fix_names])
        for coords in x['fix']:
            fix[coords['name']] = OrderedDict([(coord, coords[coord].shape[1])
                for coord in fix_names[coords['name']]])
        id_size = list(self._object_ids.values())[0].shape[0]
        var = OrderedDict([(coord, id_size) if coord == 'id' else (coord,
            x['var'][0][coord].shape[1]) for coord in var_names])
        self._x_field_sizes = {'fix': fix, 'var': var}

    def _get_x_field_names(self) -> (typing.OrderedDict[str, List], List):
        '''Generates a set of lists with the sample field names.

        Returns:
            fix: The field names of the objects in the fixed part of the
                sample.
            var: The field names of the objects in the variable part of the
                sample.
        '''
        fix = OrderedDict([(obj, ['position', 'orientation']) for obj in
            self._fix_obj_names])
        var = ['id', 'position', 'orientation']
        return fix, var

    def _arrange_x(self, x: Dict[str, List[Dict[str, Any]]]) -> torch.Tensor:
        '''Builds the dataset samples from the structured data.

        Arguments:
            x: The names and coordinates (position and orientation) of the
                objects of a trial.

        Returns:
            x: The samples of a trial.
        '''
        fix = {coords['name']: torch.cat([coords[coord] for coord in
            self._x_field_sizes['fix'][coords['name']]], 1) for coords in x[
            'fix']}
        x_fix = [fix[obj] for obj in self._x_field_sizes['fix']]
        x_var = [torch.cat([self._object_ids[coords['name']].repeat([x_fix[
            0].shape[0], 1]) if coord == 'id' else coords[coord] for coord in
            self._x_field_sizes['var']], 1) for coords in x['var']]
        return torch.cat(x_fix + x_var, 1)

    def _init_sizes(self) -> None:
        '''Generates the member constants with the fixed and variable sizes.'''
        self.fix_size = sum(sum(coord for coord in coords.values()) for coords
            in self._x_field_sizes['fix'].values())
        self.var_size = sum(coord for coord in self._x_field_sizes[
            'var'].values())

    def _split_in_sets(self) -> None:
        '''Splits the data into a number of sets using stratification.'''
        n_sets = len(self.splitprops)
        self._splitsets = [[] for i in range(n_sets)]
        classifs = ['actions', 'plans', 'subjects', 'subsets']
        n_lbls = [len(self.all_lbls[self.classifs.index(classif)]) + 1 for
            classif in classifs]
        accum = torch.zeros(n_lbls + [n_sets])
        for trial in self._trials:
            lbls = trial['trial_lbls']
            accum_proj = accum
            accum_row = accum
            for classif in classifs:
                if classif in lbls:
                    accum_proj = accum_proj[lbls[classif]]
                    accum_row = accum_row[lbls[classif]]
                else:
                    accum_proj = accum_proj.sum(0)
                    accum_row = accum_row[-1]
            set_i = torch.argmin(accum_proj / torch.tensor(self.splitprops))
            self._splitsets[set_i].append(trial)
            accum_row[set_i] += 1

    def restart(self, params) -> None:
        '''Restarts the dataset, updating the current split set.

        Arguments:
            params: The restart function parameters. These parameters are
                mandatory and include:
                splitset: An integer with the identifier of the new split set.
        '''
        self.splitset = params['splitset']
        self._splitset = self._splitsets[params['splitset']]
        #self._new_trial = True

    def hard_restart(self, params) -> None:
        '''Performs a hard restart on the dataset, respliting the sets.'''
        raise NotImplementedError

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        '''Returns a sample or batch iterator.

        Returns:
            iter: The iterator returning the samples and associated labels.
        '''
        #return self
        if self.batch_size is None:
            trial = self._get_trial()
            if self.cat_x:
                for x, lbls in zip(trial['x'], trial['lbls']):
                    yield x, lbls[self.classif]
            else:
                x_fix, x_var = self.parse_data(trial['x'])
                for x_f, x_v, lbls in zip(x_fix, x_var, trial['lbls']):
                    yield x_f, x_v, lbls[self.classif]
        else:
            while True:
                trial = self._get_trial()
                lbls = trial['lbls']
                if self.cat_x:
                    x = trial['x']
                    for i in range(0, x.shape[0], self.batch_size):
                        yield x[i:i + self.batch_size], lbls[i:i +
                            self.batch_size, self.classif]
                else:
                    x_fix, x_var = self.parse_data(trial['x'])
                    for i in range(0, x_fix.shape[0], self.batch_size):
                        yield x_fix[i:i + self.batch_size], x_var[i:i +
                            self.batch_size], lbls[i:i + self.batch_size,
                            self.classif]

    def _get_trial(self) -> Dict[str, Any]:
        '''Returns a possibly augmented trial to be used by __iter__.

        Returns:
            trial: The samples and the labels corresponding to a trial.
        '''
        trial = random.choice(self._splitset)
        while self._discard_trial(trial):
            trial = random.choice(self._splitset)
        trial = copy.copy(trial)
        if self.mem_opt:
            trial['x'] = trial['x'].to(torch.float32)
            trial['x'][trial['x'] > 0] /= torch.finfo(torch.float16).max
            trial['x'][trial['x'] <= 0] *= torch.finfo(torch.float16).tiny / \
                torch.finfo(torch.float16).min
        if self.makeup_objs:
            trial['x'] = self._augment_with_objs(trial['x'], trial[
                'extra_info'])
        if self.random_mirror and random.getrandbits(1):
            self._mirror_trial(trial)
        if self.random_starts:
            start = random.randrange(32)
            trial['x'] = trial['x'][start:]
            trial['lbls'] = trial['lbls'][start:]
        return trial

    def _discard_trial(self, trial: Dict[str, Any]) -> bool:
        '''Checks if given trial satisfies the criteria to be pooled by __iter__.

        Arguments:
            trial: The trial to be checked.

        Returns:
            discard: True if the trial should be discarded.
        '''
        return any(trial['classifs'][classif] is False for classif in
            self.must_classifs) or any(any(trial['trial_lbls'][classif] == lbl
            for lbl in lbls) is False for classif, lbls in
            self.must_trial_lbls.items())

    def _augment_with_objs(self, x: torch.Tensor, trial_info: Dict[str, Any]) \
        -> torch.Tensor:
        '''Augments the trial samples by adding objects.

        Arguments:
            x: The trial samples to be augmented.
            trial_info: Extra information about the trial.

        Returns:
            x: The augmented trial samples.
        '''        
        n_obj_orig = (x.shape[1] - self.fix_size) // self.var_size
        n_obj_new = random.randint(0, self._max_n_objs - n_obj_orig)
        obj_names = random.choices(list(self._madeup_objs['zq'].keys()), k=
            n_obj_new)
        x_cats, static_pos = [], trial_info['static_pos']
        for name in obj_names:
            madeup_xys = torch.stack([xy for sbst_xys in self._madeup_objs[
                'xy'].values() for xy in random.choices(sbst_xys, k=32)])
            xy_weights = ((madeup_xys - static_pos) ** 2).sum(2).min(1)[0] ** 3
            xy = random.choices(madeup_xys, weights=xy_weights, k=1)[0]
            zq = random.choice(self._madeup_objs['zq'][name])
            coords = {'name': name, 'position': torch.cat([xy, zq[:, :1]], 1),
                'orientation': zq[:, 1:]}
            static_pos = torch.cat([static_pos, xy.unsqueeze(0)], 1)
            self._process_coords(coords, trial_info)
            self._norm_coords(coords)
            x_cats.append(self._arrange_coords(name, coords))
        return torch.cat([x] + x_cats, -1)

    def _process_coords(self, coords: Dict[str, torch.Tensor], trial_info:
        Dict[str, Any]) -> None:
        '''Processes the data of a single object and adds noise.

        Arguments:
            coords: The position and orientation of the object.
            trial_info: Extra information about the trial.
        '''
        coords['position'] = coords['position'].repeat([trial_info[
            'trial_size'], 1])
        coords['orientation'] = coords['orientation'].repeat([trial_info[
            'trial_size'], 1])
        self._add_static_noise(coords, {'position': 'position', 'orientation':
            'orientation'})

    @staticmethod
    def _add_static_noise(coords: Dict[str, torch.Tensor], types: Dict[str,
        str]) -> None:
        '''Adds noise similar to that found in static objects.

        Arguments:
            coords: The position and orientation of the object.
            types: The names and types of the coordinates (position or
                orientation).
        '''
        stds = {'position': .02, 'orientation': .0001}
        for key, coord in coords.items():
            if key in types:
                coords[key] += torch.normal(0, stds[types[key]], coords[
                    key].shape)

    def _norm_coords(self, coords: Dict[str, torch.Tensor]) -> None:
        '''Normalizes the data of a single object.

        Arguments:
            coords: The position and orientation of the object.
        '''
        for type, keys in self._norm_types.items():
            self._coord_norm_fns[type](coords, keys)

    def _coord_norm_positions(self, coords: Dict[str, torch.Tensor], keys:
        List[str]) -> None:
        '''Truncates and rescales position data to range [.1, .9].

        Arguments:
            coords: The position and orientation of the object.
            keys: The types of data to be normalized.
        '''
        mn = self._norm_params['positions']['mn']
        pstd = self._norm_params['positions']['pstd']
        for key in keys:
            if key in coords:
                coords[key] = (coords[key] - mn).clamp(min=-pstd, max=pstd) * \
                    .4 / pstd + .5

    def _coord_norm_orientations(self, coords: Dict[str, torch.Tensor], keys:
        List[str]) -> None:
        '''Truncates and rescales orientation data to range [.1, .9].

        Arguments:
            coords: The position and orientation of the object.
            keys: The types of data to be normalized.
        '''
        for key in keys:
            if key in coords:
                coords[key] = (coords[key] / 2 + .5).clamp(min=.1, max=.9)

    def _arrange_coords(self, obj_name: str, coords: Dict[str, torch.Tensor]) \
        -> None:
        '''Arranges the data of a single object.

        Arguments:
            obj_name: The name of the object.
            coords: The coordinates (position and orientation) of the object.

        Returns:
            obj_x: The part of the sample of a trial corresponding to the
                object.
        '''
        for name, coord in coords.items():
            if name != 'name':
                trial_size = coord.shape[0]
                break
        return torch.cat([self._object_ids[obj_name].repeat([trial_size, 1]) if
            coord == 'id' else coords[coord] for coord in self._x_field_sizes[
            'var']], 1)

    def _mirror_trial(self, trial: Dict[str, Any]) -> None:
        '''Mirrors all the objects of the scene.

        Arguments:
            trial: The trial to be mirrored.
        '''
        warnings.warn('Mirror augmentation not implemented for this dataset')

    #def __next__(self) -> Tuple[torch.Tensor, int]:
    #    '''Returns the next sample and label.

    #    Returns:
    #        sample: The next sample.
    #        label: The label associated to sample.
    #    '''
    #    if self._new_trial:
    #        self._trial = random.choice(self._splitset)
    #        while self._trial['classifs'][self.classif] is False:
    #            self._trial = random.choice(self._splitset)
    #        self._i = -1
    #        self._new_trial = False
    #    self._i += 1
    #    if self._i >= self._trial['x'].shape[0]:
    #        self._new_trial = True
    #        raise StopIteration
    #    else:
    #        return self._trial['x'][self._i], self._trial['lbls'][self._i,
    #            self.classif]

    def set_sample_iter(self) -> None:
        '''Sets the sample iterator as the default iterator.'''
        self.batch_size = None

    def set_batch_iter(self, batch_size: int) -> None:
        '''Sets the batch iterator as the default iterator.

        Arguments:
            batch_size: How many samples per batch to load.
        '''
        self.batch_size = batch_size

    def set_classif(self, classif: str) -> None:
        '''Sets the current classification.

        Arguments:
            classif: The current classification name (which is used in method
                __next__).
        '''
        self.classif = self.classifs.index(classif)

    def set_must_classifs(self, classifs: List[str]) -> None:
        '''Sets the current mandatory classifications.

        Arguments:
            classifs: The current mandatory classifications (method __iter__
                will only pool trials with the indicated classifications).
        '''
        self.must_classifs = [self.classifs.index(classif) for classif in
            classifs]

    def set_must_trial_lbls(self, lbls: Dict[str, List[str]]) -> None:
        '''Sets the current mandatory trial labels per classification.

        Arguments:
            lbls: The current mandatory labels per trial classification (method
                __iter__ will only pool trials with the indicated labels).
        '''
        self.must_trial_lbls = {classif: [self.all_lbls[self.classifs.index(
            classif)].index(lbl_str) for lbl_str in lbl_strs] for classif,
            lbl_strs in lbls.items()}

    def parse_data(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Separates a batch into its fixed and multiple variable components.

        Arguments:
            x: The batch as generated by the dataset object.

        Returns:
            x_fix: The fixed part of the batch.
            x_var: The variable part of the batch.
        '''
        return x[:, :self.fix_size], x[:, self.fix_size:].reshape(x.shape[0],
            -1, self.var_size)

    def recover_orig_data(self, x_fix: torch.Tensor, x_var: torch.Tensor, lbls:
        torch.Tensor, params: Dict[str, Any] = None) -> Dict[str, Dict[str,
        Any]]:
        '''Recovers the original unprocessed data from the given batch.

        Arguments:
            x_fix: The fixed part of the sample batch as generated by the
                dataset object.
            x_var: The variable part of the sample batch as generated by the
                dataset object.
            lbls: The label batch as generated by the dataset object.
            params: Optional parameters to better recover the data.

        Returns:
            batch: An estimation of the original batch before being processed.
        '''
        if params is None:
            params = {}
        batch = self._unarrange_data(x_fix, x_var, lbls, params)
        self._unnorm_data(batch['x'], params)
        self._unprocess_data(batch['x'], params)
        return batch

    def _unarrange_data(self, x_fix: torch.Tensor, x_var: torch.Tensor, lbls:
        torch.Tensor, params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        '''Recovers the data before _arrange_data.

        Arguments:
            x_fix: The fixed part of the sample batch as generated by the
                dataset object.
            x_var: The variable part of the sample batch as generated by the
                dataset object.
            lbls: The label batch as generated by the dataset object.
            params: Optional parameters to better recover the data.

        Returns:
            batch: The batch with the samples classified by object and type of
                data and the labels classified by object.
        '''
        batch = {}
        batch['x'] = self._unarrange_x(x_fix, x_var, params)
        batch['lbls'] = {classif: lbl for classif, lbl in zip(self.classifs,
            lbls.T) if lbl[0] != -1}
        trial_lbls = {'subjects': lbls[0, self.classifs.index('subjects')]}
        if lbls[0, self.classifs.index('plans')] != -1:
            trial_lbls['plans'] = lbls[0, self.classifs.index('plans')]
        else:
            trial_lbls['actions'] = lbls[0, self.classifs.index('actions')]
        batch['trial_lbls'] = trial_lbls
        return batch

    def _unarrange_x(self, x_fix: torch.Tensor, x_var: torch.Tensor, params:
        Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        '''Recovers the data before _arrange_x.

        Arguments:
            x_fix: The fixed part of the sample batch as generated by the
                dataset object.
            x_var: The variable part of the sample batch as generated by the
                dataset object.
            params: Optional parameters to better recover the data (not used).

        Returns:
            x: The batch with the data classified by object and type of data.
        '''
        x = {'fix': [], 'var': []}
        i = 0
        for obj_name, sizes in self._x_field_sizes['fix'].items():
            coords = {'name': obj_name}
            for coord, size in sizes.items():
                coords[coord] = x_fix[:, i:i + size]
                i += size
            x['fix'].append(coords)
        i = 0
        var_coords = {}
        for coord, size in self._x_field_sizes['var'].items():
            if coord == 'id':
                obj_is = torch.argmax(torch.mean(x_var[:, :, i:i + size], 0) @
                    torch.stack(list(self._object_ids.values())).T, 1)
            else:
                var_coords[coord] = x_var[:, :, i:i + size]
            i += size
        obj_names = list(self._object_ids)
        for i, obj_i in enumerate(obj_is):
            coords = {key: coord[:, i, :] for key, coord in var_coords.items()}
            coords['name'] = obj_names[obj_i]
            x['var'].append(coords)
        return x

    def _unnorm_data(self, x: Dict[str, List[Dict[str, Any]]], params: Dict[
        str, Any]) -> None:
        '''Recovers the data before _norm_data.

        Arguments:
            x: The batch as returned by _unarrange_data.
            params: Optional parameters to better recover the data (not used).
        '''
        for part in x.values():
            for coords in part:
                for type, keys in self._norm_types.items():
                    for key in keys:
                        if key in coords:
                            coords[key] = self._unnorm_fns[type](coords[key],
                                coords['name'], key, params)

    def _unnorm_positions(self, pos: torch.Tensor, obj : str, key: str, params:
        Dict[str, Any]) -> torch.Tensor:
        '''Recovers the original positions previous to normalization.

        Arguments:
            pos: The normalized position.
            obj: The object name.
            key: The normalized type of data.
            params: Parameters where the mask with the clamped elements is
                introduced.

        Returns:
            pos: The original unnormalized position.
        '''
        if 'clamp_masks' not in params:
            params['clamp_masks'] = {}
        if obj not in params['clamp_masks']:
            params['clamp_masks'][obj] = {}
        params['clamp_masks'][obj][key] = torch.any(torch.logical_or(pos > .89,
            pos < .11), 1)
        return ((pos - .5) * self._norm_params['positions']['pstd'] / .4) + \
            self._norm_params['positions']['mn']

    def _unnorm_orientations(self, rot: torch.Tensor, obj : str, key: str,
        params: Dict[str, Any]) -> torch.Tensor:
        '''Recovers the original orientations previous to normalization.

        Arguments:
            rot: The normalized orientation.
            obj: The object name.
            key: The normalized type of data.
            params: Optional parameters to better recover the data (not used).

        Returns:
            rot: The original unnormalized orientation.
        '''
        return rot * 2 - 1

    def _unprocess_data(self, x: Dict[str, List[Dict[str, Any]]], params: Dict[
        str, Any]) -> None:
        '''Recovers the data before _process_data.

        Arguments:
            x: The batch as returned by _unnorm_data.
            params: Optional parameters to better recover the data (not used).
        '''
        pass


class KitInput(KitRawInput):
    '''A PyTorch Dataset for the KIT Whole-Body Human Motion Database.'''
    def _process_data(self) -> None:
        '''Corrects positions and rotations and switches to egocentric view.'''
        self._init_coord_corrections()
        for trial in self._trials:
            x = trial['x']
            self._filter_outliers(x, 5)
            if self.makeup_objs:
                self._update_madeup_objs(x, trial['trial_lbls'])
                trial['extra_info']['static_pos'] = self._get_static_positions(
                    x)
            self._correct_coords(x)
            ref_info = self._set_ego_perspective(x)
            if self.makeup_objs:
                trial['extra_info'].update(ref_info)
            self._convert_orientations(x)
            for coords in x['var']:
                if 'handle_com' not in coords:
                    coords['handle_com'] = coords['main_com']
            trial['x'] = x

    def _init_coord_corrections(self) -> None:
        '''Generates the member dictionary with the coordinate corrections.'''
        self._coord_corrections = {
            'left_hand':         {'main_com': [0, 0, 0],
                                  'orientation': [('local_y', torch.pi / 2)]},
            'right_hand':        {'main_com': [0, 0, 0],
                                  'orientation': [('local_y', torch.pi / 2),
                                                  ('local_x', torch.pi)]},
            'head':              {'main_com': [0, 0, 0],
                                  'orientation': [('local_z', torch.pi / 2)]},
            'torso':             {'main_com': [0, 0, 0],
                                  'orientation': [('local_z', torch.pi / 2)]},
            'apple_juice':       {'main_com': [0, 120, 0],
                                  'orientation': [('local_x', -torch.pi / 2)]},
            'apple_juice_lid':   {'main_com': [0, 7, 0],
                                  'orientation': [('local_x', -torch.pi / 2)]},
            'broom':             {'main_com': [-1190, 0, 0],
                                  'handle_com': [-610, 0, 0],
                                  'orientation': [('local_x', -torch.pi / 2)]},
            'cucumber_attachment':
                                 {'main_com': [0, 0, 0],
                                  'orientation': [('local_y', torch.pi / 2),
                                                  ('local_x', -torch.pi / 2)]},
            'cup_large':         {'main_com': [0, 65, 0],
                                  'orientation': [('local_x', -torch.pi / 2)]},
            'cup_small':         {'main_com': [0, 45, 0],
                                  'orientation': [('local_x', -torch.pi / 2)]},
            'cutting_board_small':
                                 {'main_com': [0, 8, 0],
                                  'handle_com': [0, 8, -155],
                                  'orientation': [('local_y', torch.pi / 2),
                                                  ('local_x', -torch.pi / 2)]},
            'draining_rack':     {'main_com': [0, 21, 0],
                                  'orientation': [('local_x', -torch.pi / 2)]},
            'egg_whisk':         {'main_com': [0, 160, 45],
                                  'handle_com': [0, -63, 45],
                                  'orientation': [('local_z', -torch.pi / 2)]},
            'eggplant_attachment':
                                 {'main_com': [0, 0, 0],
                                  'orientation': [('local_z', -torch.pi / 2)]},
            'frying_pan':        {'main_com': [0, 28, 0],
                                  'handle_com': [215, 28, 0],
                                  'orientation': [('local_x', -torch.pi / 2)]},
            'knife_black':       {'main_com': [0, 100, 0],
                                  'handle_com': [0, -62, 0],
                                  'orientation': [('local_z', -torch.pi / 2),
                                                  ('local_x', torch.pi)]},
            'ladle':             {'main_com': [-25, 240, 12],
                                  'handle_com': [-25, 50, 12],
                                  'orientation': [('local_z', -torch.pi / 2),
                                                  ('local_x', torch.pi / 2)]},
            'milk_small':        {'main_com': [0, 70, 0],
                                  'orientation': [('local_x', -torch.pi / 2)]},
            'milk_small_lid':    {'main_com': [0, 5, 0],
                                  'orientation': [('local_x', -torch.pi / 2)]},
            'mixing_bowl_big':   {'main_com': [0, 28, 77],
                                  'orientation': []},
            'mixing_bowl_green': {'main_com': [0, 85, 0],
                                  'orientation': [('local_y', -torch.pi / 2),
                                                  ('local_x', -torch.pi / 2)]},
            'mixing_bowl_small': {'main_com': [0, 82, 0],
                                  'orientation': [('local_y', torch.pi),
                                                  ('local_x', -torch.pi / 2)]},
            'oil_bottle':        {'main_com': [0, 140, 0],
                                  'orientation': [('local_x', -torch.pi / 2)]},
            'peeler':            {'main_com': [-146, 0, 0],
                                  'handle_com': [-63, 0, 0],
                                  'orientation': [('local_x', -torch.pi / 2)]},
            'plate_dish':        {'main_com': [0, 13, 0],
                                  'orientation': [('local_x', -torch.pi / 2)]},
            'rolling_pin':       {'main_com': [0, 0, 0],
                                  'orientation': []},
            'salad_fork':        {'main_com': [0, 241, 0],
                                  'handle_com': [0, 84, 0],
                                  'orientation': [('local_z', -torch.pi / 2),
                                                  ('local_x', torch.pi / 2)]},
            'salad_spoon':       {'main_com': [0, 239, 0],
                                  'handle_com': [0, 87, 0],
                                  'orientation': [('local_z', -torch.pi / 2),
                                                  ('local_x', -torch.pi / 2)]},
            'spatula':           {'main_com': [0, 243, 0],
                                  'handle_com': [0, 60, 0],
                                  'orientation': [('local_z', -torch.pi / 2),
                                                  ('local_x', -torch.pi / 2)]},
            'sponge_small':      {'main_com': [5, 18, 9],
                                  'orientation': []},
            'tablespoon':        {'main_com': [0, 148, -3],
                                  'handle_com': [0, 44, -3],
                                  'orientation': [('local_z', -torch.pi / 2),
                                                  ('local_x', torch.pi / 2)]}}
        for coords in self._coord_corrections.values():
            coords['main_com'] = torch.tensor(coords['main_com'], dtype=
                torch.float32)
            if 'handle_com' in coords:
                coords['handle_com'] = torch.tensor(coords['handle_com'],
                    dtype=torch.float32)

    def _correct_coords(self, x: Dict[str, List[Dict[str, Any]]]) -> None:
        '''Switches to a more consistent reference system across objects.

        Arguments:
            x: The names and coordinates (position and orientation) of the
                objects of a trial.
        '''
        for part in x.values():
            for coords in part:
                self._coord_correct_coords(coords)

    @classmethod
    def _set_ego_perspective(cls, x: Dict[str, List[Dict[str, Any]]]) -> None:
        '''Obtains the coordinates with respect to the vertical of the torso.

        Arguments:
            x: The names and coordinates (position and orientation) of the
                objects of a trial.
        '''
        for torso_coords in x['fix']:
            if torso_coords['name'] == 'torso':
                break
        ref_com = torso_coords['main_com']
        ref_yaw = _get_yaw_from_quaternion(torso_coords['orientation'])
        ref_coords = {'ref_com': ref_com, 'ref_yaw': ref_yaw}
        for part in x.values():
            for coords in part:
                if coords['name'] != 'torso':
                    cls._coord_set_ego_perspective(coords, ref_coords)
        diff_com = _rot_vector_z(ref_com.diff(dim=0, prepend=ref_com[0:1]),
            -ref_yaw)
        diff_yaw = ref_yaw.diff(prepend=ref_yaw[0:1]).unsqueeze(1)
        diff_yaw[diff_yaw > torch.pi] -= 2 * torch.pi
        diff_yaw[diff_yaw < -torch.pi] += 2 * torch.pi
        del torso_coords['main_com']
        del torso_coords['orientation']
        torso_coords['diff_com'] = diff_com
        torso_coords['diff_yaw'] = diff_yaw
        #return {'ref_com': ref_com[0], 'ref_yaw': ref_yaw[0]}
        return {'ref_com': ref_com, 'ref_yaw': ref_yaw}

    @classmethod
    def _convert_orientations(cls, x: Dict[str, List[Dict[str, Any]]]) -> None:
        '''Converts angles to (co)sines and quaternions to rotation matrices.

        Arguments:
            x: The names and coordinates (position and orientation) of the
                objects of a trial.
        '''
        for part in x.values():
            for coords in part:
                if coords['name'] == 'head':
                    yaw = _get_yaw_from_quaternion(coords['orientation'])
                    pitch = _get_pitch_from_quaternion(coords['orientation'])
                    coords['orientation'] = torch.stack([yaw.sin(), yaw.cos(),
                        pitch.sin(), pitch.cos()]).T
                elif coords['name'] != 'torso':
                    cls._coord_convert_orientations(coords)

    def _define_norm_criteria(self) -> None:
        '''Defines criteria to normalize the different types of data.'''
        self._norm_types = {'positions': ['main_com', 'handle_com'],
            'orientations': ['orientation'], 'defaults': ['diff_com',
            'diff_yaw']}
        self._norm_fns = {'positions': self._norm_positions, 'orientations':
            self._norm_orientations, 'defaults': self._norm_defaults}
        self._coord_norm_fns = {'positions': self._coord_norm_positions,
            'orientations': self._coord_norm_orientations, 'defaults':
            self._coord_norm_defaults}
        self._unnorm_fns = {'positions': self._unnorm_positions,
            'orientations': self._unnorm_orientations, 'defaults':
            self._unnorm_defaults}

    def _norm_defaults(self, keys: str) -> None:
        '''Truncates and rescales the data to the range [.1, .9].

        Arguments:
            keys: The types of data to be normalized.
        '''
        for key in keys:
            n = 0
            s = 0
            s2 = 0
            for trial in self._trials:
                for part in trial['x'].values():
                    for coords in part:
                        if key in coords:
                            n += coords[key].shape[0]
                            s += coords[key].sum(0)
                            s2 += (coords[key] ** 2).sum(0)
            mn = s / n
            pstd = 3 * ((n * s2 - s ** 2) / (n * (n - 1))).sqrt()
            for trial in self._trials:
                for part in trial['x'].values():
                    for coords in part:
                        if key in coords:
                            coords[key] = (coords[key] - mn).clamp(min=-pstd,
                                max=pstd) * .4 / pstd + .5
            self._norm_params[key] = {'mn': mn, 'pstd': pstd}

    def _init_object_ids(self) -> Dict[str, torch.Tensor]:
        '''Generates the member dictionary with the object identifiers.'''
        obj_props = {           # size geom leng empt open cont hand shar mate
            'apple_juice':       ['b', 'p', 'z', 'y', 'y', 'y', 'n', 'n', 'm'],
            'apple_juice_lid':   ['s', 'c', 'm', 'y', 'y', 'y', 'n', 'n', 'm'],
            'broom':             ['b', 'p', 'y', 'n', 'n', 'n', 'l', 'n', 'h'],
            'cucumber_attachment':
                                 ['b', 'c', 'x', 'n', 'n', 'y', 'n', 'n', 'm'],
            'cup_large':         ['b', 'c', 'z', 'y', 'y', 'y', 'n', 'n', 'm'],
            'cup_small':         ['m', 'c', 'm', 'y', 'y', 'y', 'n', 'n', 'm'],
            'cutting_board_small':
                                 ['b', 'p', 'f', 'n', 'n', 'y', 's', 'n', 'h'],
            'draining_rack':     ['b', 'p', 'm', 'y', 'y', 'n', 'n', 'n', 'm'],
            'egg_whisk':         ['m', 's', 'x', 'y', 'n', 'n', 's', 'n', 'm'],
            'eggplant_attachment':
                                 ['b', 's', 'x', 'n', 'n', 'y', 'n', 'n', 'm'],
            'frying_pan':        ['b', 'c', 'f', 'y', 'y', 'y', 's', 'n', 'h'],
            'knife_black':       ['m', 'p', 'x', 'n', 'n', 'y', 's', 'y', 'h'],
            'ladle':             ['m', 's', 'm', 'y', 'y', 'y', 's', 'n', 'm'],
            'milk_small':        ['m', 'p', 'm', 'y', 'y', 'y', 'n', 'n', 'm'],
            'milk_small_lid':    ['s', 'c', 'm', 'y', 'y', 'y', 'n', 'n', 'm'],
            'mixing_bowl_big':   ['b', 's', 'm', 'y', 'y', 'y', 'n', 'n', 'm'],
            'mixing_bowl_green': ['b', 's', 'm', 'y', 'y', 'y', 'n', 'n', 'm'],
            'mixing_bowl_small': ['m', 'c', 'z', 'y', 'y', 'y', 'n', 'n', 'm'],
            'oil_bottle':        ['b', 'p', 'z', 'y', 'y', 'y', 'n', 'n', 'h'],
            'peeler':            ['s', 'c', 'f', 'n', 'n', 'n', 's', 'y', 'h'],
            'plate_dish':        ['b', 'c', 'f', 'y', 'y', 'y', 'n', 'n', 'm'],
            'rolling_pin':       ['b', 'c', 'x', 'n', 'n', 'y', 's', 'n', 'h'],
            'salad_fork':        ['m', 'p', 'f', 'y', 'y', 'n', 's', 'y', 'h'],
            'salad_spoon':       ['m', 's', 'f', 'y', 'y', 'y', 's', 'n', 'h'],
            'spatula':           ['m', 'p', 'f', 'n', 'n', 'y', 's', 'n', 'm'],
            'sponge_small':      ['m', 'p', 'm', 'n', 'n', 'n', 'n', 'n', 's'],
            'tablespoon':        ['s', 's', 'f', 'y', 'y', 'y', 's', 'n', 'h']}
        prop_lists = [['b', 'm', 's'], ['s', 'c', 'p'],
                      ['x', 'y', 'z', 'm', 'f'], ['y', 'n'], ['y', 'n'],
                      ['y', 'n'], ['l', 's', 'n'], ['y', 'n'], ['h', 'm', 's']]
        prop_dicts, i = [], 0
        for prop_list in prop_lists:
            prop_dicts.append({prop: id for id, prop in enumerate(prop_list,
                i)})
            i += len(prop_list)
        obj_ids = {}
        for obj_name, obj_prop in obj_props.items():
            obj_id = .1 * torch.ones(i)
            for prop_val, prop_dict in zip(obj_prop, prop_dicts):
                obj_id[prop_dict[prop_val]] = .9
            obj_ids[obj_name] = obj_id
        self._object_ids = obj_ids

    def _get_x_field_names(self) -> (typing.OrderedDict[str, List], List):
        '''Generates a set of lists with the sample field names.

        Returns:
            fix: The field names of the objects in the fixed part of the
                sample.
            var: The field names of the objects in the variable part of the
                sample.
        '''
        fix = OrderedDict([(obj, ['main_com', 'orientation']) for obj in
            self._fix_obj_names])
        fix['torso'] = ['diff_com', 'diff_yaw']
        var = ['id', 'main_com', 'handle_com', 'orientation']
        return fix, var

    #def _get_trial_info(self, x: torch.Tensor, trial_info: Dict[str, Any]) -> \
    #    Dict[str, Any]:
    #    '''Gets relevant trial information for the coordinates processing.

    #    Arguments:
    #        x: The trial samples to be augmented.
    #        trial_info: Extra information about the trial.

    #    Returns:
    #        trial_info: Extra information about the trial.
    #    '''
    #    ref_coords, i = {}, 0
    #    for obj_name, sizes in self._x_field_sizes['fix'].items():
    #        if obj_name == 'torso':
    #            for coord, size in sizes.items():
    #                ref_coords[coord] = x[:, i:i + size]
    #                i += size
    #            break
    #        for size in sizes.values():
    #            i += size
    #    for key in ['diff_com', 'diff_yaw']:
    #        for type, keys in self._norm_types.items():
    #            if key in keys:
    #                break
    #        ref_coords[key] = self._unnorm_fns[type](ref_coords[key], 'torso',
    #            key, {})
    #    self._unprocess_diff_torso(ref_coords)
    #    ref_com = (torch.cat([trial_info['ref_com'] + ref_coords['diff_com'][
    #        :1], ref_coords['diff_com'][1:]], 0)).cumsum(0)
    #    ref_yaw = (torch.cat([trial_info['ref_yaw'] + ref_coords['diff_yaw'][
    #        :1, 0], ref_coords['diff_yaw'][1:, 0]], 0)).cumsum(0)
    #    ref_yaw[ref_yaw > torch.pi] -= 2 * torch.pi
    #    ref_yaw[ref_yaw < -torch.pi] += 2 * torch.pi
    #    return {'ref_com': ref_com, 'ref_yaw': ref_yaw}

    def _unprocess_diff_torso(self, coords: Dict[str, torch.Tensor]) -> None:
        '''Recovers the differential coordinates of the torso.

        Arguments:
            coords: The position and orientation of the torso.
        '''
        pass

    def _process_coords(self, coords: Dict[str, torch.Tensor], trial_info:
        Dict[str, Any]) -> None:
        '''Processes the data of a single object and adds noise.

        Arguments:
            coords: The position and orientation of the object.
            trial_info: Extra information about the trial.
        '''
        self._coord_correct_coords(coords)
        self._add_static_noise(coords, {'main_com': 'position', 'handle_com':
            'position', 'orientation': 'orientation'})
        self._coord_set_ego_perspective(coords, trial_info)
        self._coord_convert_orientations(coords)
        if 'handle_com' not in coords:
            coords['handle_com'] = coords['main_com']

    def _coord_correct_coords(self, coords: Dict[str, torch.Tensor]) -> None:
        '''Switches to a more consistent reference system across objects.

        Arguments:
            coords: The position and orientation of the object.
        '''
        pos, rot = coords['position'], coords['orientation']
        crctns = self._coord_corrections[coords['name']]
        del coords['position']
        coords['main_com'] = pos + _rot_vector_quaternion(crctns['main_com'],
            rot)
        if 'handle_com' in crctns:
            coords['handle_com'] = pos + _rot_vector_quaternion(crctns[
                'handle_com'], rot)
        for axis, angle in crctns['orientation']:
            rot = _rot_quaternion_axis(rot, axis, angle)
        coords['orientation'] = rot

    @staticmethod
    def _coord_set_ego_perspective(coords: Dict[str, torch.Tensor], ref_coords:
        Dict[str, torch.Tensor]) -> None:
        '''Obtains the coordinates with respect to the vertical of the torso.

        Arguments:
            coords: The position and orientation of the object.
            ref_coords: The initial position and yaw of the torso.
        '''
        ref_com, ref_yaw = ref_coords['ref_com'], ref_coords['ref_yaw']
        coords['main_com'] = _rot_vector_z(coords['main_com'] - ref_com,
            -ref_yaw)
        if 'handle_com' in coords:
            coords['handle_com'] = _rot_vector_z(coords['handle_com'] -
                ref_com, -ref_yaw)
        coords['orientation'] = _rot_quaternion_axis(coords['orientation'],
            'global_z', -ref_yaw)

    @staticmethod
    def _coord_convert_orientations(coords: Dict[str, torch.Tensor]) -> None:
        '''Converts angles to (co)sines and quaternions to rotation matrices.

        Arguments:
            coords: The position and orientation of the object.
        '''
        coords['orientation'] = _get_rot_matrix_from_quaternion(coords[
            'orientation'])

    def _coord_norm_defaults(self, coords: Dict[str, torch.Tensor], keys: List[
        str]) -> None:
        '''Truncates and rescales the data to the range [.1, .9].

        Arguments:
            coords: The position and orientation of the object.
            keys: The types of data to be normalized.
        '''
        pass
        #for key in keys:
        #    if key in coords:
        #        mn = self._norm_params[key]['mn']
        #        pstd = self._norm_params[key]['pstd']
        #        coords[key] = (coords[key] - mn).clamp(min=-pstd, max=pstd) * \
        #            .4 / pstd + .5

    def _unnorm_defaults(self, data: torch.Tensor, obj: str, key: str, params:
        Dict[str, Any]) -> torch.Tensor:
        '''Recovers the original data previous to normalization.

        Arguments:
            data: The normalized data.
            obj: The object name.
            key: The normalized type of data.
            params: Optional parameters to better recover the data (not used).

        Returns:
            data: The original unnormalized data.
        '''
        return ((data - .5) * self._norm_params[key]['pstd'] / .4) + \
            self._norm_params[key]['mn']

    def _unprocess_data(self, x: Dict[str, List[Dict[str, Any]]], params: Dict[
        str, Any]) -> None:
        '''Recovers the data before _process_data.

        Arguments:
            x: The batch as returned by _unnorm_data.
            params: Parameters containing the reference position and yaw at
                batch start as well as the mask with the clamped elements.
        '''
        self._unconvert_orientations(x)
        self._unset_ego_perspective(x, params)
        self._uncorrect_coords(x)

    @staticmethod
    def _unconvert_orientations(x: Dict[str, List[Dict[str, Any]]]) -> None:
        '''Recovers the data before _correct_orientations.

        Arguments:
            x: The batch as returned by _unnorm_data.
        '''
        for part in x.values():
            for coords in part:
                if coords['name'] == 'head':
                    yaw = torch.atan2(coords['orientation'][:, 0], coords[
                        'orientation'][:, 1])
                    pitch = torch.atan2(coords['orientation'][:, 2], coords[
                        'orientation'][:, 3])
                    coords['orientation'] = _get_quaternion_from_yaw_pitch(yaw,
                        pitch)
                elif coords['name'] != 'torso':
                    coords['orientation'] = _get_quaternion_from_rot_matrix(
                        _unclamp_rot_matrix(coords['orientation']))

    @staticmethod
    def _unset_ego_perspective(x: Dict[str, List[Dict[str, Any]]], params:
        Dict[str, Any]) -> None:
        '''Recovers the data before _set_ego_perspective.

        Arguments:
            x: The batch as returned by _unconvert_orientations.
            params: Parameters containing the reference position and yaw at
                batch start as well as the mask with the clamped elements.
        '''
        if 'ref_com' not in params:
            params['ref_com'] = torch.tensor([[1000, 700, 1500]])
        if 'ref_yaw' not in params:
            params['ref_yaw'] = torch.tensor([-2.6])
        for torso_coords in x['fix']:
            if torso_coords['name'] == 'torso':
                break
        ref_yaw = (torch.cat([params['ref_yaw'] + torso_coords['diff_yaw'][:1,
            0], torso_coords['diff_yaw'][1:, 0]], 0)).cumsum(0)
        ref_yaw[ref_yaw > torch.pi] -= 2 * torch.pi
        ref_yaw[ref_yaw < -torch.pi] += 2 * torch.pi
        diff_pos = _rot_vector_z(torso_coords['diff_com'], ref_yaw)
        ref_pos = (torch.cat([params['ref_com'] + diff_pos[:1], diff_pos[1:]],
            0)).cumsum(0)
        params.update({'ref_com': ref_pos[-1:], 'ref_yaw': ref_yaw[-1:]})
        del torso_coords['diff_com']
        del torso_coords['diff_yaw']
        torso_coords['main_com'] = ref_pos
        torso_coords['orientation'] = _get_quaternion_from_yaw(ref_yaw)
        for part in x.values():
            for coords in part:
                if coords['name'] != 'torso':
                    coords['main_com'] = ref_pos + _rot_vector_z(coords[
                        'main_com'], ref_yaw)
                    coords['orientation'] = _rot_quaternion_axis(coords[
                        'orientation'], 'global_z', ref_yaw)
        if 'clamp_masks' in params:
            for coords in x['var']:
                coords['main_com'] = _unclamp_position(coords['main_com'],
                    params['clamp_masks'][coords['name']]['main_com'])

    def _uncorrect_coords(self, x: Dict[str, List[Dict[str, Any]]]) -> None:
        '''Recovers the data before _correct_coords.

        Arguments:
            x: The batch as returned by _unset_ego_perspective.
        '''
        for part in x.values():
            for coords in part:
                pos = coords['main_com']
                rot = coords['orientation']
                crctns = self._coord_corrections[coords['name']]
                for axis, angle in reversed(crctns['orientation']):
                    rot = _rot_quaternion_axis(rot, axis, -angle)
                del coords['main_com']
                coords['position'] = pos - _rot_vector_quaternion(crctns[
                    'main_com'], rot)
                coords['orientation'] = rot


class KitSparseInput(KitInput):
    '''A PyTorch Dataset for the KIT Whole-Body Human Motion Database.

    Attributes:
        _sparse_params: A dictionary of strings to tensors or lists with the
            parameters used to obtain the sparse representations of the
            position and orientation coordinates.
    '''
    def _process_data(self) -> None:
        '''Corrects positions and rotations and switches to egocentric view.'''
        self._init_coord_corrections()
        for trial in self._trials:
            x = trial['x']
            self._filter_outliers(x, 5)
            if self.makeup_objs:
                self._update_madeup_objs(x, trial['trial_lbls'])
                trial['extra_info']['static_pos'] = self._get_static_positions(
                    x)
            self._correct_coords(x)
            ref_info = self._set_ego_perspective(x)
            if self.makeup_objs:
                trial['extra_info'].update(ref_info)
            self._build_sparse_representations(x)
            for coords in x['var']:
                if 'handle_com' not in coords:
                    coords['handle_com'] = coords['main_com']
            trial['x'] = x

    def _build_sparse_representations(self, x: Dict[str, List[Dict[str,
        Any]]]) -> None:
        '''Obtains sparse representations of the coordinates.

        Arguments:
            x: The names and coordinates (position and orientation) of the
                objects of a trial.
        '''
        self._sparse_params = {}
        self._build_sparse_torso(x)
        self._build_sparse_positions(x)
        self._build_sparse_orientations(x)

    def _build_sparse_torso(self, x: Dict[str, List[Dict[str, Any]]]) -> None:
        '''Obtains sparse representations of the torso coordinates.

        Arguments:
            x: The names and coordinates (position and orientation) of the
                objects of a trial.
        '''
        com_params = {'ref': torch.tensor([0, 0, 0]),
                      'shrink_factors': torch.tensor([1 / 4] * 3),
                      'stds': [.5] * 3,
                      'grid_sizes': [3] * 3,
                      'combine_xy': True}
        yaw_params = {'ref': 0,
                      'shrink_factors': 64,
                      'stds': [.5],
                      'grid_sizes': [3],
                      'combine_xy': False}
        self._complete_sparse_position_params(com_params)
        self._complete_sparse_position_params(yaw_params)
        self._sparse_params['torso'] = {'com': com_params, 'yaw': yaw_params}
        for torso_coords in x['fix']:
            if torso_coords['name'] == 'torso':
                break
        torso_coords['diff_com'] = self._build_sparse_position(torso_coords[
            'diff_com'], com_params)
        torso_coords['diff_yaw'] = self._build_sparse_position(torso_coords[
            'diff_yaw'], yaw_params)

    def _build_sparse_positions(self, x: Dict[str, List[Dict[str, Any]]]) -> \
        None:
        '''Obtains sparse representations of the position coordinates.

        Arguments:
            x: The names and coordinates (position and orientation) of the
                objects of a trial.
        '''
        params = {'ref': torch.tensor([300, 0, -400]),
                  'shrink_factors': torch.tensor([1 / 512] * 3),
                  'stds': [.25] * 3,
                  'grid_sizes': [5] * 3,
                  'combine_xy': True}
        self._complete_sparse_position_params(params)
        self._sparse_params['position'] = params
        for parts in x.values():
            for coords in parts:
                if coords['name'] != 'torso':
                    self._coord_build_sparse_positions(coords)

    @staticmethod
    def _complete_sparse_position_params(params: Dict[str, Any]) -> None:
        '''Generates the dependent parameters used for the sparse positions.

        Arguments:
            params: The parameters used to obtain the sparse representations
                of the position coordinates.
        '''
        params['ks'] = [-1 / (2 * std ** 2) for std in params['stds']]
        params['coord_slices'] = torch.cat([torch.tensor([0]), torch.tensor(
            params['grid_sizes']).cumsum(0)]).contiguous().as_strided((len(
            params['grid_sizes']), 2), (1, 1)).contiguous()
        params['grids'] = [torch.linspace(-1, 1, size).reshape(-1, 1) for size
            in params['grid_sizes']]
        if params['combine_xy']:
            xy_size = params['grid_sizes'][0] * params['grid_sizes'][1]
            params['coord_slices'][2:] += xy_size - params['coord_slices'][1,
                1]
            params['coord_slices'][1, 0] = 0
            params['coord_slices'][0:2, 1] = xy_size
            #params['recover_grids'] = [params['grids'][0].repeat_interleave(
            #    params['grid_sizes'][1], 0), params['grids'][1].repeat(params[
            #    'grid_sizes'][0], 1), params['grids'][2]]
        #else:
            #params['recover_grids'] = params['grids']

    @staticmethod
    def _build_sparse_position(coords: torch.Tensor, params: Dict[str, Any]) \
        -> torch.Tensor:
        '''Obtains a sparse representation of the given position coordinates.

        Arguments:
            coords: The position coordinates to be made sparse.
            params: The parameters used to obtain the sparse representations
                of the position coordinates.

        Returns:
            coords: The resultant sparse position coordinates.
        '''
        squashed = torch.tanh(params['shrink_factors'] * (coords - params[
            'ref']))
        discreted = []
        for coord, k, grid in zip(squashed.T, params['ks'], params['grids']):
            discreted.append(torch.exp(k * (grid - coord) ** 2).T)
        if params['combine_xy']:
            sizes = params['grid_sizes']
            xy = (discreted[0].reshape(-1, sizes[0], 1) * discreted[1].reshape(
                -1, 1, sizes[1])).reshape(-1, sizes[0] * sizes[1])
            return torch.cat([xy, discreted[2]], 1)
        else:
            return torch.cat(discreted, 1)

    def _build_sparse_orientations(self, x: Dict[str, List[Dict[str, Any]]]) \
        -> None:
        '''Obtains sparse representations of the orientation coordinates.

        Since a dodecahedron is dual to an icosahedron, the unitary vectors
        pointing to the face centers of a regular dodecahedron are the same as
        those pointing to the vertices of a regular icosahedron.
        Source: https://en.wikipedia.org/wiki/Regular_icosahedron

        Arguments:
            x: The names and coordinates (position and orientation) of the
                objects of a trial.
        '''
        phi = (1 + torch.sqrt(torch.tensor(5))) / 2
        vectors = torch.tensor([1, phi]) / torch.sqrt(1 + phi ** 2)
        vectors = torch.cat([torch.zeros(4, 1), torch.cartesian_prod(*([
            torch.tensor([-1, 1])] * 2)) * vectors], 1)
        vectors = torch.cat([vectors, vectors.roll(1, 1), vectors.roll(2, 1)],
            0)
        params = {'k': 1 / .5 ** 2, 'vectors': vectors}
        params['ks'] = [params['k'], 2 * params['k'], 4 * params['k']]
        params['Cs'] = [1 / torch.exp(torch.tensor(k)) for k in params['ks']]
        self._define_symmetries(params)
        params['neighbors'] = ((vectors @ vectors.T) > 0).nonzero(as_tuple=
            True)[1].reshape(-1, 6)
        params['opposites'] = (vectors @ vectors.T).argmin(-1)
        self._sparse_params['orientation'] = params
        for part in x.values():
            for coords in part:
                if coords['name'] != 'torso':
                    self._coord_build_sparse_orientations(coords)
        if self.random_mirror:
            self._mirror_params['orientation'] = ((vectors * torch.tensor([1,
                -1, 1])) @ vectors.T).argmax(1)

    @staticmethod
    def _define_symmetries(params: Dict[str, Any]) -> torch.Tensor:
        '''Defines the symmetries existing in the objects.

        Arguments:
            params: The parameters used to obtain the sparse representations
                of the axes.
        '''
        params['symmetries'] = {'left_hand': [None, None],
                                'right_hand': [None, None],
                                'head': [None, None],
                                'torso': [None, None],
                                'apple_juice': ['90', None],
                                'apple_juice_lid': ['sph', 'sph'],#['crc', None],
                                'broom': [None, None],
                                'cucumber_attachment': ['180', 'crc'],
                                'cup_large': ['crc', None],
                                'cup_small': ['crc', None],
                                'cutting_board_small': [None, '180'],
                                'draining_rack': [None, None],
                                'egg_whisk': [None, None],
                                'eggplant_attachment': [None, 'crc'],
                                'frying_pan': [None, None],
                                'knife_black': [None, None],
                                'ladle': [None, None],
                                'milk_small': ['90', None],
                                'milk_small_lid': ['sph', 'sph'],#['crc', None],
                                'mixing_bowl_big': ['crc', None],
                                'mixing_bowl_green': ['crc', None],
                                'mixing_bowl_small': ['crc', None],
                                'oil_bottle': ['90', None],
                                'peeler': [None, None],
                                'plate_dish': ['crc', None],
                                'rolling_pin': ['180', 'crc'],
                                'salad_fork': [None, None],
                                'salad_spoon': [None, None],
                                'spatula': [None, None],
                                'sponge_small': ['180', '180'],
                                'tablespoon': [None, None]}

    @staticmethod
    def _build_sparse_axis(axis: torch.Tensor, params: Dict[str, Any],
        symmetry: str, symmetry_axis: torch.Tensor) -> torch.Tensor:
        '''Obtains a sparse representation of the given axis.

        C has been chosen so that the exponential takes a value of 1 at its
        maximum.
        Source: https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution

        Arguments:
            axis: The axes to be made sparse.
            params: The parameters used to obtain the sparse representations
                of the axes.
            symmetry: The symmetry of the object defining this axis
                representation.
            symmetry_axis: The axes along which the symmetry is defined.

        Returns:
            axis: The resultant sparse axes.
        '''
        if symmetry == 'sph':
            return .1 * torch.ones(axis.shape[0], params['vectors'].shape[0])
        if symmetry == 'crc':
            C, k = params['Cs'][2], params['ks'][2]
            exps = [torch.sqrt((1 - (symmetry_axis @ params['vectors'].T) **
                2).clamp(0))]
        else:
            exps = [axis @ params['vectors'].T]
            if symmetry is None:
                C, k = params['Cs'][0], params['ks'][0]
            elif symmetry == '180':
                C, k = params['Cs'][1], params['ks'][1]
                exps.append(-exps[0])
            elif symmetry == '90':
                C, k = params['Cs'][2], params['ks'][2]
                exps.append(torch.linalg.cross(axis, symmetry_axis) @ params[
                    'vectors'].T)
                exps.extend([-exps[0], -exps[1]])
            else:
                raise ValueError(f'{symmetry} symmetry is unknown')
        return C * sum(torch.exp(k * exp) for exp in exps)

    def _define_norm_criteria(self) -> None:
        '''Defines criteria to normalize the different types of data.'''
        self._norm_types = {'raws': ['main_com', 'handle_com',
            'handle_axis', 'top_axis', 'diff_com', 'diff_yaw']}
        self._norm_fns = {'raws': self._norm_raws}
        self._coord_norm_fns = {'raws': self._coord_norm_raws}
        self._unnorm_fns = {'raws': self._unnorm_raws}

    def _norm_raws(self, keys: str) -> None:
        '''Leaves data as is.

        Arguments:
            keys: The types of data to be normalized.
        '''
        pass

    def _get_x_field_names(self) -> (typing.OrderedDict[str, List], List):
        '''Generates a set of lists with the sample field names.

        Returns:
            fix: The field names of the objects in the fixed part of the
                sample.
            var: The field names of the objects in the variable part of the
                sample.
        '''
        fix = OrderedDict([(obj, ['main_com', 'handle_axis', 'top_axis']) for
            obj in ['left_hand', 'right_hand']])
        fix['head'] = ['main_com', 'handle_axis']
        fix['torso'] = ['diff_com', 'diff_yaw']
        var = ['id', 'main_com', 'handle_com', 'handle_axis', 'top_axis']
        return fix, var

    def _unprocess_diff_torso(self, coords: Dict[str, torch.Tensor]) -> None:
        '''Recovers the differential coordinates of the torso.

        Arguments:
            coords: The position and orientation of the torso.
        '''
        coords['diff_com'] = self._unbuild_sparse_position(coords['diff_com'],
            self._sparse_params['torso']['com'])
        coords['diff_yaw'] = self._unbuild_sparse_position(coords['diff_yaw'],
            self._sparse_params['torso']['yaw'])

    def _process_coords(self, coords: Dict[str, torch.Tensor], trial_info:
        Dict[str, Any]) -> None:
        '''Processes the data of a single object and adds noise.

        Arguments:
            coords: The position and orientation of the object.
            trial_info: Extra information about the trial.
        '''
        self._coord_correct_coords(coords)
        self._add_static_noise(coords, {'main_com': 'position', 'handle_com':
            'position', 'orientation': 'orientation'})
        self._coord_set_ego_perspective(coords, trial_info)
        self._coord_build_sparse_representations(coords)
        if 'handle_com' not in coords:
            coords['handle_com'] = coords['main_com']

    def _coord_build_sparse_representations(self, coords: Dict[str,
        torch.Tensor]) -> None:
        '''Obtains sparse representations of the coordinates.

        Arguments:
            coords: The position and orientation of the object.
        '''
        self._coord_build_sparse_positions(coords)
        self._coord_build_sparse_orientations(coords)

    def _coord_build_sparse_positions(self, coords: Dict[str, torch.Tensor]) \
        -> None:
        '''Obtains sparse representations of the position coordinates.

        Arguments:
            coords: The position and orientation of the object.
        '''
        coords['main_com'] = self._build_sparse_position(coords['main_com'],
            self._sparse_params['position'])
        if 'handle_com' in coords:
            coords['handle_com'] = self._build_sparse_position(coords[
                'handle_com'], self._sparse_params['position'])

    def _coord_build_sparse_orientations(self, coords: Dict[str,
        torch.Tensor]) -> None:
        '''Obtains sparse representations of the orientation coordinates.

        Arguments:
            coords: The position and orientation of the object.
        '''
        axes = _get_rot_matrix_from_quaternion(coords['orientation']).reshape(
            -1, 3, 3)
        x, z = axes[:, :, 0], axes[:, :, 2]
        params = self._sparse_params['orientation']
        symmetries = params['symmetries'][coords['name']]
        coords['handle_axis'] = self._build_sparse_axis(x, params, symmetries[
            0], z)
        if coords['name'] != 'head':
            coords['top_axis'] = self._build_sparse_axis(z, params, symmetries[
                1], x)
        del coords['orientation']

    def _coord_norm_raws(self, coords: Dict[str, torch.Tensor], keys: List[
        str]) -> None:
        '''Leaves data as is.

        Arguments:
            coords: The position and orientation of the object.
            keys: The types of data to be normalized.
        '''
        pass

    def _mirror_trial(self, trial: Dict[str, Any]) -> None:
        '''Mirrors all the objects of the scene.

        Arguments:
            trial: The trial to be mirrored.
        '''
        torso_x = self._sparse_params['torso']['com']['grid_sizes'][0]
        torso_y = self._sparse_params['torso']['com']['grid_sizes'][1]
        torso_xy = torso_x * torso_y
        pos_x = self._sparse_params['position']['grid_sizes'][0]
        pos_y = self._sparse_params['position']['grid_sizes'][1]
        pos_xy = pos_x * pos_y
        rot_map = self._mirror_params['orientation']
        x_fix, x_var = self.parse_data(trial['x'])
        x = self._unarrange_x(x_fix, x_var, None)
        for part in x.values():
            for coords in part:
                if coords['name'] == 'torso':
                    coords['diff_com'][:, :torso_xy] = coords['diff_com'][:,
                        :torso_xy].reshape(-1, torso_x, torso_y).flip(
                        2).reshape(-1, torso_xy)
                    coords['diff_yaw'] = coords['diff_yaw'].flip(1)
                else:
                    coords['main_com'][:, :pos_xy] = coords['main_com'][:,
                        :pos_xy].reshape(-1, pos_x, pos_y).flip(2).reshape(-1,
                        pos_xy)
                    coords['handle_axis'] = coords['handle_axis'][:, rot_map]
                    if coords['name'] != 'head':
                        coords['top_axis'] = coords['top_axis'][:, rot_map]
                        if coords['name'] == 'left_hand':
                            coords['name'] = 'right_hand'
                        elif coords['name'] == 'right_hand':
                            coords['name'] = 'left_hand'
                        else:
                            coords['handle_com'][:, :pos_xy] = coords[
                                'handle_com'][:, :pos_xy].reshape(-1, pos_x,
                                pos_y).flip(2).reshape(-1, pos_xy)
        trial['x'] = self._arrange_x(x)
        trial['lbls'] = trial['lbls'][:, self._mirror_params['lbls']]

    def _unnorm_raws(self, data: torch.Tensor, obj: str, key: str, params:
        Dict[str, Any]) -> torch.Tensor:
        '''Recovers the original data previous to normalization.

        Arguments:
            data: The normalized data.
            obj: The object name.
            key: The normalized type of data.
            params: Optional parameters to better recover the data (not used).

        Returns:
            data: The original unnormalized data.
        '''
        return data

    def _unprocess_data(self, x: Dict[str, List[Dict[str, Any]]], params: Dict[
        str, Any]) -> None:
        '''Recovers the data before _process_data.

        Arguments:
            x: The batch as returned by _unnorm_data.
            params: Parameters containing the reference position and yaw at
                batch start as well as the mask with the clamped elements.
        '''
        self._unbuild_sparse_representations(x)
        self._unset_ego_perspective(x, params)
        self._uncorrect_coords(x)

    def _unbuild_sparse_representations(self, x: Dict[str, List[Dict[str,
        Any]]]) -> None:
        '''Recovers the data before _build_sparse_representations.

        Arguments:
            x: The batch as returned by _unnorm_data.
        '''
        self._unbuild_sparse_torso(x)
        self._unbuild_sparse_positions(x)
        self._unbuild_sparse_orientations(x)

    def _unbuild_sparse_torso(self, x: Dict[str, List[Dict[str, Any]]]) -> \
        None:
        '''Recovers the data before _build_sparse_torso.

        Arguments:
            x: The batch as returned by _unnorm_data.
        '''
        for coords in x['fix']:
            if coords['name'] == 'torso':
                break
        self._unprocess_diff_torso(coords)

    def _unbuild_sparse_positions(self, x: Dict[str, List[Dict[str, Any]]]) \
        -> None:
        '''Recovers the data before _build_sparse_positions.

        Arguments:
            x: The batch as returned by _unbuild_sparse_torso.
        '''
        params = self._sparse_params['position']
        for part in x.values():
            for coords in part:
                if coords['name'] != 'torso':
                    coords['main_com'] = self._unbuild_sparse_position(coords[
                        'main_com'], params)

    @staticmethod
    def _unbuild_sparse_position(coords: torch.Tensor, params: Dict[str,
        Any]) -> torch.Tensor:
        '''Recovers the dense representation of the given position coordinates.

        Arguments:
            coords: The sparse position coordinates to be recovered.
            params: The parameters used to obtain the sparse representations
                of the position coordinates.

        Returns:
            coords: The recovered position coordinates.
        '''
        coords = coords.clamp(torch.finfo(torch.float32).tiny)
        discreted = [coords[:, lims[0]:lims[1]] for lims in params[
            'coord_slices']]
        ks = params['ks']
        grids = params['grids']
        squashed = []
        if params['combine_xy']:
            coord = discreted[0].reshape(-1, params['grid_sizes'][1], params[
                'grid_sizes'][0])
            exp = coord.log()
            grid = grids[0].reshape(-1, params['grid_sizes'][0], 1)
            combs = torch.combinations(torch.arange(params['grid_sizes'][
                0])).T
            guesses = -(((exp[:, combs[0], :] - exp[:, combs[1], :]) / ks[0] -
                (grid[:, combs[0], :] ** 2 - grid[:, combs[1], :] ** 2)) / (2 *
                (grid[:, combs[0], :] - grid[:, combs[1], :])))
            weights = coord[:, combs[0], :] * coord[:, combs[1], :]
            squashed.append((guesses * weights).sum([1, 2]) / weights.sum([1,
                2]))
            #squashed.append(guesses.mean([1, 2]))
            grid = grids[1].reshape(-1, 1, params['grid_sizes'][1])
            combs = torch.combinations(torch.arange(params['grid_sizes'][
                1])).T
            guesses = -(((exp[:, :, combs[0]] - exp[:, :, combs[1]]) / ks[1] -
                (grid[:, :, combs[0]] ** 2 - grid[:, :, combs[1]] ** 2)) / (2 *
                (grid[:, :, combs[0]] - grid[:, :, combs[1]])))
            weights = coord[:, :, combs[0]] * coord[:, :, combs[1]]
            squashed.append((guesses * weights).sum([1, 2]) / weights.sum([1,
                2]))
            #squashed.append(guesses.mean([1, 2]))
            discreted = discreted[2:]
            ks = ks[2:]
            grids = grids[2:]
        for coord, k, grid in zip(discreted, ks, grids):
            exp = coord.log()
            grid = grid.T
            combs = torch.combinations(torch.arange(grid.shape[1])).T
            guesses = -(((exp[:, combs[0]] - exp[:, combs[1]]) / k - (grid[:,
                combs[0]] ** 2 - grid[:, combs[1]] ** 2)) / (2 * (grid[:,
                combs[0]] - grid[:, combs[1]])))
            weights = coord[:, combs[0]] * coord[:, combs[1]]
            squashed.append((guesses * weights).sum(1) / weights.sum(1))
            #squashed.append(guesses.mean(1))
        squashed = torch.stack(squashed).T
        #for coord, grid in zip(discreted, params['recover_grids']):
        #    squashed.append(coord @ grid / coord.sum(1, keepdim=True))
        #squashed = torch.cat(squashed, 1)
        squashed[squashed >= 1] = .9
        squashed[squashed <= -1] = -.9
        return torch.atanh(squashed) / params['shrink_factors'] + params['ref']

    def _unbuild_sparse_orientations(self, x: Dict[str, List[Dict[str,
        Any]]]) -> None:
        '''Recovers the data before _build_sparse_orientations.

        Arguments:
            x: The batch as returned by _unbuild_sparse_positions.
        '''
        iiss = [[0, 1, 4, 5, 8, 9], [3, 2, 7, 6, 11, 10]]
        params = self._sparse_params['orientation']
        for part in x.values():
            for coords in part:
                if coords['name'] != 'torso':
                    symmetry = params['symmetries'][coords['name']]
                    if symmetry[0] == '180':
                        x = self._unbuild_sparse_orientation_180(coords[
                            'handle_axis'])
                    else:
                        x = F.normalize(coords['handle_axis'] @ params[
                            'vectors'])
                    del coords['handle_axis']
                    if coords['name'] == 'head':
                        y = torch.linalg.cross(torch.tensor([[0, 0, 1.]]), x)
                        z = torch.linalg.cross(x, y)
                    else:
                        if symmetry[1] == '180':
                            z = self._unbuild_sparse_orientation_180(coords[
                            'top_axis'])
                        else:
                            z = F.normalize(coords['top_axis'] @ params[
                                'vectors'])
                        if symmetry[0] == 'sph':
                            x = torch.zeros_like(x) + torch.tensor([1, 0, 0])
                            z = torch.zeros_like(x) + torch.tensor([0, 0, 1])
                        elif symmetry[0] in ['90', 'crc']:
                            x = torch.linalg.cross(torch.tensor([[0, 1., 0]]),
                                z)
                        elif symmetry[1] in ['90', 'crc']:
                            z = torch.linalg.cross(x, torch.tensor([[0, 1.,
                                0]]))
                        y = torch.linalg.cross(z, x)
                        del coords['top_axis']
                    coords['orientation'] = _get_quaternion_from_rot_matrix(
                        torch.cat([x, y, z], 1).reshape(-1, 3, 3).permute(0, 2,
                        1).reshape(-1, 9))

    def _unbuild_sparse_orientation_180(self, axis: torch.Tensor) -> \
        torch.Tensor:
        '''Recovers the orientation data for symmetries of 180.

        Arguments:
            axis: The axes to recover from their sparse representations.

        Returns:
            axis: the recovered axes.
        '''
        params = self._sparse_params['orientation']
        centers = axis.argmax(-1)
        neighbors = params['neighbors'][centers]
        axis = (axis[torch.arange(axis.shape[0]).unsqueeze(1), neighbors] -
            axis[torch.arange(axis.shape[0]).unsqueeze(1), params['opposites'][
            neighbors]]).reshape(-1, 1, 6)
        return F.normalize((axis @ params['vectors'][neighbors]).squeeze(1))


def _rot_quaternion_axis(q: torch.Tensor, axis: str, angle: float) -> \
    torch.Tensor:
    '''Rotates set of quaternions around given axis by given angle(s).

    r = p·q
    rw = pw·qw - px·qx - py·qy - pz·qz
    rx = pw·qx + px·qw + py·qz - pz·qy
    ry = pw·qy - px·qz + py·qw + pz·qx
    rz = pw·qz + px·qy - py·qx + pz·qw
    Source: https://en.wikipedia.org/wiki/Quaternion

    Arguments:
        q: The quaternions to be rotated.
        axis: The axis around which the quaternions are to be rotated.
        angle: The angle(s) by which the quaternions are to be rotated.

    Returns:
        q_rot: The resultant quaternions.
    '''
    signs = {'local_x': [-1, +1, +1, -1],
             'local_y': [-1, -1, +1, +1],
             'local_z': [-1, +1, -1, +1],
             'global_x': [-1, +1, -1, +1],
             'global_y': [-1, +1, +1, -1],
             'global_z': [-1, -1, +1, +1]}
    elems = {'local_x': [1, 0, 3, 2],
             'local_y': [2, 3, 0, 1],
             'local_z': [3, 2, 1, 0],
              'global_x': [1, 0, 3, 2],
             'global_y': [2, 3, 0, 1],
             'global_z': [3, 2, 1, 0]}
    if not torch.is_tensor(angle):
        angle = torch.tensor(angle)
    return torch.cos(angle / 2).view(-1, 1) * q + torch.tensor(signs[axis]) * \
        torch.sin(angle / 2).view(-1, 1) * q.index_select(1, torch.tensor(
        elems[axis]))

def _rot_vector_quaternion(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    '''Rotates vector according to given set of quaternions.

    q·v·q* = (qr^2 - |qi|^2)·v + 2·(qi·v)·qi + 2·qr·(qi×v)
    Source: https://graphics.stanford.edu/courses/cs348a-17-winter/Papers/quaternion.pdf

    Arguments:
        v: The vector to be rotated.
        q: The quaternions indicating the rotation.

    Returns:
        v_rot: The resultant vectors.
    '''
    return (q[:, 0] ** 2 - torch.sum(q[:, 1:] ** 2, 1)).unsqueeze(1) * v + 2 \
        * torch.sum(q[:, 1:] * v, 1).unsqueeze(1) * q[:, 1:] + 2 * q[:, 0:1] \
        * torch.linalg.cross(q[:, 1:], v.unsqueeze(0))

def _rot_vector_z(v: torch.Tensor, angle: float) -> torch.Tensor:
    '''Rotates set of vectors around global z axis by given angle(s).

    Arguments:
        v: The vectors to be rotated.
        angle: The angle(s) by which the vectors are to be rotated.

    Returns:
        v_rot: The resultant vectors.
    '''
    sin = torch.sin(angle)
    cos = torch.cos(angle)
    v_rot = torch.empty(v.shape)
    v_rot[:, 0] = cos * v[:, 0] - sin * v[:, 1]
    v_rot[:, 1] = sin * v[:, 0] + cos * v[:, 1]
    v_rot[:, 2] = v[:, 2]
    return v_rot

def _get_yaw_from_quaternion(q: torch.Tensor) -> torch.Tensor:
    '''Returns set of yaw angles of given quaternions.

    Source: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    Arguments:
        q: The quaternions from which to obtain the yaw angles.

    Returns:
        angle: The yaw angles.
    '''
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[
        :, 2] ** 2 + q[:, 3] ** 2))

def _get_pitch_from_quaternion(q: torch.Tensor) -> torch.Tensor:
    '''Returns set of pitch angles corresponding to given quaternions.

    Source: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    Arguments:
        q: The quaternions from which to obtain the pitch angles.

    Returns:
        angle: The pitch angles.
    '''
    return 2 * torch.atan2(torch.sqrt(1 + 2 * (q[:, 0] * q[:, 2] - q[:, 1] * q[
        :, 3])), torch.sqrt(1 - 2 * (q[:, 0] * q[:, 2] - q[:, 1] * q[:, 3]))) \
        - torch.pi / 2

def _get_quaternion_from_yaw_pitch(yaw: torch.Tensor, pitch: torch.Tensor) -> \
    torch.Tensor:
    '''Returns set of quaternions corresponding to given yaw and pitch angles.

    Source: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    Arguments:
        yaw: The yaw angles from which to obtain the quaternions.
        pitch: The pitch angles from which to obtain the quaternions.

    Returns:
        q: The quaternions.
    '''
    sy = torch.sin(yaw / 2)
    cy = torch.cos(yaw / 2)
    sp = torch.sin(pitch / 2)
    cp = torch.cos(pitch / 2)
    return torch.stack([cy * cp, - sy * sp, cy * sp, sy * cp]).T

def _get_quaternion_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    '''Returns set of quaternions corresponding to given yaw angles.

    Source: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    Arguments:
        yaw: The yaw angles from which to obtain the quaternions.

    Returns:
        q: The quaternions.
    '''
    sy = torch.sin(yaw / 2)
    cy = torch.cos(yaw / 2)
    return torch.stack([cy, torch.zeros_like(yaw), torch.zeros_like(yaw),
        sy]).T

def _get_rot_matrix_from_quaternion(q: torch.Tensor) -> torch.Tensor:
    '''Returns set of flat rot matrices corresponding to given quaternions.

    Source: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    Arguments:
        q: The quaternions from which to obtain the rotation matrices.

    Returns:
        rot_matrix: The flattened rotation matrices.
    '''
    q_sq = q.unsqueeze(2) @ q.unsqueeze(1)
    return torch.stack([1 - 2 * (q_sq[:, 2, 2] + q_sq[:, 3, 3]),
                        2 * (q_sq[:, 1, 2] - q_sq[:, 0, 3]),
                        2 * (q_sq[:, 1, 3] + q_sq[:, 0, 2]),
                        2 * (q_sq[:, 1, 2] + q_sq[:, 0, 3]),
                        1 - 2 * (q_sq[:, 1, 1] + q_sq[:, 3, 3]),
                        2 * (q_sq[:, 2, 3] - q_sq[:, 0, 1]),
                        2 * (q_sq[:, 1, 3] - q_sq[:, 0, 2]),
                        2 * (q_sq[:, 2, 3] + q_sq[:, 0, 1]),
                        1 - 2 * (q_sq[:, 1, 1] + q_sq[:, 2, 2])]).T

## unstable when trace ~ -1
#def _get_quaternion_from_rot_matrix(rot_matrix: torch.Tensor) -> torch.Tensor:
#    '''Returns set of quaternions corresponding to given flat rot matrices.
#
#    Source: https://en.wikipedia.org/wiki/Rotation_matrix

#    Arguments:
#        rot_matrix: The rotation matrices from which to obtain the quaternions.

#    Returns:
#        q: The quaternions.
#    '''
#    r = torch.sqrt(1 + rot_matrix[:, 0] + rot_matrix[:, 4] + rot_matrix[:, 8])
#    s = 1 / (2 * r)
#    return torch.stack([r / 2,
#                        (rot_matrix[:, 7] - rot_matrix[:, 5]) * s,
#                        (rot_matrix[:, 2] - rot_matrix[:, 6]) * s,
#                        (rot_matrix[:, 3] - rot_matrix[:, 1]) * s]).T

def _get_quaternion_from_rot_matrix(rot_matrix: torch.Tensor) -> torch.Tensor:
    '''Returns set of quaternions corresponding to given flat rot matrices.


    if m00 > 0:        # |w,x| > |y,z|
        if m11 > -m22: # |w| > |x|
            t = 1 + m00 + m11 + m22
            q = [t, m21-m12, m02-m20, m10-m01]
        else:
            t = 1 + m00 - m11 - m22
            q = [m21-m12, t, m10+m01, m02+m20]
    else:
        if m11 > m22:  # |y| > |z|
            t = 1 - m00 + m11 - m22
            q = [m02-m20, m10+m01, t, m21+m12]
        else:
            t = 1 - m00 - m11 + m22
            q = [m10-m01, m02+m20, m21+m11, t]
    q *= 0.5 / sqrt(t)
    Source: https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf

    Arguments:
        rot_matrix: The rotation matrices from which to obtain the quaternions.

    Returns:
        q: The quaternions.
    '''
    wx = (rot_matrix[:, 0] > 0).to(torch.int64) * 2 - 1
    wy = (rot_matrix[:, 4] > -wx * rot_matrix[:, 8]).to(torch.int64) * 2 - 1
    t = 1 + wx * rot_matrix[:, 0] + wy * rot_matrix[:, 4] + wx * wy * \
        rot_matrix[:, 8]
    q = torch.stack([t,
                     rot_matrix[:, 7] - wx * rot_matrix[:, 5],
                     rot_matrix[:, 2] -wy * rot_matrix[:, 6],
                     rot_matrix[:, 3] -wx * wy * rot_matrix[:, 1]]).T
    wx, wy = wx + 1, (wy + 1) // 2
    q = q.take_along_dim(torch.stack([3 - wx - wy, 2 - wx + wy, 1 + wx - wy, wx
        + wy]).T, 1)
    q /= 2 * torch.sqrt(t).unsqueeze(1)
    return q

def _unclamp_rot_matrix(rot_matrix: torch.Tensor) -> torch.Tensor:
    '''Obtains max row absolute values from rest of matrix row values.

    Arguments:
        rot_matrix: The rotation matrices to be unclamped.

    Returns:
        rot_matrix: The unclamped rotation matrices.
    '''
    rot = rot_matrix.reshape([-1, 3])
    idcs = torch.argmax(torch.abs(rot), 1)
    signs = torch.sgn(rot[torch.arange(rot.shape[0]), idcs])
    rot[torch.arange(rot.shape[0]), idcs] = 0
    rot[torch.arange(rot.shape[0]), idcs] = signs * torch.sqrt(1 - torch.sum(
        rot ** 2, 1))
    return rot.reshape([-1, 9])

def _unclamp_position(pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    '''Recovers clamped values by assuming far objects are static.

    Arguments:
        pos: The positions to be unclamped.
        mask: The elements to be unclamped.

    Returns:
        pos: The unclamped positions.
    '''
    pos = pos.clone()
    for cur, prev, unclamp in zip(pos[1:], pos[:-1], mask[1:]):
        cur[unclamp] = prev[unclamp]
    for cur, next, unclamp in reversed(list(zip(pos[:-1], pos[1:], mask[:
        -1]))):
        cur[unclamp] = next[unclamp]
    return pos


def main(dataset_name: str = 'KitInput', mode: str = 'orig') -> None:
    '''Visualizes sequential samples of the dataset.

    Arguments:
        dataset_name: The name of the PyTorch Dataset.
        mode: What data is visualized: the original trials ('orig'), the
            original trials transformed to ego perspective ('ego'), the
            original trials with the rotations converted to rotation matrices
            ('mat'), the original trials with ego perspective and rotation
            matrices ('ego_mat'), or the recovered data from the processed
            output samples from the dataset ('out').
    '''
    if mode == 'orig':
        orig_trials, ego_perspective, rot_matrices = True, False, False
    elif mode == 'ego':
        orig_trials, ego_perspective, rot_matrices = True, True, False
    elif mode == 'mat':
        orig_trials, ego_perspective, rot_matrices = True, False, True
    elif mode == 'ego_mat':
        orig_trials, ego_perspective, rot_matrices = True, True, True
    elif mode == 'out':
        orig_trials, ego_perspective, rot_matrices = False, False, False
    else:
        raise ValueError(f'{mode} is not a valid mode')

    params = {'splitprops': [0.6, 0.2, 0.2], 'splitset': 0}
    if orig_trials:
        params['keep_orig_trials'] = True
    dataset = getattr(sys.modules[__name__], dataset_name)(params)

    if orig_trials:
        if ego_perspective:
            for trial in dataset._orig_trials:
                x = trial['x']
                for part in x.values():
                    for coords in part:
                        coords['main_com'] = coords['position']
                KitInput._set_ego_perspective(x)
                l = x['var'][0]['main_com'].shape[0]
                for coords in x['fix']:
                    if coords['name'] == 'torso':
                        coords['main_com'] = torch.zeros([l, 3])
                        coords['orientation'] = torch.tensor([[1, 0, 0,
                            0]]).expand([l, 4])
                for part in x.values():
                    for coords in part:
                        coords['position'] = coords['main_com']

        if rot_matrices:
            for trial in dataset._orig_trials:
                x = trial['x']
                for coords in x['fix']:
                    if coords['name'] == 'head':
                        coords['orientation'] = _rot_quaternion_axis(coords[
                            'orientation'], 'local_z', torch.pi / 2)
                        break
                KitInput._convert_orientations(x)
                for coords in x['fix']:
                    if coords['name'] == 'torso':
                        coords['orientation'] = \
                            _get_rot_matrix_from_quaternion(coords[
                            'orientation'])
                    elif coords['name'] == 'head':
                        cy = coords['orientation'][:, 0]
                        sy = -coords['orientation'][:, 1]
                        sp = -coords['orientation'][:, 2]
                        cp = coords['orientation'][:, 3]
                        # Source: https://en.wikipedia.org/wiki/Rotation_matrix
                        coords['orientation'] = torch.stack([cy, -sy * cp, sy *
                            sp, sy, cy * cp, -cy * sp, torch.zeros_like(sy),
                            sp, cp], 1)

    else:
        dataset.classif = slice(None, None)
        dataloader = DataLoader(dataset, batch_size=128)
        #dataset.set_batch_iter(128)
        #dataloader = DataLoader(dataset, batch_size=None)

    def update_fig(fig, axss, x, im_sizes):
        n_rows = math.ceil(math.sqrt(len(x)))
        n_obj_axs = len(axss)
        for i, axs in enumerate(axss):
            for ax in axs:
                ax.remove()
        axss = [[fig.add_subplot(n_rows, n_rows * n_obj_axs, j * n_obj_axs + i
            + 1, xticks=[], yticks=[]) for j in range(len(x))] for i in range(
            n_obj_axs)]
        for ax, coords in zip(axss[0], x):
            ax.set_title(coords['name'])
        ims = [init_axs_ims(axs, [size] * len(axs)) for axs, size in zip(
            axss, im_sizes)]
        return axss, ims

    def init_axs_ims(axs, sizes):
        return [ax.imshow(torch.rand(size), cmap='gray') for ax, size in zip(
            axs, sizes)]

    def get_dodecahedron_vis(values):
        vis = torch.zeros(10, 6)
        vis[4:6, 0:2] = values[0]
        vis[0:2, 0:2] = values[1]
        vis[4:6, 4:6] = values[2]
        vis[0:2, 4:6] = values[3]
        vis[8:10, 2:4] = values[4]
        vis[3:5, 2:4] = values[5]
        vis[6:8, 2:4] = values[6]
        vis[1:3, 2:4] = values[7]
        vis[7:9, 0:2] = values[8]
        vis[7:9, 4:6] = values[9]
        vis[2:4, 0:2] = values[10]
        vis[2:4, 4:6] = values[11]
        return vis

    datavis = kitdatavis.KitInputVis(dataset, ego_perspective, rot_matrices)
    datavis.show()
    if isinstance(dataset, KitSparseInput) and not orig_trials:
        torso_com_grid_sizes = dataset._sparse_params['torso']['com'][
            'grid_sizes']
        torso_yaw_grid_sizes = dataset._sparse_params['torso']['yaw'][
            'grid_sizes']
        torso_im_sizes = [torso_com_grid_sizes[0:2], [torso_com_grid_sizes[2],
            1], [1, torso_yaw_grid_sizes[0]]]
        pos_grid_sizes = dataset._sparse_params['position']['grid_sizes']
        pos_im_sizes = [pos_grid_sizes[0:2], [pos_grid_sizes[2], 1]]
        dodec_im_size = get_dodecahedron_vis([0] * 12).shape
        rot_im_sizes = [dodec_im_size, dodec_im_size]
        torso_fig, torso_axs = plt.subplots(1, 3)
        torso_fig.suptitle('Torso')
        torso_axs[0].set_title('diff CoM')
        torso_axs[2].set_title('diff yaw')
        torso_ims = init_axs_ims(torso_axs, torso_im_sizes)
        pos_fig = plt.figure()
        pos_fig.suptitle('Positions')
        pos_axs = [[], []]
        rot_fig = plt.figure()
        rot_fig.suptitle('Orientations')
        rot_axs = [[], []]

    while True:
        if orig_trials:
            trial = random.choice(dataset._orig_trials)
        else:
            xfs, xvs, ys = [], [], []
            for xf, xv, y in dataloader:
                xfs.append(xf)
                xvs.append(xv)
                ys.append(y)
            trial = dataset.recover_orig_data(torch.cat(xfs, 0), torch.cat(xvs,
                0), torch.cat(ys, 0))
            if isinstance(dataset, KitSparseInput):
                sparse_trial = dataset._unarrange_data(torch.cat(xfs, 0),
                    torch.cat(xvs, 0), torch.cat(ys, 0), {})
                for i, torso_coords in enumerate(sparse_trial['x']['fix']):
                    if torso_coords['name'] == 'torso':
                        break
                del sparse_trial['x']['fix'][i]
                sparse_trial['x'] = sparse_trial['x']['fix'] + sparse_trial[
                    'x']['var']
                pos_axs, pos_ims = update_fig(pos_fig, pos_axs, sparse_trial[
                    'x'], pos_im_sizes)
                rot_axs, rot_ims = update_fig(rot_fig, rot_axs, sparse_trial[
                    'x'], rot_im_sizes)
                for im, coords in zip(rot_ims[1], sparse_trial['x']):
                    if coords['name'] == 'head':
                        im.remove()
                        break
        datavis.init_trial(trial)
        for i in range(0, len(trial['x']['fix'][0]['position']), 5):
            datavis.update(i)
            if isinstance(dataset, KitSparseInput) and not orig_trials:
                torso_ims[0].set(data=torso_coords['diff_com'][i,
                    :torso_im_sizes[0][0] * torso_im_sizes[0][1]].reshape(
                    torso_im_sizes[0]))
                torso_ims[1].set(data=torso_coords['diff_com'][i:i + 1,
                    torso_im_sizes[0][0] * torso_im_sizes[0][1]:].T.flip(0))
                torso_ims[2].set(data=torso_coords['diff_yaw'][i:i+1])
                for j, coords in enumerate(sparse_trial['x']):
                    pos_ims[0][j].set(data=coords['main_com'][i, :pos_im_sizes[
                        0][0] * pos_im_sizes[0][1]].reshape(pos_im_sizes[0]))
                    pos_ims[1][j].set(data=coords['main_com'][i:i + 1,
                        pos_im_sizes[0][0] * pos_im_sizes[0][1]:].T.flip(0))
                    rot_ims[0][j].set(data=get_dodecahedron_vis(coords[
                        'handle_axis'][i]))
                    if coords['name'] != 'head':
                        rot_ims[1][j].set(data=get_dodecahedron_vis(coords[
                            'top_axis'][i]))
                plt.show(block=False)
                plt.pause(0.033)
            #time.sleep(.01)
        

if __name__ == '__main__':
    import sys
    from torch.utils.data import DataLoader
    import kitdatavis
    import matplotlib.pyplot as plt

    main(*sys.argv[1:])
