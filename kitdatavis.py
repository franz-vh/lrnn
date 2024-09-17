#!/usr/bin/env python3

'''Functionality to visualize the KIT Whole-Body Human Motion Database.'''

import kitdata
import torch
import math
import pyvista as pv
import os
from typing import Any, List, Dict


class KitInputVis:
    '''Visualization functionality for the KIT dataset.

    Attributes:
        dataset: A dataset with the KIT data.
        ego_perspective: A boolean indicating whether the camera position
            should be set for data in ego perspective or not.
        rot_matrices: A boolean indicating whether the rotations are given as
            rotation matrices or quaternions.
        def_color: A list of floats with the default colors of the plotted
            meshes.
        _datasets_dir: A string with the directory containing the datasets.
        _action_lbls: A list of strings with the action classification class
            names.
        _plan_lbls: A list of strings with the plan classification class names.
        _subaction_lbls: A list of strings with the subaction classification
            class names.
        _object_lbls: A list of strings with the object classification class
            names.
        _get_rot_vector_angle: A function that returns a rotation axis vector
            and the rotation angle corresponding to a quaternion or rotation
            matrix.
        _meshes: A list with the used object meshes.
        _pl: A plot with the data visualization.
        _plot_texts: A text actor with the plot texts.
        _trial_meshes: A list with the object meshes currently in the plot.
        _trial_actors: A list with the object mesh actors currently in the
            plot.
        _trial: A dictionary with the samples and labels of the current trial.
        _i: An integer with the index of the next sample and label in the
            current trial.
    '''
    def __init__(self, dataset: torch.utils.data.Dataset, ego_perspective: bool
         = False, rot_matrices: bool = False) -> None:
        '''Inits KitInputVis.

        Arguments:
            dataset: The KIT data.
            ego_perspective: If true, the camera position is set for data in
                ego perspective.
            rot_matrices: If true, the rotations are given as rotation
                matrices.
        '''
        self.dataset = dataset
        self.ego_perspective = ego_perspective
        self.rot_matrices = rot_matrices
        self.def_color = [pv.global_theme.color._red / 255,
            pv.global_theme.color._green / 255, pv.global_theme.color._blue /
            255]
        self._datasets_dir = kitdata._datasets_dir
        self._action_lbls = dataset.all_lbls[dataset.classifs.index('actions')]
        self._plan_lbls = dataset.all_lbls[dataset.classifs.index('plans')]
        self._subaction_lbls = dataset.all_lbls[dataset.classifs.index(
            'leftsubactions')]
        self._object_lbls = dataset.all_lbls[dataset.classifs.index(
            'leftmainobjects')]
        if rot_matrices:
            # Source: https://en.wikipedia.org/wiki/Rotation_matrix
            self._get_rot_vector_angle = lambda rot: ([rot[7] - rot[5], rot[2]
                - rot[6], rot[3] - rot[1]], math.acos(((rot[0] + rot[4] + rot[
                8] - 1) / 2).clamp(-1, 1)))
        else:
            self._get_rot_vector_angle = lambda rot: (rot[1:], 2 * math.acos(
                rot[0].clamp(-1, 1)))
        self._load_body_meshes()
        self._pl = pv.Plotter()

    def _load_body_meshes(self) -> None:
        '''Loads the meshes corresponding to the body objects.'''
        meshes = {}
        meshes['right_hand'] = pv.read(os.path.join(self._datasets_dir, 'kit',
            'models', 'hand.stl'))
        meshes['right_hand'].scale(10, inplace=True)
        meshes['right_hand'].rotate_z(-90, inplace=True)
        meshes['right_hand'].rotate_x(-30, inplace=True)
        meshes['right_hand'].rotate_y(45, inplace=True)
        meshes['right_hand'].translate([50, 0, 0], inplace=True)
        meshes['left_hand'] = meshes['right_hand'].copy()
        meshes['left_hand'].flip_x(inplace=True)
        meshes['head'] = pv.read(os.path.join(self._datasets_dir, 'kit',
            'models', 'head.stl'))
        meshes['head'].scale(17, inplace=True)
        meshes['head'].translate([0, 120, 0], inplace=True)
        meshes['head'].rotate_x(-90, inplace=True)
        meshes['torso'] = pv.read(os.path.join(self._datasets_dir, 'kit',
            'models', 'torso.stl'))
        meshes['torso'].scale(9, inplace=True)
        meshes['torso'].rotate_x(-90, inplace=True)
        meshes['torso'].rotate_z(180, inplace=True)
        meshes['torso'].translate([0, 0, -300], inplace=True)
        self._meshes = meshes

    def show(self) -> None:
        '''Displays the plotting window.'''
        self._pl.show(interactive_update=True)

    def init_trial(self, trial: Dict[str, Dict[str, Any]]) -> None:
        '''Initializes the visualization corresponding to the given trial.

        Arguments:
            trial: The samples and labels of the trial to be visualized.
        '''
        self._trial = trial
        self._i = 0
        self._pl.clear_actors()
        if 'actions' in trial['trial_lbls']:
            lbl = self._action_lbls[trial['trial_lbls']['actions']]
        else:
            lbl = self._plan_lbls[trial['trial_lbls']['plans']]
        self._pl.add_title(lbl, font_size=10)
        self._plot_texts = self._pl.add_text('', font_size=10)
        trial_meshes, trial_actors = [], []
        for coords in trial['x']['fix'] + trial['x']['var']:
            if coords['name'] not in self._meshes:
                self._load_obj_mesh(coords['name'])
            trial_meshes.append(self._meshes[coords['name']].copy())
        for mesh in trial_meshes:
            trial_actors.append(self._pl.add_mesh(mesh))
        self._trial_meshes, self._trial_actors = trial_meshes, trial_actors
        if self.ego_perspective:
            self._pl.camera_position = [(0, -2000, 3000), (0, 400, -500), (0,
                0, 1)]
        else:
            self._pl.camera_position = [(-3000, -2000, 3200), (500, 400, 1200),
                (0, 0, 1)]

    def _load_obj_mesh(self, obj_name: str) -> None:
        '''Loads the mesh corresponding to the indicated object.

        Arguments:
            obj_name: The name of the object whose mesh needs to be loaded.
        '''
        mesh = pv.read(os.path.join(self._datasets_dir, 'kit', 'models',
            obj_name + '.stl'))
        mesh.scale(1000, inplace=True)
        if obj_name in ('cup_small', 'eggplant_attachment', 'frying_pan',
            'oil_bottle', 'spatula'):
            mesh.rotate_x(-90, inplace=True)
        self._meshes[obj_name] = mesh

    def update(self, i: int = -1, colors: List[float] = None) -> None:
        '''Updates the window with the next or indicated sample.

        Arguments:
            i: The index of the sample to be visualized.
            colors: The colors of the meshes to be visualized.
        '''
        if i < 0:
            i = self._i
        if colors is None:
            colors = [self.def_color] * len(self._trial_meshes)
        if 'rightsubactions' in self._trial['lbls']:
            action_lbl = self._trial['lbls']['rightsubactions'][i]
            if action_lbl < 0:
                text = 'unknown action'
            else:
                text = self._subaction_lbls[action_lbl]
                if 'rightmainobjects' in self._trial['lbls']:
                    text += ' '
                    text += self._object_lbls[self._trial['lbls'][
                        'rightmainobjects'][i]]
        else:
            text = 'unknown action'
        self._plot_texts.SetText(2,
            f'\n\n\n\n                                        {text}')
        if 'leftsubactions' in self._trial['lbls']:
            action_lbl = self._trial['lbls']['leftsubactions'][i]
            if action_lbl < 0:
                text = 'unknown action'
            else:
                text = self._subaction_lbls[action_lbl]
                if 'leftmainobjects' in self._trial['lbls']:
                    text += ' '
                    text +=  self._object_lbls[self._trial['lbls'][
                        'leftmainobjects'][i]]
        else:
            text = 'unknown action'
        self._plot_texts.SetText(3,
            f'\n\n\n\n{text}                                        ')
        for coords, mesh, actor, color in zip(self._trial['x']['fix'] +
            self._trial['x']['var'], self._trial_meshes, self._trial_actors,
                colors):
                mesh.points = self._meshes[coords['name']].points.copy()
                vector, angle = self._get_rot_vector_angle(coords[
                    'orientation'][i])
                mesh.rotate_vector(vector=vector.numpy(), angle=angle * 180 /
                    math.pi, inplace=True)
                mesh.translate(coords['position'][i], inplace=True)
                actor.prop.color = color
        self._pl.update()
        self._i = i + 1
