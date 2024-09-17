#!/usr/bin/env python3

'''Functionality to test an action selection system in simulation.

The simulation class extracts the velocity and height of the hand from a sample
(possibly generated/predicted by a neural network), updates the state of the
environment, generates the actual new sample, and checks whether the goals are
satisfied.
On the original data, white noise is added to the data, followed by a low-pass
filter, before extracting the sparse representations. This code skips that
processing to keep it simpler (the noise can be seen as a mechanism to make the
system more robust during training).
For the goal achievement detection, since the final object positions are
affected by gaussian noise, the system considers a goal is achieved when all
objects are under a distance of 3 sigma from their ideal position without noise
(3*sqrt(2) sigma for relative positions, as the variance of a difference of
gaussians is the sum of the variances).
The fact that the synthetic action environment was only partially designed
thinking of using it for simulation leads to some minor issues when trying to
detect the beginning and end and the target object of the "move" primitives of
the different actions. The start of the picknplace "move" is quite
straightforward to detect because, in the original data, the hand fully opens
and then fully closes around the center of the object. The start of the push
and of the pull "move" is more problematic, because, in the original data, the
hand simply takes a position behind or in front of the object, respectively,
and then moves it, but this positioning can also happen by chance. The code
here relies on the position of the hand wrt the object and on the fact that the
hand is stopping and changing velocity. The detection of the end of the "move"
works better by just relying on the change of velocity of the hand (and on the
opening of the hand for picknplace, even though this alone is not enough, as,
in the original data, how much the hand is open at the end is decided randomly
and can be very little). In addition, the full openning of the hand has been
added as an end condition for all actions, for the cases when the hand is
pushing or pulling an object by mistake and the next action is picknplace. Some
common errors are:
- After having pushed or pulled an object, if the next object is in the same
direction, since the hand is stopped and located just behind or in front of the
last object, it will start pushing or pulling it.
- After having pushed or pulled an object, if the hand tries to picknplace the
same object, for the same reason as before, it will push it, move it and fail
to pick it.
- Due to the fact that white noise is added to the ideal position to manipulate
an object, the system only considers a target object if the hand is at a
distance of less than 3 sigma from that ideal position. This way, when the
noise is large enough, the object is not moved.
- When there are several objects close to each other, and for the reason
mentioned just before, sometimes the system detects the wrong action over the
wrong object.
To deal with these issues, the id, position, and velocity information of the
object of attention included in the sample could also be considered, but this
would bring other issues such as how to deal with situations where multiple
objects are being paid attention to. Similarly, considering the environment
information with the position of the 4 objects would also bring similar issues.
In any case, this is probably something to test in future work.
'''


import inputdata
import lrnn
import torch
import numpy as np
import random
import math
from typing import Any, Tuple, Dict, Callable


class SynthPlanInput(inputdata.SynthPlanInput):
    '''An extension of SynthPlanInput that returns the plans label as input.'''
    def __init__(self, params: Dict[str, Any] = {}, matlab_batch_size: int =
        16384) -> None:
        '''Inits SynthPlanInput.

        Arguments:
            params: The parameters for the MATLAB class constructor (they are
                ignored for this class).
            matlab_batch_size: The number of samples retrieved from the MATLAB
                object each time new samples are needed.
        '''
        super(SynthPlanInput, self).__init__(params, matlab_batch_size)
        self.classif = self.classifs.index('plans')

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''Returns the next sample and label.

        Returns:
            sample: The next sample.
            label: The label associated to sample.
        '''
        sample, label = super().__next__()
        return sample, torch.nn.functional.one_hot(label, len(self.all_lbls[
            self.classif])), torch.tensor([], dtype=torch.int64)

    def set_classif(self, classif: 'str') -> None:
        '''Overloads the parent function to avoid changing the classification.

        Arguments:
            classif: What would be the current classification name (which would
                be used in method __next__).
        '''
        raise RuntimeError('This dataset does not allow changing the current '
                           'classification')


class SynthActionsEnvironment:
    '''A class to simulate the environment of SynthActions.

    Attributes:
        dataset: A dataset with the synthetic actions input data whose
            environment is to be simulated.
        data: A dictionary of strings to anything with the environment
            constants.
        vis_data: A dictionary of strings to anything with the environment
            visualization constants.
        state: A dictionary of strings to anything with the state of the
            environment.
    '''
    def __init__(self, dataset: inputdata.SynthActionsWrapper) -> None:
        '''Inits SynthActionsEnvironment.'''
        self.dataset = dataset
        self._init_data()
        self.generate_scenario()

    def _init_data(self):
        '''Inits the environment constants.'''
        data = inputdata._matlab_eng.eval(self.dataset.matlab_name + '.Data')
        data['n_objtypes'] = int(data['n_objtypes'])
        data['in_p']['objid_sz'] = int(data['in_p']['objid_sz'])
        data['in_p']['opcl_sz'] = int(data['in_p']['opcl_sz'])
        data['in_p']['v_sd'] = int(data['in_p']['v_sd'])
        data['in_p']['pos_sd'] = int(data['in_p']['pos_sd'])
        data['in_p']['v_nthfactor'] = torch.tensor(data['in_p']['v_nthfactor'])
        data['in_p']['pos_nthfactor'] = torch.tensor(data['in_p'][
            'pos_nthfactor'])
        data['in_p']['env_h'] = int(data['in_p']['env_h'])
        data['in_p']['env_w'] = int(data['in_p']['env_w'])
        data['objid_actv'] = int(data['objid_actv'])
        data['n_objects'] = int(data['n_objects'])
        data['dims'] = {name: int(dim) for name, dim in data['dims'].items()}
        data['traj_p']['plans_scen'] = int(data['traj_p']['plans_scen'])
        data['in_g'] = {name: torch.from_numpy(np.array(value, dtype=
            np.float32)) for name, value in data['in_g'].items()}
        data['in_g']['opcl_ref'] = data['in_g']['opcl_ref'].flatten()
        data['in_g']['Xv'] = data['in_g']['Xv'].reshape(data['in_g'][
            'Xv'].numel(), 1)
        data['in_g']['Yv'] = data['in_g']['Yv'].reshape(data['in_g'][
            'Yv'].numel(), 1)
        data['in_g']['Xv_hand'] = data['in_g']['Xv_hand'].squeeze(1)
        data['in_g']['Yv_hand'] = data['in_g']['Yv_hand'].squeeze(1)
        data['in_g']['Xpos'] = data['in_g']['Xpos'].reshape(data['in_g'][
            'Xpos'].numel(), 1)
        data['in_g']['Ypos'] = data['in_g']['Ypos'].reshape(data['in_g'][
            'Ypos'].numel(), 1)
        if 'Xenv' in data['in_g']:
            data['in_g']['Xenv'] = data['in_g']['Xenv'].reshape(data['in_g'][
                'Xenv'].numel(), 1)
            data['in_g']['Yenv'] = data['in_g']['Yenv'].reshape(data['in_g'][
                'Yenv'].numel(), 1)
        data['objtype_ids'] = torch.from_numpy(np.array(data['objtype_ids'],
            dtype=np.float32))
        data['affordances'] = torch.from_numpy(np.array(data['affordances'],
            dtype=np.bool_))
        goal_names = ['horizontal', 'vertical', 'square', 'triangle']
        n_objects = data['n_objects']
        scen_w, scen_h = data['dims']['scen_w'], data['dims']['scen_h']
        horizontal_x = torch.linspace(1 / (2 * n_objects), 1 - 1 / (2 *
            n_objects), n_objects) * scen_w
        vertical_y = torch.linspace(1 / (2 * n_objects), 1 - 1 / (2 *
            n_objects), n_objects) * scen_h
        sd = (scen_w + scen_h) / 2
        square_r = sd / math.sqrt(8)
        triangle_d = math.sqrt(3) * sd / 4
        goals = {'names': goal_names, 'horizontal_x': horizontal_x,
            'vertical_y': vertical_y, 'square_r': square_r, 'triangle_d':
            triangle_d}
        data['goals'] = goals
        self.data = data
        vis_data = inputdata._matlab_eng.eval(self.dataset.matlab_name +
            '.VisData')
        vis_data['hand_w'] = int(vis_data['hand_w'])
        vis_data['margin'] = int(vis_data['margin'])
        vis_data['colors'] = torch.from_numpy(np.array(vis_data['colors'],
            dtype=np.float32))
        vis_data['color_objs'] = torch.from_numpy(np.array(vis_data[
            'color_objs'], dtype=np.float32)).permute(3,0,1,2)
        vis_data['Xscenario'] = torch.from_numpy(np.array(vis_data[
            'Xscenario'], dtype=np.float32))
        vis_data['Yscenario'] = torch.from_numpy(np.array(vis_data[
            'Yscenario'], dtype=np.float32))
        self.vis_data = vis_data

    def generate_scenario(self):
        '''Generates a new scenario.'''
        objtypes = torch.randint(0, self.data['n_objtypes'], (self.data[
            'n_objects'],))
        x_hand = torch.rand([]) * self.data['dims']['scen_w']
        y_hand = torch.rand([]) * self.data['dims']['scen_h']
        hand_h = self.data['dims']['clhand_h'] + torch.rand([]) * (self.data[
            'dims']['ophand_h'] - self.data['dims']['clhand_h'])
        x = torch.rand(self.data['n_objects']) * self.data['dims']['scen_w']
        y = torch.rand(self.data['n_objects']) * self.data['dims']['scen_h']
        vx = torch.zeros(self.data['n_objects'])
        vy = torch.zeros(self.data['n_objects'])
        self.state = {'objtypes': objtypes, 'x_hand': x_hand, 'y_hand': y_hand,
            'hand_h': hand_h, 'x': x, 'y': y, 'vx': vx, 'vy': vy}
        action = None
        hand_hs = hand_h * torch.ones(round(self.data['traj_p']['tmax_cl'] *
            self.data['traj_p']['fs']))
        vs = torch.zeros(2)
        vnx, vny = 0, 0
        self._state = {'action': action, 'hand_hs': hand_hs, 'vs': vs, 'vnx':
            vnx, 'vny': vny}

    def move_hand(self, vx: torch.Tensor, vy: torch.Tensor, hand_h:
        torch.Tensor) -> Tuple[torch.Tensor, str]:
        '''Updates the environment according to the given hand movement.

        Arguments:
            vx: The x component of the hand velocity.
            vy: The y component of the hand velocity.
            hand_h: The height of the hand (how open it is).

        Returns:
            sample: The resultant sample.
            goal: The name of the achieved goal.
        '''
        v = (vx ** 2 + vy ** 2).sqrt() + 1e-5
        vnx, vny = vx / v, vy / v
        goal = None
        if self._state['action'] is not None:
            goal = self.check_end_move(v, vnx, vny, hand_h)
        if self._state['action'] is None:
            self.check_start_move(v, vnx, vny, hand_h)
        self._state['hand_hs'] = self._state['hand_hs'].roll(-1)
        self._state['hand_hs'][-1] = hand_h
        self._state['vs'] = self._state['vs'].roll(-1)
        self._state['vs'][-1] = v
        self._state['vnx'], self._state['vny'] = vnx, vny
        self.state['x_hand'] = (self.state['x_hand'] + vx / self.data[
            'traj_p']['fs']).clamp(0, self.data['dims']['scen_w'])
        self.state['y_hand'] = (self.state['y_hand'] + vy / self.data[
            'traj_p']['fs']).clamp(0, self.data['dims']['scen_h'])
        self.state['hand_h'] = hand_h
        self.state['vx'] = torch.zeros(self.data['n_objects'])
        self.state['vy'] = torch.zeros(self.data['n_objects'])
        if self._state['action'] is not None:
            x_obj = (self.state['x'][self._state['tgt_obj']] + vx / self.data[
                'traj_p']['fs']).clamp(0, self.data['dims']['scen_w'])
            y_obj = (self.state['y'][self._state['tgt_obj']] + vy / self.data[
                'traj_p']['fs']).clamp(0, self.data['dims']['scen_h'])
            self.state['vx'][self._state['tgt_obj']] = (x_obj - self.state[
                'x'][self._state['tgt_obj']]) * self.data['traj_p']['fs']
            self.state['vy'][self._state['tgt_obj']] = (y_obj - self.state[
                'y'][self._state['tgt_obj']]) * self.data['traj_p']['fs']
            self.state['x'][self._state['tgt_obj']] = x_obj
            self.state['y'][self._state['tgt_obj']] = y_obj
        return self.get_sample(v, vx, vy), goal

    def check_start_move(self, v: torch.Tensor, vnx: torch.Tensor, vny:
        torch.Tensor, hand_h: torch.Tensor) -> None:
        '''Checks whether the start of a move primitive is detected.

        Arguments:
            v: The magnitude of the hand velocity.
            vnx: The x component of the normalized hand velocity.
            vny: The y component of the normalized hand velocity.
            hand_h: The height of the hand (how open it is).
        '''
        affordances = self.data['affordances'][self.state['objtypes']]
        if hand_h < self.data['dims']['clhand_h'] + .2 * (self.data['dims'][
            'ophand_h'] - self.data['dims']['clhand_h']) and self._state[
            'hand_hs'].max() > self.data['dims']['clhand_h'] + .8 * (self.data[
            'dims']['ophand_h'] - self.data['dims']['clhand_h']):
            aff = affordances[:, 0]
            if any(aff):
                dist2, tgt_obj = ((self.state['x'][aff] - self.state['x_hand'])
                    ** 2 + (self.state['y'][aff] - self.state['y_hand']) **
                    2).min(0)
                if dist2 < (3 * self.data['traj_p']['hand_sigma']) ** 2:
                    self._state['action'] = 'picknplace'
                    self._state['tgt_obj'] = [i for i, af in enumerate(aff) if
                        af][tgt_obj]
                    return
        if v > self._state['vs'][1] and self._state['vs'][1] <= self._state[
            'vs'][0] and v < .1 * self.data['traj_p']['vxymax']:
            shift_x = self.data['dims']['obj_w'] * vnx
            shift_y = self.data['dims']['obj_h'] * vny
            aff = affordances[:, 1]
            if any(aff):
                dist2, tgt_obj = ((self.state['x'][aff] - shift_x - self.state[
                    'x_hand']) ** 2 + (self.state['y'][aff] - shift_y -
                    self.state['y_hand']) ** 2).min(0)
                if dist2 < (3 * self.data['traj_p']['hand_sigma']) ** 2:
                    self._state['action'] = 'push'
                    self._state['tgt_obj'] = [i for i, af in enumerate(aff) if
                        af][tgt_obj]
                    return
            aff = affordances[:, 2]
            if any(aff):
                dist2, tgt_obj = ((self.state['x'][aff] + shift_x - self.state[
                    'x_hand']) ** 2 + (self.state['y'][aff] + shift_y -
                    self.state['y_hand']) ** 2).min(0)
                if dist2 < (3 * self.data['traj_p']['hand_sigma']) ** 2:
                    self._state['action'] = 'pull'
                    self._state['tgt_obj'] = [i for i, af in enumerate(aff) if
                        af][tgt_obj]
                    return

    def check_end_move(self, v: torch.Tensor, vnx: torch.Tensor, vny:
        torch.Tensor, hand_h: torch.Tensor) -> str:
        '''Checks whether the end of a move primitive is detected.

        Arguments:
            v: The magnitude of the hand velocity.
            vnx: The x component of the normalized hand velocity.
            vny: The y component of the normalized hand velocity.
            hand_h: The height of the hand (how open it is).

        Returns:
            goal: The name of the achieved goal.
        '''
        if (v > self._state['vs'][1] and self._state['vs'][1] <= self._state[
            'vs'][0] and v < .1 * self.data['traj_p']['vxymax']) or (vnx *
            self._state['vnx'] + vny * self._state['vny'] < torch.tensor(
            torch.pi / 4).cos()) or (hand_h > self._state['hand_hs'].max() and
            hand_h > self.data['dims']['clhand_h'] + .8 * (self.data['dims'][
            'ophand_h'] - self.data['dims']['clhand_h'])) or (self._state[
            'action'] == 'picknplace' and hand_h > self.data['dims'][
            'clhand_h'] + .2 * (self.data['dims']['ophand_h'] - self.data[
            'dims']['clhand_h'])):
            self._state['action'] = None
            for goal in self.data['goals']['names']:
                if self.check_goal(goal):
                    return goal

    def get_sample(self, v: torch.Tensor, vx: torch.Tensor, vy: torch.Tensor) \
        -> torch.Tensor:
        '''Generates a dataset sample from the state and given hand velocity.

        Arguments:
            v: The magnitude of the hand velocity.
            vx: The x component of the hand velocity.
            vy: The y component of the hand velocity.

        Returns:
            sample: The resultant sample.
        '''
        att_p = self.data['att_p']
        in_g = self.data['in_g']
        in_p = self.data['in_p']
        vxymax = self.data['traj_p']['vxymax']
        x0 = self.state['x_hand'] + att_p['x0_factor'] * vx
        y0 = self.state['y_hand'] + att_p['y0_factor'] * vy
        sigma_x = att_p['sigma_X_min'] + att_p['sigma_X_factor']* v
        sigma_y = att_p['sigma_Y_min'] + att_p['sigma_Y_factor'] * v
        theta = torch.atan2(vy, -vx)
        A = att_p['A_factor'] * (att_p['sigma_X_min'] * att_p['sigma_Y_min'] /
            (sigma_x * sigma_y)).sqrt()
        a = theta.cos() ** 2 / (2 * sigma_x ** 2) + theta.sin() ** 2 / (2 *
            sigma_y ** 2)
        b = -(2 * theta).sin()/ (4 * sigma_x ** 2) + (2 * theta).sin() / (4 *
            sigma_y ** 2)
        c = theta.sin() ** 2 / (2 * sigma_x ** 2) + theta.cos() ** 2 / (2 *
            sigma_y ** 2)
        x, y = self.state['x'], self.state['y']
        attention = A * torch.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y -
            y0) + c * (y - y0) ** 2))
        attention = attention + att_p['v_factor'] * (self.state['vx'] ** 2 +
            self.state['vy'] ** 2).sqrt() / vxymax
        att_sum = attention.sum()
        if att_sum > 1:
            attention /= att_sum
        att_object_id = attention @ self.data['objtype_ids'][self.state[
            'objtypes']]
        open_close = (1 - (self.state['hand_h'] - in_g['opcl_ref']).abs() / (
            in_g['opcl_ref'][1] - in_g['opcl_ref'][0])).clamp(0)
        att_object_v = torch.exp(-1 / (2 * in_p['v_var']) * ((in_g['Xv'] -
            self.ntanh(self.state['vx'] /vxymax, in_p['v_nthfactor'])) ** 2 + (
            in_g['Yv'] - self.ntanh(self.state['vy'] /vxymax, in_p[
            'v_nthfactor'])) ** 2)) @ attention
        hand_v = torch.exp(-1 / (2 * in_p['v_var']) * ((in_g['Xv_hand'] -
            self.ntanh(vx / vxymax, in_p['v_nthfactor'])) ** 2 + (in_g[
            'Yv_hand'] - self.ntanh(vy / vxymax, in_p['v_nthfactor'])) ** 2))
        att_object_hand_pos = torch.exp(-1 / (2 * in_p['pos_var']) * ((in_g[
            'Xpos'] - self.ntanh((self.state['x'] - self.state['x_hand']) /
            self.data['dims']['scen_w'], in_p['pos_nthfactor'])) ** 2 + (
            in_g['Ypos'] - self.ntanh((self.state['y'] - self.state['y_hand'])
            / self.data['dims']['scen_h'], in_p['pos_nthfactor'])) ** 2)) @ \
            attention
        if 'Xenv' in in_g:
            env = torch.exp(-1 / (2 * in_p['obj_var']) * ((in_g['Xenv'] -
                self.state['x']) ** 2 + (in_g['Yenv'] - self.state['y']) **
                2)).sum(1).clamp(0, 1)
        else:
            env = torch.tensor([])
        return torch.cat([att_object_id, open_close, att_object_v, hand_v,
            att_object_hand_pos, env], 0)

    @staticmethod
    def ntanh(input: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        '''Maps non-linearly expanding small inputs and compressing large ones.

        Arguments:
            input: The input value to be mapped.
            factor: The factor defining the non-linear mapping.

        Returns:
            output: The mapped output value.
        '''
        return (factor * input).tanh() / factor.tanh()

    def get_movement_from_sample(self, sample: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        '''Extracts the hand movement command from the given sample.

        Arguments:
            sample: The sample with the desired hand movement.

        Returns:
            vx: The x component of the hand velocity.
            vy: The y component of the hand velocity.
            hand_h: The height of the hand (how open it is).
        '''
        in_g = self.data['in_g']
        in_p = self.data['in_p']
        vxymax = self.data['traj_p']['vxymax']
        i_0 = in_p['objid_sz']
        i_f = i_0 + in_p['opcl_sz']
        hand_h = ((1 - sample[i_0:i_f].clamp(0, 1)) * (in_g['opcl_ref'][1] -
            in_g['opcl_ref'][0]) * torch.tensor([1, -1]) + in_g[
            'opcl_ref']).mean()
        i_0 = i_f + in_p['v_sd'] ** 2
        i_f = i_0 + in_p['v_sd'] ** 2
        hand_v = sample[i_0:i_f].reshape(in_p['v_sd'], in_p['v_sd'])
        v = -1 * 2 * in_p['v_var'] * hand_v.log()
        combs = torch.combinations(torch.arange(in_p['v_sd'])).T
        Xv = self.data['in_g']['Xv'].reshape(in_p['v_sd'], in_p['v_sd'])[:, [
            0]]
        vxs = -(((v[combs[0], :] - v[combs[1], :]) - (Xv[combs[0], :] ** 2 -
            Xv[combs[1], :] ** 2)) / (2 * (Xv[combs[0], :] - Xv[combs[1],
            :])))
        w_vx = hand_v[combs[0], :] * hand_v[combs[1], :]
        vx = (vxs * w_vx).sum() / w_vx.sum()
        #vx = vxs.mean()
        vx = (vx * in_p['v_nthfactor'].tanh()).atanh() / in_p['v_nthfactor'] \
            * vxymax
        Yv = self.data['in_g']['Yv'].reshape(in_p['v_sd'], in_p['v_sd'])[[0],
            :]
        vys = -(((v[:, combs[0]] - v[:, combs[1]]) - (Yv[:, combs[0]] ** 2 -
            Yv[:, combs[1]] ** 2)) / (2 * (Yv[:, combs[0]] - Yv[:, combs[
            1]])))
        w_vy = hand_v[:, combs[0]] * hand_v[:, combs[1]]
        vy = (vys * w_vy).sum() / w_vy.sum()
        #vy = vys.mean()
        vy = (vy * in_p['v_nthfactor'].tanh()).atanh() / in_p['v_nthfactor'] \
            * vxymax
        return vx, vy, hand_h

    def check_goal(self, goal: str) -> bool:
        '''Checks whether the current state satisfies the given goal.

        Arguments:
            goal: The name of the goal to be checked.

        Returns:
            achieved: If true, the goal conditions are satisfied.
        '''
        tol = 3 * self.data['traj_p']['hand_sigma']
        difftol = math.sqrt(2) * tol
        if goal == 'horizontal':
            x, _ = self.state['x'].sort()
            ymin, ymax = self.state['y'].min(), self.state['y'].max()
            if ((x - self.data['goals']['horizontal_x']).abs() < tol).all() \
                and ymax - ymin < difftol:
                return True
        elif goal == 'vertical':
            y, _ = self.state['y'].sort()
            xmin, xmax = self.state['x'].min(), self.state['x'].max()
            if ((y - self.data['goals']['vertical_y']).abs() < tol).all() and \
                xmax - xmin < difftol:
                return True
        elif goal == 'square':
            scen_w = self.data['dims']['scen_w']
            scen_h = self.data['dims']['scen_h']
            sd = (scen_w + scen_h) / 2
            r = ((self.state['x'] - sd / 2) ** 2 + (self.state['y'] - sd / 2)
                 ** 2).sqrt()
            _, indices = torch.atan2(self.state['y'] - scen_h / 2, self.state[
                'x'] - scen_w / 2).sort()
            x, y = self.state['x'][indices], self.state['y'][indices]
            x, y = torch.cat([x, x[[0]]], 0), torch.cat([y, y[[0]]], 0)
            d = ((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2).sqrt()
            if ((r - self.data['goals']['square_r']).abs() < tol).all() and ((d
                - sd / 2).abs() < difftol).all():
                return True
        elif goal == 'triangle':
            scen_w = self.data['dims']['scen_w']
            scen_h = self.data['dims']['scen_h']
            sd = (scen_w + scen_h) / 2
            r = ((self.state['x'] - sd / 2) ** 2 + (self.state['y'] - sd / 2)
                 ** 2).sqrt()
            rc, idx = r.min(0)
            indices = [i for i in range(len(r)) if i != idx]
            x, y = self.state['x'][indices], self.state['y'][indices]
            x, y = torch.cat([x, x[[0]]], 0), torch.cat([y, y[[0]]], 0)
            d = ((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2).sqrt()
            if rc < tol and ((r[indices] - sd / 4).abs() < tol).all() and ((d -                                                            
                data['goals']['triangle_d']).abs() < difftol).all():
                return True
        else:
            raise ValueError(f'{goal} is not a known goal')
        return False

    def get_scenario_plot(self) -> torch.Tensor:
        '''Returns the visualization of the current scenario.

        Returns:
            scenario_vis: The visualization of the current scenario.
        '''
        scenario_vis = torch.zeros(self.data['dims']['scen_h'] + 2 *
            self.vis_data['margin'], self.data['dims']['scen_w'] + 2 *
            self.vis_data['margin'], 3)
        x = (self.state['x'] + self.vis_data['margin'] - self.data['dims'][
            'obj_w'] / 2).round().int()
        y = (self.state['y'] + self.vis_data['margin'] - self.data['dims'][
            'obj_h'] / 2).round().int()
        for xi, yi, objtype in zip(x, y, self.state['objtypes']):
            scenario_vis[yi:yi + self.data['dims']['obj_h'], xi:xi + self.data[
                'dims']['obj_w'], :] = self.vis_data['color_objs'][objtype]
        x_hand = (self.state['x_hand'] + self.vis_data['margin'] -
            self.vis_data['hand_w'] / 2).round().int()
        y_hand = (self.state['y_hand'] + self.vis_data['margin'] - self.state[
            'hand_h'] / 2).round().int()
        scenario_vis[y_hand:y_hand + self.state['hand_h'].round().int(),
            x_hand:x_hand + self.vis_data['hand_w'], :] = 1
        return scenario_vis


def test(config: Dict[str, Any], dataset: inputdata.SynthActionsWrapper, model:
    lrnn.HLRNN, callback_fn: Callable[[SynthActionsEnvironment, str,
    torch.Tensor], None] = None) -> Tuple[float, float]:
    '''Obtains the metrics of the action selection given the parameters.

    Arguments:
        config: The HLRNN action selection test configuration parameters.
        dataset: A dataset with the synthetic actions input data.
        model: The HLRNN model.
        callback_fn: Code executed at every iteration given the simulation
            environment object, the desired goal and the predicted sample.

    Returns:
        achieved: The estimated probability that the given goal is achieved.
        avg_num_samples: The average number of samples required to achieve the
            goal.
    '''
    min_samples_real, max_samples_real = 100, 150
    env = SynthActionsEnvironment(dataset)
    goals = dataset.all_lbls[dataset.classif]
    goals = {name: one_hot.unsqueeze(0) for name, one_hot in zip(goals,
        torch.eye(len(goals)))}
    model.eval()
    hs = []
    achieved, num_samples = 0, []
    with torch.no_grad():
        for trial in range(config['num_trials']):
            if trial % env.data['traj_p']['plans_scen'] == 0:
                env.generate_scenario()
                x = env.get_sample(torch.tensor(0.), torch.tensor(0.),
                    torch.tensor(0.))
            goal = random.choice(env.data['goals']['names'])
            for sample in range(config['num_samples_factor'] *
                max_samples_real):
                pred, *hs = model.closed_loop_forward_step(x.unsqueeze(0),
                    goals[goal], *hs)
                hs = [h.detach() if h is not None else None for h in hs[:-3]] \
                     + hs[-3:]
                x, achvd_goal = env.move_hand(*env.get_movement_from_sample(
                    pred.squeeze(0)))
                if callback_fn is not None:
                    callback_fn(env, goal, pred)
                if achvd_goal == goal:
                    achieved += 1
                    num_samples.append(sample + 1)
                    break
    return achieved / config['num_trials'], None if len(num_samples) == 0 \
        else sum(num_samples) / len(num_samples)


def main() -> None:
    '''Visualizes the synthetic actions environment for hand movements.'''
    dataset = SynthPlanInput()
    env = SynthActionsEnvironment(dataset)

    class HandMotion:
        def __init__(self, env):
            self.env = env
            self.restart()

        def generate_movement(self, env_state):
            if self.state['x_hand'].shape[0] == 0:
                self.generate_primitive_traj(env_state)
            vx = (self.state['x_hand'][0] - env_state['x_hand']) * self.env[
                'traj_p']['fs']
            vy = (self.state['y_hand'][0] - env_state['y_hand']) * self.env[
                'traj_p']['fs']
            hand_h = self.state['hand_h'][0]
            self.state['x_hand'] = self.state['x_hand'][1:]
            self.state['y_hand'] = self.state['y_hand'][1:]
            self.state['hand_h'] = self.state['hand_h'][1:]
            return vx, vy, hand_h

        def generate_primitive_traj(self, env_state):
            if len(self.state['primitives']) == 0:
                self.init_action(env_state)
            primitive = self.state['primitives'].pop(0)
            x1, y1, x2, y2 = self.state['x1'], self.state['y1'], self.state[
                'x2'], self.state['y2']
            a = self.env['traj_p']['amin'] + torch.rand([]) * (self.env[
                'traj_p']['amax'] - self.env['traj_p']['amin'])
            if primitive == 'move':
                distance = ((x1 - env_state['x_hand']) ** 2 + (y1 - env_state[
                    'y_hand']) ** 2).sqrt() + 1e-5
                t = torch.arange(1 / self.env['traj_p']['fs'], (6 * distance /
                    a).sqrt(), 1 / self.env['traj_p']['fs'])
                a = 6 * distance / t[-1] ** 2
                s = (a * t ** 2 / 2 - a * t ** 3/ (3 * t[-1])) / distance
                x_hand = env_state['x_hand'] + (x1 - env_state['x_hand']) * s
                y_hand = env_state['y_hand'] + (y1 - env_state['y_hand']) * s
                hand_h = env_state['hand_h'] * torch.ones(t.shape[0])
                if self.state['action'] == 'picknplace':
                    t_open = (self.env['traj_p']['tmin_op'] + torch.rand([]) \
                        * (self.env['traj_p']['tmax_op'] - self.env['traj_p'][
                        'tmin_op'])).clamp(1 / self.env['traj_p']['fs'], t[-1])
                    n_open = (t_open * self.env['traj_p']['fs']).round().int()
                    hand_h[-n_open:] = torch.linspace(env_state['hand_h'],
                        self.env['dims']['ophand_h'], n_open)
            elif primitive == 'pick':
                t_grip = self.env['traj_p']['tmin_cl'] + torch.rand([]) * (
                    self.env['traj_p']['tmax_cl'] - self.env['traj_p'][
                    'tmin_cl'])
                t = torch.arange(1 / self.env['traj_p']['fs'], t_grip, 1 /
                    self.env['traj_p']['fs'])
                x_hand = env_state['x_hand'] * torch.ones(t.shape[0])
                y_hand = env_state['y_hand'] * torch.ones(t.shape[0])
                hand_h = env_state['hand_h'] + (self.env['dims']['clhand_h'] -
                    env_state['hand_h']) * t / t[-1]
            elif primitive == 'carry':
                distance = ((x2 - env_state['x_hand']) ** 2 + (y2 - env_state[
                    'y_hand']) ** 2).sqrt() + 1e-5
                t = torch.arange(1 / self.env['traj_p']['fs'], 6 * (distance /
                    a).sqrt(), 1 / self.env['traj_p']['fs'])
                a = 6 * distance / t[-1] ** 2
                s = (a * t ** 2 / 2 - a * t ** 3/ (3 * t[-1])) / distance
                x_hand = env_state['x_hand'] + (x2 - env_state['x_hand']) * s
                y_hand = env_state['y_hand'] + (y2 - env_state['y_hand']) * s
                hand_h = env_state['hand_h'] * torch.ones(t.shape[0])
            elif primitive == 'place':
                t_grip = self.env['traj_p']['tmin_cl'] + torch.rand([]) * (
                    self.env['traj_p']['tmax_cl'] - self.env['traj_p'][
                    'tmin_cl'])
                t = torch.arange(1 / self.env['traj_p']['fs'], t_grip, 1 /
                    self.env['traj_p']['fs'])
                x_hand = env_state['x_hand'] * torch.ones(t.shape[0])
                y_hand = env_state['y_hand'] * torch.ones(t.shape[0])
                hand_hf = env_state['hand_h'] + torch.rand([]) * (self.env[
                    'dims']['ophand_h'] - env_state['hand_h'])
                hand_h = env_state['hand_h'] + (hand_hf - env_state[
                    'hand_h']) * t / t[-1]
            else:
                raise ValueError(f'{primitive} is not a known primitive')
            self.state['x_hand'] = x_hand
            self.state['y_hand'] = y_hand
            self.state['hand_h'] = hand_h

        def init_action(self, env_state):
            actions = ['picknplace', 'push', 'pull']
            action = random.choice(actions)
            tgt_obj = random.randrange(self.env['n_objects'])
            tgt_x = torch.rand([]) * self.env['dims']['scen_w']
            tgt_y = torch.rand([]) * self.env['dims']['scen_h']
            print(f'Trying to {action} object type '
                  f'{env_state["objtypes"][tgt_obj]} from position ('
                  f'{env_state["x"][tgt_obj]:.1f}, '
                  f'{env_state["y"][tgt_obj]:.1f}) to position ({tgt_x:.1f}, '
                  f'{tgt_y:.1f})')
            if self.env['affordances'][env_state['objtypes'][tgt_obj],
                actions.index(action)]:
                print('Should succeed')
            else:
                print('Should fail')
            if action == 'picknplace':
                primitives = ['move', 'pick', 'carry', 'place']
                shift_x = 0
                shift_y = 0
            elif action == 'push':
                primitives = ['move', 'carry']
                distance = ((tgt_x - env_state['x'][tgt_obj]) ** 2 + (tgt_y -
                    env_state['y'][tgt_obj]) ** 2).sqrt() + 1e-5
                shift_x = self.env['dims']['obj_w'] * (env_state['x'][tgt_obj]
                    - tgt_x) / distance
                shift_y = self.env['dims']['obj_h'] * (env_state['y'][tgt_obj]
                    - tgt_y) / distance
            elif action == 'pull':
                primitives = ['move', 'carry']
                distance = ((tgt_x - env_state['x'][tgt_obj]) ** 2 + (tgt_y -
                    env_state['y'][tgt_obj]) ** 2).sqrt() + 1e-5
                shift_x = self.env['dims']['obj_w'] * (tgt_x - env_state['x'][
                    tgt_obj]) / distance
                shift_y = self.env['dims']['obj_h'] * (tgt_y - env_state['y'][
                    tgt_obj]) / distance
            else:
                raise ValueError(f'{action} is not a known action')
            x1 = env_state['x'][tgt_obj] + shift_x + self.env['traj_p'][
                'hand_sigma'] * torch.randn([])
            y1 = env_state['y'][tgt_obj] + shift_y + self.env['traj_p'][
                'hand_sigma'] * torch.randn([])
            x2 = tgt_x + shift_x + self.env['traj_p']['hand_sigma'] * \
                 torch.randn([])
            y2 = tgt_y + shift_y + self.env['traj_p']['hand_sigma'] * \
                 torch.randn([])
            self.state = {'action': action, 'primitives': primitives, 'x1': x1,
                'y1': y1, 'x2': x2, 'y2': y2}

        def restart(self):
            self.state = {'primitives': [], 'x_hand': torch.tensor([]),
                'y_hand': torch.tensor([]), 'hand_h': torch.tensor([])}

    motion = HandMotion(env.data)

    fig, axs = plt.subplots(1, 3)
    axs[0].set_title('scenario')
    axs[1].set_title('sample')
    axs[2].set_title('picknplace push pull')
    scenarioplot = axs[0].imshow(env.get_scenario_plot())
    sample = env.get_sample(torch.tensor(0.), torch.tensor(0.), torch.tensor(
        0.))
    sampleplot = axs[1].imshow(dataset.get_input_plot(sample))
    axs[2].imshow(env.data['affordances'].unsqueeze(2) * env.vis_data[
        'colors'][:env.data['affordances'].shape[0]].unsqueeze(1))
    while True:
        for _ in range(1000):
            plt.show(block=False)
            plt.pause(0.033)
            vx, vy, hand_h = motion.generate_movement(env.state)
            sample, goal = env.move_hand(vx, vy, hand_h)
            if goal is not None:
                print(f'{goal} goal achieved!')
            scenarioplot.set_data(env.get_scenario_plot())
            sampleplot.set_data(dataset.get_input_plot(sample))
        env.generate_scenario()
        motion.restart()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys

    main(*sys.argv[1:])
