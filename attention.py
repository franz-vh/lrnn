#!/usr/bin/env python3

'''A class with the self-supervised attention learning system.'''

import lrnn
import torch
from torch import nn
import torch.nn.functional as F
import math
import os
from typing import List, Tuple
    

class Attention(nn.Module):
    '''An autoencoder with an attention mechanism and output memory.

    Attributes:
        fix_in_features: An integer with the size of the non-attention
            component of each input sample.
        att_in_features: An integer with the size of the attention component of
            each input sample.
        nhead: An integer with the number of heads in the multiheadattention
            model.
        query_gru_features: An integer with the size of the GRU hidden layer
            used to generate the query.
        query_hidden_features: An integer with the size of the linear hidden
            layer used to generate the query.
        dec_gru_features: An integer with the size of the GRU hidden layer in
            the decoder.
        dec_hidden_features: An integer with the size of the linear hidden
            layer in the decoder.
        train_horizon: An integer with the number of timesteps ahead of the
            prediction.
        dropout_p: A float with the probability of an element in the dropout
            layer to be zero-ed.
        noise_std: A float with the standard deviation of the noise layer.
        return_attw: A boolean indicating whether the forward function will
            return the attention weights.
        norm_sparse_vbles: A list of normalized ND tensors of floats with the
            variables over which to apply the sparsity loss.
        q_gru: A PyTorch module with the GRU hidden layer used to generate the
            query.
        q_hidden: A PyTorch module with the linear hidden layer used to
            generate the query.
        weight_q: A 2D tensor of floats with the learnable query weights of the
            module of shape (query_hidden_features, att_in_features * nhead).
            The values are initialized according to Xavier initialization.
        dropout_1d: A PyTorch module with the dropout layer.
        noise: A PyTorch module with the noise layer.
        dec_gru: A PyTorch module with the GRU hidden layer in the decoder.
        dec_hidden: A PyTorch module with the linear hidden layer in the
            decoder.
        dec_out: A PyTorch module with the decoder output layer.
        _last_mask: A torch tensor with the mask with the output elements to
             be considered in the loss function.
    '''
    def __init__(self, fix_in_features: int, att_in_features: int, nhead: int,
        query_gru_features: int, query_hidden_features: int, dec_gru_features:
        int, dec_hidden_features: int, train_horizon: int, dropout_p: float,
        noise_std: float, return_attw: bool = False) -> None:
        '''Inits Attention.

        Arguments:
            fix_in_features: The size of the non-attention component of each
                input sample.
            att_in_features: The size of the attention component of each input
                sample.
            nhead: The number of heads in the multiheadattention model.
            query_gru_features: The size of the GRU hidden layer used to
                generate the query.
            query_hidden_features: The size of the linear hidden layer used to
                generate the query.
            dec_gru_features: The size of the GRU hidden layer in the decoder.
            dec_hidden_features: The size of the linear hidden layer in the
                decoder.
            train_horizon: The number of timesteps ahead of the prediction.
            dropout_p: The probability of an element in the dropout layer to be
                zero-ed.
            noise_std: The standard deviation of the noise layer.
            return_attw: If true, the forward function will return the
                attention weights.
        '''
        super(Attention, self).__init__()
        self.fix_in_features, self.att_in_features = fix_in_features, \
            att_in_features
        self.nhead = nhead
        self.query_gru_features, self.query_hidden_features = \
            query_gru_features, query_hidden_features
        self.dec_gru_features, self.dec_hidden_features = dec_gru_features, \
            dec_hidden_features
        self.train_horizon = train_horizon
        self.dropout_p, self.noise_std = dropout_p, noise_std
        self.return_attw = return_attw
        self.norm_sparse_vbles = []
        self._last_mask = torch.tensor([])
        self.q_gru = nn.GRUCell(fix_in_features + 2 * att_in_features * nhead,
            query_gru_features)
        self.q_hidden = nn.Linear(query_gru_features + fix_in_features +
            att_in_features * nhead, query_hidden_features)
        self.weight_q = nn.Parameter(torch.empty(query_hidden_features,
            att_in_features * nhead))
        self.dropout_1d = nn.Dropout1d(dropout_p)
        self.noise = lrnn.Noise(noise_std)
        self.dec_gru = nn.GRU(fix_in_features + att_in_features * nhead,
            dec_gru_features)
        self.dec_hidden = nn.Linear(dec_gru_features + fix_in_features +
            att_in_features * nhead + att_in_features, dec_hidden_features)
        self.dec_out = nn.Linear(dec_hidden_features, att_in_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''Initializes weights following Xavier initialization and bias to 0'''
        self.q_gru.reset_parameters()
        nn.init.xavier_uniform_(self.q_hidden.weight)
        nn.init.zeros_(self.q_hidden.bias)
        nn.init.xavier_uniform_(self.weight_q)
        self.dec_gru.reset_parameters()
        nn.init.xavier_uniform_(self.dec_hidden.weight)
        nn.init.zeros_(self.dec_hidden.bias)
        nn.init.xavier_uniform_(self.dec_out.weight)
        nn.init.zeros_(self.dec_out.bias)

    def forward(self, x_fix: torch.Tensor, x_att: List[torch.Tensor], q_gru_0:
        torch.Tensor = None, dec_gru_0: torch.Tensor = None, query_0:
        torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor,
        torch.Tensor]:
        '''Performs a forward pass.

        Arguments:
            x_fix: The non-attention component of the input batch.
            x_att: The attention component of the input batch.
            q_gru_0: The initial query GRU hidden state. Defaults to zeros if
                not provided.
            dec_gru_0: The initial decoder GRU hidden state. Defaults to zeros
                if not provided.
            query_0: The initial query. Defaults to ones if not provided.

        Returns:
            pred: The output batch (or prediction).
            y: The desired output batch (only in training mode).
            attw: The attention weights if required.
            q_gru_n: The final query GRU hidden state.
            dec_gru_n: The final query GRU hidden state.
            query_n: The final query.
        '''
        del self.norm_sparse_vbles[:]
        q_gru_sample = q_gru_0
        if q_gru_sample == None:
            q_gru_sample = torch.zeros(self.query_gru_features)
        query = query_0
        if query == None:
            query = torch.ones(self.att_in_features * self.nhead)
        attw_list = []
        att_list = []
        gru_list = []
        for att_sample, fix_sample in zip(x_att, x_fix):
            attw_sample = F.softmax(query.reshape(self.nhead,
                self.att_in_features) @ att_sample.T / math.sqrt(
                self.att_in_features), dim=1)
            att_sample = (attw_sample @ att_sample).flatten()
            q_gru_sample = self.q_gru(torch.cat([fix_sample, att_sample,
                query]), q_gru_sample)
            query = torch.relu(self.q_hidden(torch.cat([q_gru_sample,
                fix_sample, att_sample]))) @ self.weight_q
            attw_list.append(attw_sample)
            att_list.append(att_sample)
        attw = torch.stack(attw_list)
        att_a = torch.stack(att_list)
        self.norm_sparse_vbles.append(attw)
        if self.training:
            if dec_gru_0 == None:
                dec_gru_0 = torch.zeros(1, self.dec_gru_features)
            dec_attw = F.pad(attw.permute(2, 1, 0), (self.train_horizon, 0),
                mode='replicate').permute(2, 1, 0)[:-self.train_horizon]
            dec_att_a = (dec_attw @ x_att).reshape(x_att.shape[0], -1)
            dec_x = torch.cat([x_fix, dec_att_a], -1)
            dec_x_obj = self.noise(self.dropout_1d(x_att.permute(2, 1,
                0)).permute(2, 1, 0))
            dec_gru_a, dec_gru_n = self.dec_gru(dec_x, dec_gru_0)
            pred = torch.relu(self.dec_out(torch.relu(self.dec_hidden(
                torch.cat([torch.cat([dec_gru_a, dec_x], 1).unsqueeze(
                1).expand(-1, x_att.shape[1], -1), dec_x_obj], 2)))))
            if self.return_attw:
                return pred, x_att, attw, q_gru_sample, dec_gru_n, query
            else:
                return pred, x_att, q_gru_sample, dec_gru_n, query
        else:
            if self.return_attw:
                return torch.cat([x_fix, att_a], -1), attw, q_gru_sample, \
                    None, query
            else:
                return torch.cat([x_fix, att_a], -1), q_gru_sample, None, query

    def get_mask(self, batch_size: int) -> torch.Tensor:
        '''Returns a mask for the loss function.

        Arguments:
            batch_size: The size of the input batch.

        Returns:
            mask: The mask with the output elements to be considered in the
                loss function.
        '''
        if batch_size == self._last_mask.shape[0]:
            return self._last_mask
        mask = torch.ones(batch_size, 1, 1)
        mask[:self.train_horizon] = 0
        self._last_mask = mask
        return mask


def main(dataset_name = 'KitSparseInput') -> None:
    '''Builds and evaluates the attention system.

    Arguments:
        dataset_name: The name of the dataset.
    '''
    dataset = getattr(kitdata, dataset_name)({'splitprops': [0.6, 0.2, 0.2],
        'splitset': 0})
    #device = "cuda" if torch.cuda.is_available() else "cpu"

    model = lrnn.HLRNN(OrderedDict([
            ('downsample_fix', lrnn.Downsample(3)),
            ('downsample_var', lrnn.Downsample(3)),
            ('attention', Attention(dataset.fix_size, dataset.var_size, 3, 125,
                45, 125, 45, 10, .3, .5))]),
        ['input_fix', 'input_var'],
        {'downsample_fix': 'input_fix',
         'downsample_var': 'input_var',
         'attention': ['downsample_fix', 'downsample_var']},
        ['q_gru', 'dec_gru', 'query'],
        {'attention': ['q_gru', 'dec_gru', 'query']})#.to(device)

    def unsup_train(dataloader, model, loss_fn, opt, epoch):
        num_batches = 50
        num_samples = num_batches * dataloader.dataset.batch_size
        model.train()
        hs = [None] * len(model.hidden_names)
        sample = 0
        print_sample = 0
        for *x, _ in dataloader:
            #x = tuple(c.to(device) for c in x)
            sample += x[0].shape[0]
            pred, y, *hs = model(*x, *hs)
            hs = [h.detach() for h in hs]
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sample > print_sample:
                loss = loss.item()
                print(f'loss: {loss:>7f}  [{sample:>5d}/{num_samples:>5d}]')
                print_sample += 10 * dataloader.dataset.batch_size
            if sample > num_samples:
                break

    def full_unsup(dataset, model):
        block_batch_size = 128
        input_batch_size = math.ceil(block_batch_size /
            model.train_scale_factor)
        dataset.set_batch_iter(input_batch_size)
        dataloader = DataLoader(dataset, batch_size=None)
        lr, lmbd, nbeta = 1e-4, 1e-5, 3e-4
        mse_loss_fn = lrnn.MSELoss(model.get_mask)
        weight_decay_loss_fn = lrnn.WeightDecayLoss(model.cur_block)
        norm_sparsity_loss_fn = lrnn.NormSparsityLoss(
            model.cur_block.norm_sparse_vbles)
        loss_fn = lambda pred, y: mse_loss_fn(pred, y) + lmbd * \
            weight_decay_loss_fn() + nbeta * norm_sparsity_loss_fn()
        opt = torch.optim.Adam(model.cur_block.parameters(), lr=lr)
        epochs = 5
        dataset.restart({'splitset': 0})
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}\n-------------------------------')
            unsup_train(dataloader, model, loss_fn, opt, epoch)

    print('\n\nBlock att:\n-------------------------------')
    model.set_cur_block('attention')
    full_unsup(dataset, model)
    print("\nDone!")

    def attention_test(dataset, model):
        block_batch_size = 100
        input_batch_size = math.ceil(block_batch_size /
            model.train_scale_factor)
        dataset.set_batch_iter(input_batch_size)
        if hasattr(dataset, 'set_must_classifs'):
            dataset.set_must_classifs([dataset.classifs[dataset.classif]])
        dataloader = DataLoader(dataset, batch_size=None)
        num_batches = 100
        num_samples = num_batches * block_batch_size
        model.eval()
        hs = [None] * len(model.hidden_names)
        sample, correct, focus = 0, 0, 0
        with torch.no_grad():
            for *x, y in dataloader:
                #x, y = tuple(c.to(device) for c in x), y.to(device)
                _, attw, *hs = model(*x, *hs)
                hs = [h.detach() if h is not None else None for h in hs]
                y = y[::round(1 / model.scale_factor)]
                attw, y = attw[y >= 0], y[y >= 0]
                sample += y.shape[0]
                ws, idcs = attw.max(-1)
                correct_heads = idcs == y.unsqueeze(1)
                correct_samples = correct_heads.any(-1)
                correct += correct_samples.type(torch.float).sum().item()
                ws[~correct_heads] = 0
                focus += ws[correct_samples].max(-1)[0].sum().item()
                if sample > num_samples:
                    break
        focus /= correct
        correct /= sample
        print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Focus: '
              f'{focus:>3f}')

    print('\n\nTest attention:\n-------------------------------')
    if not dataset.random_mirror:
        warnings.warn('The dataset is not mirror-augmented and the attention '
                      'test will only show results for the left hand')
    model.set_cur_block('attention')
    model.cur_block.return_attw = True
    for side in ['left', 'right']:
        dataset.set_classif(f'{side}mainobjectindices')
        print(f'\n{side}:')
        attention_test(dataset, model)

    sampleplot = kitdatavis.KitInputVis(dataset)
    reconstrplot = kitdatavis.KitInputVis(dataset)
    sampleplot.show()
    reconstrplot.show()
    dataset.classif = slice(None, None)
    _, attw_axs = plt.subplots(model.attention.nhead, 1)
    if model.attention.nhead == 1:
        attw_axs = [attw_axs]
    attwplots = []
    for ax in attw_axs:
        attwplots.append(ax.bar(range(4), [0] * 4))
        ax.axis([-.6, 3.4, 0, 1.1])
    dataset.restart({'splitset': 1})
    model.set_cur_block('attention')
    hs = [None] * len(model.hidden_names)
    model.cur_block.return_attw = True
    batch_size = math.ceil(128 / model.train_scale_factor)
    model.train()
    dataset.set_batch_iter(batch_size)
    dataloader = DataLoader(dataset, batch_size=None)
    with torch.no_grad():
        for *x, y in dataloader:
            #x, y = tuple(c.to(device) for c in x), y.to(device)
            x_fix, x_var = x
            pred, _, attw, *hs = model(x_fix, x_var, *hs)
            x_fix = x_fix[::round(1 / model.scale_factor)]
            x_var = x_var[::round(1 / model.scale_factor)]
            orig_data = dataset.recover_orig_data(x_fix, x_var, y)
            sampleplot.init_trial(orig_data)
            reconstrplot.init_trial(dataset.recover_orig_data(x_fix, pred[:, :,
                :dataset.var_size], y))
            n_obj = x_var.shape[1]
            bar_lbls = [coords['name'][:4] + '_l' if coords['name'].endswith(
                'lid') else coords['name'][7:13] if coords['name'].startswith(
                'mixing') else coords['name'][:6] for coords in orig_data['x'][
                'var']]
            for i, (plot, ax) in enumerate(zip(attwplots, attw_axs)):
                plot.remove()
                attwplots[i] = ax.bar(range(n_obj), [0] * n_obj, tick_label=
                    bar_lbls)
                ax.axis([-.6, n_obj - .4, 0, 1.1])
                for bar in attwplots[i]:
                    bar.set_color('brown')
            for i, attw_sample in enumerate(attw):
                plt.show(block=False)
                plt.pause(0.033)
                heats, _ = attw_sample.max(0)
                colors = heats.unsqueeze(1) * torch.tensor([1., .3, .3]) + (1 -
                    heats.unsqueeze(1)) * torch.tensor(sampleplot.def_color)
                sampleplot.update(colors=[sampleplot.def_color] * 4 + [
                    color.numpy() for color in colors])
                reconstrplot.update()
                for attwplot, attw_head in zip(attwplots, attw_sample):
                    for j, (bar, w) in enumerate(zip(attwplot, attw_head)):
                        bar.set_height(w)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import kitdata
    from collections import OrderedDict
    import sys
    import warnings
    import matplotlib.pyplot as plt
    import kitdatavis
    
    main(*sys.argv[1:])
