"""
Code modified from DETR tranformer:
https://github.com/facebookresearch/detr
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""

import copy
from typing import Optional, List
import pickle as cp

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        '''
        tgt: disease queries
        memory: visual features
        '''
        output = tgt
        T,B,C = memory.shape
        intermediate = []
        atten_layers = []
        for n,layer in enumerate(self.layers):
   
            residual=True
            ## Visual-grounded feature learning - output [B, N_disease, dim], ws [B, N_disease, N_visual]
            output,ws = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,residual=residual)
            atten_layers.append(ws)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output,atten_layers


class MultiTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, high_memory, low_memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        '''
        tgt: disease queries
        memory: visual features
        '''
        output = tgt
        #T,B,C = memory.shape
        intermediate = []
        atten_layers = []
        for n,layer in enumerate(self.layers):
   
            residual=True
            ## Visual-grounded feature learning - output [B, N_disease, dim], ws [B, N_disease, N_visual]
            output,ws = layer(output, high_memory, low_memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,residual=residual)
            atten_layers.append(ws)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output,atten_layers
    

class ThreeLevelTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, high_memory, low_memory, global_memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        '''
        tgt: disease queries
        memory: visual features
        '''
        output = tgt
        #T,B,C = memory.shape
        intermediate = []
        atten_layers = []
        for n,layer in enumerate(self.layers):
   
            residual=True
            ## Visual-grounded feature learning - output [B, N_disease, dim], ws [B, N_disease, N_visual]
            output,ws = layer(output, high_memory, low_memory, global_memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,residual=residual)
            atten_layers.append(ws)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output,atten_layers
    



class MultiTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, self_attention=True, low_level_idx=None):
        super().__init__()
    
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.low_level_index = low_level_idx

        if self_attention:
            self.norm1 = nn.LayerNorm(d_model)
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.self_attention = self_attention

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, high_memory, low_memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     residual=True):
        if self.self_attention:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2,ws = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)
            tgt = self.norm1(tgt)
        high_level_index = [i for i in range(tgt.shape[0]) if i not in self.low_level_idx]
        low_tgt = tgt[self.low_level_index]

        low_tgt,ws = self.multihead_attn(query=self.with_pos_embed(low_tgt, query_pos),
                                   key=self.with_pos_embed(low_memory, pos),
                                   value=low_memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)

        high_tgt = tgt[self.high_level_index]

        high_tgt,ws = self.multihead_attn(query=self.with_pos_embed(high_tgt, query_pos),
                                   key=self.with_pos_embed(high_memory, pos),
                                   value=high_memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)

        tgt2 = torch.zeros_like(tgt).type(tgt2.dtype).to(tgt.device)
        tgt2[high_level_index] = high_tgt
        tgt2[self.low_level_idx] = low_tgt
        # attn_weights [B,NUM_Q,T]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt,ws

    ## Apply norm
    def forward_pre(self, tgt, high_memory, low_memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        if self.self_attention:
            tgt2 = self.norm1(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2, ws = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        high_level_index = [i for i in range(tgt.shape[0]) if i not in self.low_level_index]
        
        N_high, B, _ = high_memory.shape
        N_low, B, _ = low_memory.shape
        high_res = int(np.sqrt(N_high)), int(np.sqrt(N_high))
        low_res = int(np.sqrt(N_low)), int(np.sqrt(N_low))
        #
        ## TODO fix attn_weights
        low_tgt2 = tgt2[self.low_level_index]
        low_tgt2,low_attn_weights = self.multihead_attn(query=self.with_pos_embed(low_tgt2, query_pos),
                                   key=self.with_pos_embed(low_memory, pos),
                                   value=low_memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        high_tgt2 = tgt2[high_level_index]
        high_tgt2, high_attn_weights = self.multihead_attn(query=self.with_pos_embed(high_tgt2, query_pos),
                                   key=self.with_pos_embed(high_memory, pos),
                                   value=high_memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        high_attn_weights = high_attn_weights.view(B, -1, high_res[0], high_res[1])
        high_attn_weights = F.interpolate(high_attn_weights, low_res, mode='bilinear').view(B, -1, N_low)
        attns = torch.zeros(tgt.shape[1], tgt.shape[0], low_memory.shape[0]).type(tgt.dtype).to(tgt.device)
        attns[:, high_level_index] = high_attn_weights
        attns[:, self.low_level_index] = low_attn_weights
        tgt2 = torch.zeros_like(tgt2).type(tgt.dtype).to(tgt2.device)
        tgt2[self.low_level_index] = low_tgt2
        tgt2[high_level_index] = high_tgt2

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt,attns
    
    def forward(self, tgt,  high_memory, low_memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                residual=True):
        if self.normalize_before:
            return self.forward_pre(tgt, high_memory, low_memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, high_memory, low_memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,residual)
    


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, self_attention=True):
        super().__init__()
    
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if self_attention:
            self.norm1 = nn.LayerNorm(d_model)
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.self_attention = self_attention

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     residual=True):
        if self.self_attention:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2,ws = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)
            tgt = self.norm1(tgt)
        tgt2,ws = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)


        # attn_weights [B,NUM_Q,T]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt,ws

    ## Apply norm
    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        if self.self_attention:
            tgt2 = self.norm1(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2,ws = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2,attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt,attn_weights
    
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                residual=True):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,residual)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

if __name__ == '__main__':
    decoder_layer = MultiTransformerDecoderLayer(d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1, 
                                                 activation='relu', normalize_before=True, 
                                                low_level_idx=[1,3])
    decoder = MultiTransformerDecoder(decoder_layer=decoder_layer, num_layers=4, norm=torch.nn.LayerNorm(256))
    query = torch.rand(20, 4, 256)
    high_memory = torch.rand(14*14, 4, 256)
    low_memory = torch.rand(28*28, 4, 256)
    x, ws = decoder(query,  high_memory, low_memory)
    print(x.shape)
    print(ws[0].shape)

