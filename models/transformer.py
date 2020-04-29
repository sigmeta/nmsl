import torch
import torch.nn as nn
from torch.nn import Transformer
import math


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MyTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, 
                dim_feedforward=2048, dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None,
                vocab_size=5000):
        super(MyTransformer,self).__init__()
        self.transformer=Transformer(d_model=d_model,nhead=nhead,num_encoder_layers=num_encoder_layers,num_decoder_layers=num_decoder_layers,
                                    dim_feedforward=dim_feedforward,dropout=dropout,activation=activation,custom_encoder=custom_encoder,
                                    custom_decoder=custom_decoder)
        self.embedding=nn.Embedding(vocab_size,d_model)
        self.positional_encoding=PositionalEncoding(d_model,dropout=dropout)
        self.linear=nn.Linear(d_model,vocab_size)
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src_embed=self.embedding(src)
        tgt_embed=self.embedding(tgt)
        src_embed=self.positional_encoding(src_embed.transpose(0,1))
        tgt_embed=self.positional_encoding(tgt_embed.transpose(0,1))
        #tgt_mask=self.generate_square_subsequent_mask(tgt.size(-1)).cuda()
        transormer_out=self.transformer(src_embed,tgt_embed,src_mask=src_mask,tgt_mask=tgt_mask,memory_mask=memory_mask,
                                src_key_padding_mask=src_key_padding_mask,tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask)
        out=self.linear(transormer_out)
        return out
