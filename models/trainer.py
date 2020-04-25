import numpy as np
import copy
import torch
import torch.nn as nn
from models.transformer import MyTransformer
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm,trange


def get_dataset(text_path, arr_to_idx, src_max_len, tgt_max_len):
    with open(text_path, 'r', encoding='utf8') as f:
        text = f.read()
    src_list=[]
    tgt_list=[]
    label_list=[]
    for t in text.strip().split('\n'):
        src_raw,tgt_raw=t.split('\t')
        src=arr_to_idx(src_raw)[:src_max_len]
        tgt=arr_to_idx(tgt_raw)[:tgt_max_len-1]
        label=copy.copy(tgt)
        tgt=[1]+tgt
        label.append(1)
        while len(src)<src_max_len:
            src.append(0)
        while len(tgt)<tgt_max_len:
            tgt.append(0)
            label.append(0)
        src_list.append(src)
        tgt_list.append(tgt)
        label_list.append(label)
    src_tensor=torch.tensor(src_list)
    tgt_tensor=torch.tensor(tgt_list)
    src_padding=torch.eq(src_tensor,0)
    tgt_padding=torch.eq(tgt_tensor,0)
    label_tensor=torch.tensor(label_list)
    print(src_tensor)
    return TensorDataset(src_tensor,tgt_tensor,src_padding,tgt_padding,label_tensor)

def get_data(convert,opt):
    dataset= get_dataset(opt.data_path, convert.text_to_arr, opt.src_max_len, opt.tgt_max_len)
    return DataLoader(dataset, opt.batch_size, shuffle=True)

class Trainer_Transformer(object):
    def __init__(self, convert, opt):
        self.config=opt
        self.convert=convert
    
    def train(self):
        tgt_mask=torch.triu(torch.ones(self.config.tgt_max_len,self.config.tgt_max_len),1)
        tgt_mask=tgt_mask.masked_fill(tgt_mask.byte(),value=torch.tensor(float('-inf')))
        model=MyTransformer(d_model=self.config.hidden_dims, nhead=self.config.num_heads, 
                            num_encoder_layers=self.config.num_encoder_layers, num_decoder_layers=self.config.num_decoder_layers, 
                            dim_feedforward=4*self.config.hidden_dims, dropout=self.config.dropout,vocab_size=self.convert.vocab_size)
        if torch.cuda.is_available() and self.config.use_gpu:
            print("using gpu to accelerate")
            model=model.cuda()
            tgt_mask=tgt_mask.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        criterion=nn.CrossEntropyLoss(ignore_index=0,size_average=True)
        training_data=get_data(self.convert,self.config)
        for epoch in range(self.config.num_epochs):
            print("epoch:",epoch)
            running_loss=0
            updates=0
            for step,data in enumerate(training_data):
                src,tgt,src_padding,tgt_padding,label=data
                if torch.cuda.is_available() and self.config.use_gpu:
                    src=src.cuda()
                    tgt=tgt.cuda()
                    label=label.cuda()
                    src_padding=src_padding.cuda()
                    tgt_padding=tgt_padding.cuda()
                optimizer.zero_grad()
                out=model(src, tgt, tgt_mask=tgt_mask,
                        src_key_padding_mask=src_padding, tgt_key_padding_mask=tgt_padding, memory_key_padding_mask=src_padding)
                #out=model(tgt,use_gpu=self.config.use_gpu and torch.cuda.is_available())
                #print(out,tgt)
                out=out.transpose(0,1).contiguous().view(-1,self.convert.vocab_size)
                label=label.view(-1)
                loss=criterion(out,label)
                loss.backward()
                optimizer.step()
                running_loss+=loss.item()
                updates+=1
                #print(loss)
                if step%100==0:
                    print(f"Epoch: {epoch}, Step: {step}, Loss: {running_loss/updates}")
            print("training loss:",running_loss/updates)
        torch.save(model.cpu(),self.config.output_path)

    def test(self):
        model=torch.load(self.config.output_path)
        model.eval()
        model=model.cuda()
        src=torch.tensor(self.convert.text_to_arr(self.config.src_text))
        src=src.cuda().unsqueeze(0)
        tgt_list=[1]
        for i in range(self.config.tgt_max_len):
            tgt=torch.tensor(tgt_list)
            tgt=tgt.cuda().unsqueeze(0)
            out=model(src,tgt)
            if int(out.argmax(-1)[-1,0])==1:
                break
            tgt_list.append(int(out.argmax(-1)[-1,0]))
        print(self.convert.arr_to_text(tgt_list)[1:])

    def predict(self):
        pass