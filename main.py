import pandas as pd
import jieba
import pinyin
import json
import argparse
import time
import torch
from models.converter import TextConverter
from generate import MAX_VOCAB


MAX_LINES=1000000

def str2emoji(st, dct, dct_pinyin):
    rlist=[]
    wlist=jieba.cut(st)
    #print(wlist)
    for w in wlist:
        if w in dct:
            rlist.append(dct[w])
        elif len(w)>1:
            for c in w:
                if c in dct:
                    rlist.append(dct[c])
                else:
                    rlist.append(c)
        else:
            rlist.append(w)
    res=''.join(rlist)
    return res

def str2emoji_deep(st, dct, dct_pinyin):
    english=set([chr(ord('A')+i) for i in range(26)]+[chr(ord('a')+i) for i in range(26)])
    rlist=[]
    wlist=jieba.cut(st)
    for w in wlist:
        if w in dct:
            rlist.append(dct[w])
        elif len(w)>1:
            for c in w:
                if c in dct:
                    rlist.append(dct[c])
                elif c in english:
                    rlist.append(c)
                elif pinyin.get(c, format='strip') in dct_pinyin:
                    rlist.append(dct_pinyin[pinyin.get(c, format='strip')])
                else:
                    rlist.append(c)
        elif w in english:
            rlist.append(w)
        elif pinyin.get(w, format='strip') in dct_pinyin:
            rlist.append(dct_pinyin[pinyin.get(w, format='strip')])
        else:
            rlist.append(w)
    res=''.join(rlist)
    return res

def emoji2str(st,*args):
    model_path="output/e2s.pkl"
    data_path="data/train/e2s/data.txt"
    vocab_path="data/train/e2s/vocab.txt"
    tgt_max_len=64
    model=torch.load(model_path)
    model.eval()
    convert=TextConverter(vocab_path,max_vocab=MAX_VOCAB,min_freq=0)
    src=torch.tensor(convert.text_to_arr(st))
    src=src.unsqueeze(0)
    tgt_list=[1]
    for i in range(tgt_max_len):
        tgt=torch.tensor(tgt_list)
        tgt=tgt.unsqueeze(0)
        out=model(src,tgt)
        if int(out.argmax(-1)[-1,0])==1:
            break
        tgt_list.append(int(out.argmax(-1)[-1,0]))
    print(convert.arr_to_text(tgt_list[1:]))

def translate_s2e(func,inp,out,dct,dct_pinyin):
    if inp:
        stime=time.time()
        rlist=[]
        with open(inp,'r',encoding='utf8') as f:
            for i,line in enumerate(f):
                if i>=MAX_LINES:
                    break
                rlist.append(func(line.strip(),dct,dct_pinyin))
        num_lines=len(rlist)
        etime=time.time()
        print(f"Translation done. {num_lines} lines totally")
        print(f"Translation cost {etime-stime} seconds.")
        with open(out,'w',encoding='utf8') as f:
            f.write('\n'.join(rlist))
    else:
        while True:
            s=input('input:')
            if s=='':
                break
            e=func(s,dct,dct_pinyin)
            print(e)

def translate_e2s(inp,out,model_path,data_path,vocab_path,tgt_max_len=64):
    model=torch.load(model_path)
    model.eval()
    convert=TextConverter(vocab_path,max_vocab=MAX_VOCAB,min_freq=0)
    def func(st):
        src=torch.tensor(convert.text_to_arr(st))
        src=src.unsqueeze(0)
        tgt_list=[1]
        for i in range(tgt_max_len):
            tgt=torch.tensor(tgt_list)
            tgt=tgt.unsqueeze(0)
            out=model(src,tgt)
            if int(out.argmax(-1)[-1,0])==1:
                break
            tgt_list.append(int(out.argmax(-1)[-1,0]))
        return convert.arr_to_text(tgt_list[1:])
    if inp:
        stime=time.time()
        rlist=[]
        with open(inp,'r',encoding='utf8') as f:
            for i,line in enumerate(f):
                if i>=MAX_LINES:
                    break
                rlist.append(func(line.strip()))
        num_lines=len(rlist)
        etime=time.time()
        print(f"Translation done. {num_lines} lines totally")
        print(f"Translation cost {etime-stime} seconds.")
        with open(out,'w',encoding='utf8') as f:
            f.write('\n'.join(rlist))
    else:
        while True:
            s=input('input:')
            if s=='':
                break
            stime=time.time()
            e=func(s)
            etime=time.time()
            print(e)
            print(f"Translation cost {etime-stime} seconds.")

def main():
    parser = argparse.ArgumentParser(description='Process args')
    parser.add_argument('-m', 
                        default='s2e',
                        type=str,
                        choices=['s2e', 's2edeep', 'e2s', 'e2sdeep'],
                        help="Choose a mode.")
    parser.add_argument('-i', 
                        default='',
                        type=str,
                        help='The path of the file to translate. Interactive mode will be used if empty.')
    parser.add_argument('-o', 
                        default='',
                        type=str,
                        help='The output path of the translated sentences.')
    parser.add_argument('--interactive', 
                        action='store_true',
                        help='Interactive mode.')

    args = parser.parse_args()
    if args.i!='' and args.o=='':
        raise Exception("Output path should be given.")

    dct=json.loads(open('data/dict.json',encoding='utf8').read())
    dct_pinyin=json.loads(open('data/dict_pinyin.json',encoding='utf8').read())

    # string to emoji, normal to abstract.
    if args.m=='s2e':
        translate_s2e(str2emoji,args.i,args.o,dct,dct_pinyin)
    # normal to abstract, deep
    elif args.m=='s2edeep':
        translate_s2e(str2emoji_deep,args.i,args.o,dct,dct_pinyin)
    elif args.m=='e2s':
        model_path="output/e2s.pkl"
        data_path="data/train/e2s/data.txt"
        vocab_path="data/train/e2s/vocab.txt"
        tgt_max_len=64
        translate_e2s(args.i,args.o,model_path,data_path,vocab_path,tgt_max_len)
    elif args.m=='e2sdeep':
        pass
    else:
        print("Choose a mode from ['s2e', 's2edeep', 'e2s', 'e2sdeep']")  


if __name__=='__main__':
    main()
    




