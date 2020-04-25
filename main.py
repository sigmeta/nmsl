import pandas as pd
import jieba
import pinyin
import json
import argparse

MAX_LINES=100000

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

def translate(func,inp,out,dct,dct_pinyin):
    if inp:
        rlist=[]
        with open(inp,'r',encoding='utf8') as f:
            for i,line in enumerate(f):
                if i>=MAX_LINES:
                    break
                rlist.append(func(line.strip(),dct,dct_pinyin))
        num_lines=len(rlist)
        print(f"Translation done. {num_lines} lines totally")
        with open(out,'w',encoding='utf8') as f:
            f.write('\n'.join(rlist))
    else:
        while True:
            s=input('input:')
            if s=='':
                break
            e=func(s,dct,dct_pinyin)
            print(e)

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
        translate(str2emoji,args.i,args.o,dct,dct_pinyin)
    # normal to abstract, deep
    elif args.m=='s2edeep':
        translate(str2emoji_deep,args.i,args.o,dct,dct_pinyin)
    elif args.m=='e2s':
        pass
    elif args.m=='e2sdeep':
        pass
    else:
        print("Choose a mode from ['s2e', 's2edeep', 'e2s', 'e2sdeep']")


if __name__=='__main__':
    main()





