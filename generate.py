import os
import time
import json


MAX_LINES=1000000
MAX_VOCAB=10000

def main():
    src_path="data/train/out2.txt"
    tgt_path="data/train/tieba.dialogues"
    dct_path="data/dict.json"
    out_path="data/train/e2sdeep"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    vocab={}
    slist=[]
    tlist=[]
    # add pairs in the dictionary
    with open(dct_path,encoding='utf8') as f:
        dct=json.loads(f.read())
        for k in dct:
            slist.append(dct[k])
            tlist.append(k)
            for c in k+dct[k]:
                if c in vocab:
                    vocab[c]+=1
                else:
                    vocab[c]=1
    with open(src_path,encoding='utf8') as f:
        for i,line in enumerate(f):
            if i>=MAX_LINES:
                break
            txt=line.strip().replace('\t','').replace(' ','')
            for c in txt:
                if c in vocab:
                    vocab[c]+=1
                else:
                    vocab[c]=1
            slist.append(txt)
    with open(tgt_path,encoding='utf8') as f:
        for i,line in enumerate(f):
            if i>=MAX_LINES:
                break
            txt=line.strip().replace('\t','').replace(' ','')
            for c in txt:
                if c in vocab:
                    vocab[c]+=1
                else:
                    vocab[c]=1
            tlist.append(txt)
    vocab_count_list = []
    for word in vocab:
        vocab_count_list.append((word, vocab[word]))
    vocab_count_list.sort(key=lambda x: x[0])
    vocab_count_list.sort(key=lambda x: x[1], reverse=True)
    if len(vocab_count_list) > MAX_VOCAB:
        vocab_count_list = vocab_count_list[:MAX_VOCAB-3]
    vlist= ['<pad>\t0','<eos>\t0','<unk>\t0']+[x[0]+'\t'+str(x[1]) for x in vocab_count_list]
    with open(os.path.join(out_path,'vocab.txt'),'w',encoding='utf8') as f:
        f.write('\n'.join(vlist))
    rlist=[slist[i]+'\t'+tlist[i] for i in range(len(slist))]
    with open(os.path.join(out_path,'data.txt'),'w',encoding='utf8') as f:
        f.write('\n'.join(rlist))

if __name__=='__main__':
    stime=time.time()
    main()
    etime=time.time()
    print(f"Generation cost {etime-stime} seconds.")