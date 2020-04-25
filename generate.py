import os


MAX_LINES=100000
MAX_VOCAB=1000000

def main():
    src_path="data/train/out1.txt"
    tgt_path="data/train/tieba.dialogues"
    out_path="data/train/e2s"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    dct={}
    slist=[]
    tlist=[]
    with open(src_path,encoding='utf8') as f:
        for i,line in enumerate(f):
            if i>=MAX_LINES:
                break
            txt=line.strip().replace('\t','').replace(' ','')
            for c in txt:
                if c in dct:
                    dct[c]+=1
                else:
                    dct[c]=1
            slist.append(txt)
    with open(tgt_path,encoding='utf8') as f:
        for i,line in enumerate(f):
            if i>=MAX_LINES:
                break
            txt=line.strip().replace('\t','').replace(' ','')
            for c in txt:
                if c in dct:
                    dct[c]+=1
                else:
                    dct[c]=1
            tlist.append(txt)
    vocab_count_list = []
    for word in dct:
        vocab_count_list.append((word, dct[word]))
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
    main()