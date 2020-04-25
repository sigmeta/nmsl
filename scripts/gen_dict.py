import pandas as pd
import jieba
import pinyin
import json


#生成字典
df=pd.read_csv('../data/bible.csv')
dct={}
for i in range(len(df)):
    dct[df.loc[i,'word']]=df.loc[i,'emoji']
dct_pinyin={}
for i in range(len(df)):
    if len(df.loc[i,'word'])==1:
        dct_pinyin[pinyin.get(df.loc[i,'word'], format="strip")]=df.loc[i,'emoji']

open('../data/dict.json','w',encoding='utf8').write(json.dumps(dct,ensure_ascii=False))
open('../data/dict_pinyin.json','w',encoding='utf8').write(json.dumps(dct_pinyin,ensure_ascii=False))