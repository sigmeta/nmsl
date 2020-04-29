import os
import argparse
from tqdm import tqdm,trange
from models.trainer import Trainer_Transformer
from models.converter import TextConverter
from generate import MAX_VOCAB


def Opt():
    parser = argparse.ArgumentParser(description="Argparse of Poetry-Pytorch")
    #path
    parser.add_argument('-p','--data_path', default='data/train/e2s/data.txt')
    parser.add_argument('-v','--vocab_path', default='data/train/e2s/vocab.txt')
    parser.add_argument('-o','--output_path', default='output/e2s.pkl')
    #model config
    parser.add_argument('--model', default='transformer', choices=['transformer','lstm'])
    parser.add_argument('--hidden_dims', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_encoder_layers', type=int, default=2, help="Transformer encoder layers")
    parser.add_argument('--num_decoder_layers', type=int, default=2, help="Transformer decoder layers")
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    # training params
    parser.add_argument('--train', action='store_true',default=False)
    parser.add_argument('--test', action='store_true',default=False)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--src_max_len', type=int, default=64)
    parser.add_argument('--tgt_max_len', type=int, default=64)
    parser.add_argument('--src_text', default="你真是给👴整😁🌶️", help="used when --model=transformer, it is like a title")
    parser.add_argument('--tgt_text', default="", help="a start of the poetry")
    args = parser.parse_args()
    return args


def main():
    opt=Opt()
    print(opt)
    convert=TextConverter(opt.data_path,opt.vocab_path,max_vocab=MAX_VOCAB,min_freq=0)
    trainer=Trainer_Transformer(convert,opt)
    if opt.train:
        trainer.train()
    if opt.test:
        trainer.test()


if __name__=="__main__":
    main()