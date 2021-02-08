import os
import pandas as pd

from fastai import *
from fastai.text import *
from Bio import Seq
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation, CompoundLocation
#import networkx as nx
from utils import *
from training_helper import *

path = '/home/anhkhoa/Vu_working/NLP/CNNPromoterData'
outpath = '/home/anhkhoa/Vu_working/NLP/Human-Short-Seq/data'

def partition_data(df):
    train_size = int(len(df)*0.85*.9)
    valid_size = int(len(df)*0.85) - train_size
    
    train_df = df.sample(train_size)
    test_val = df.drop(train_df.index)
    valid_df = test_val.sample(valid_size)
    test_df = test_val.drop(valid_df.index)
    train_df['set'] = 'train'
    valid_df['set'] = 'valid'
    test_df['set'] = 'test'
    
    return (train_df, valid_df, test_df)

def get_init_vocab(ret='data'):
    df = next(pd.read_csv(os.path.join(outpath,'human_genome_data.csv'), chunksize=100000))
    cut = int(len(df)*0.8) + 1
    train_df = df[:cut]
    valid_df = df[cut:]
    tok = Tokenizer(GenomicTokenizer, n_cpus=32, pre_rules=[], post_rules=[], special_cases=['xxpad'])
    data = GenomicTextLMDataBunch.from_df(path, train_df, valid_df, bs=400, tokenizer=tok, 
                                  chunksize=10000, text_cols=0, label_cols=1)
    if ret=='data':
        return data
    else:
        return data.vocab

def main():

    fname1 = 'human_non_tata.fa'
    fname2 = 'human_nonprom_big.fa'

    fasta1 = SeqIO.parse(os.path.join(path,fname1), 'fasta')
    seqs1 = [i.seq.__str__() for i in fasta1 if set(i.seq.__str__()) == set('ATGC')]
    seq1_df = pd.DataFrame(seqs1, columns=['Sequence'])
    seq1_df['Promoter'] = 1

    fasta2 = SeqIO.parse(os.path.join(path,fname2), 'fasta')
    seqs2 = [i.seq.__str__() for i in fasta2 if set(i.seq.__str__()) == set('ATGC')]
    seq2_df = pd.DataFrame(seqs2, columns=['Sequence'])
    seq2_df['Promoter'] = 0

    seq1_df.drop_duplicates(inplace=True)
    seq2_df.drop_duplicates(inplace=True)

    t1, v1, test1 = partition_data(seq1_df)
    t2, v2, test2 = partition_data(seq2_df)
    data_df = pd.concat([t1,t2,v1,v2,test1,test2])
    data_df.to_csv(os.path.join(outpath,'human_promoters_short.csv'), index=False)

    fname = 'GCF_000001405.38_GRCh38.p12_genomic.fna'
    data = process_fasta(os.path.join(path,fname), 10000, 2000, filter_txt='NC_')
    df = pd.DataFrame(data, columns=['Sequence'])
    df['Source'] = 'NCBI Human'
    df.to_csv(os.path.join(outpath,'human_genome_data.csv'), index=False)
    init_voc = get_init_vocab()
    np.save(os.path.join(outpath,'human_vocab_5mer.npy'), init_voc.vocab.itos)

def create_genome_full_enc():
    tok = Tokenizer(GenomicTokenizer, n_cpus=32, pre_rules=[], post_rules=[], special_cases=['xxpad'])
    config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=0, qrnn=False, output_p=0.25, 
                          hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15, tie_weights=True, out_bias=True)
    drop_mult = 0.2
    lr = 2e-2
    count = 1
    df_chunks = pd.read_csv(os.path.join(outpath,'human_genome_data.csv'), chunksize=100000)
    for df in df_chunks:
        
        if count == 1:
            voc = np.load(os.path.join(outpath,'human_vocab_5mer.npy'))
            model_vocab = GenomicVocab(voc)
            
            cut = int(len(df)*0.8) + 1
            train_df = df[:cut]
            valid_df = df[cut:]
            data = GenomicTextLMDataBunch.from_df(path, train_df, valid_df, bs=400, tokenizer=tok, vocab=model_vocab,
                                        chunksize=10000, text_cols=0, label_cols=1)
            learn = get_model_LM(data, drop_mult, config)
            
        else:
            data = GenomicTextLMDataBunch.from_df(path, df, valid_df, bs=400, tokenizer=tok, 
                                        chunksize=10000, text_cols=0, label_cols=1, vocab=model_vocab)
            
        learn.data = data
        lr_iter = lr/(1.5**count)
        print(f'Learning Rate: {lr_iter}')
        learn.fit_one_cycle(1, lr_iter, moms=(0.8,0.7), pct_start=0.4)
        count += 1
    learn.save(os.path.join(outpath,'human_genome_full'))
    learn.save_encoder(os.path.join(outpath,'human_genome_full_enc'))

def create_genome_full_enc3():
    tok = Tokenizer(GenomicTokenizer, n_cpus=32, pre_rules=[], post_rules=[], special_cases=['xxpad'])
    lr = 5e-3
    count = 1
    df_chunks = pd.read_csv(os.path.join(outpath,'human_genome_data.csv'), chunksize=100000)
    for df in df_chunks:
        
        if count == 1:
            voc = np.load(os.path.join(outpath,'human_vocab_5mer.npy'))
            model_vocab = GenomicVocab(voc)
            
            cut = int(len(df)*0.8) + 1
            train_df = df[:cut]
            valid_df = df[cut:]
            data = GenomicTextLMDataBunch.from_df(path, train_df, valid_df, bs=400, tokenizer=tok, 
                                        chunksize=10000, text_cols=0, label_cols=1)
            
        else:
            data = GenomicTextLMDataBunch.from_df(path, df, valid_df, bs=400, tokenizer=tok, 
                                        chunksize=10000, text_cols=0, label_cols=1, vocab=model_vocab)
            
        config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=0, qrnn=False, output_p=0.25, 
                          hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15, tie_weights=True, out_bias=True)
        drop_mult = 0.2
        learn = get_model_LM(data, drop_mult, config)
        learn.data = data
        lr_iter = lr/(1.5**count)
        print(f'Learning Rate: {lr_iter}')
        learn.fit_one_cycle(1, lr_iter, moms=(0.8,0.7), pct_start=0.4)
        count += 1
    learn.save(os.path.join(outpath,'human_genome_full3'))
    learn.save_encoder(os.path.join(outpath,'human_genome_full_enc3'))

def lm_5_mer_Stride_1():
    print('lm5')
    df_iter = pd.read_csv(os.path.join(outpath,'human_genome_data_fa.csv'), chunksize=220000)
    df = next(df_iter)
    df_val = df[:20000]
    tok = Tokenizer(partial(GenomicTokenizer, ngram=5, stride=1), n_cpus=1, pre_rules=[], post_rules=[], special_cases=['xxpad'])
    data = GenomicTextLMDataBunch.from_df(path, df[20000:], df_val, bs=800, tokenizer=tok, 
                              chunksize=10000, text_cols=0, label_cols=1, max_vocab=80000)
    np.save(os.path.join(outpath,'human_vocab_5m1s.npy'), data.vocab.itos)
    voc = np.load(os.path.join(outpath,'human_vocab_5m1s.npy'))
    model_vocab = GenomicVocab(voc)
    data = GenomicTextLMDataBunch.from_df(path, df[20000:], df_val, bs=800, tokenizer=tok, vocab=model_vocab, max_vocab=80000,
                              chunksize=10000, text_cols=0, label_cols=1)
    config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=0, qrnn=False, output_p=0.25, 
                          hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15, tie_weights=True, out_bias=True)
    drop_mult=0.3
    learn = get_model_LM(data, drop_mult, config)
    learn = learn.to_fp16(dynamic=True)
    learn.lr_find()
    learn.recorder.plot(suggestion=True,return_fig=True)
    min_grad_lr = learn.recorder.min_grad_lr
    learn.fit_one_cycle(2, min_grad_lr, moms=(0.8,0.7))
    learn.save(os.path.join(outpath,'human_5m1s'))
    learn.save_encoder(os.path.join(outpath,'human_5m1s_enc'))

    learn.load(os.path.join(outpath,'human_5m1s'))
    count = 0
    lr = 5e-3
    for df in df_iter:
        data = GenomicTextLMDataBunch.from_df(path, df, df_val, bs=800, tokenizer=tok, vocab=model_vocab, max_vocab=80000,
                                    chunksize=10000, text_cols=0, label_cols=1)
        learn.data = data
        lr_iter = lr/1.5**count
        print(f'Learning Rate: {lr_iter}')
        learn.fit_one_cycle(2, lr, moms=(0.8,0.7))
        count += 1
    
    learn.save(os.path.join(outpath,'human_5m1s2'))
    learn.save_encoder(os.path.join(outpath,'human_5m1s_enc2'))

def lm_3_mer_Stride_1():
    print('lm3')
    df_iter = pd.read_csv(os.path.join(outpath,'human_genome_data_fa.csv'), chunksize=180000)
    df = next(df_iter)
    df_val = df[:20000]
    tok = Tokenizer(partial(GenomicTokenizer, ngram=3, stride=1), n_cpus=8, pre_rules=[], post_rules=[], special_cases=['xxpad'])
    data = GenomicTextLMDataBunch.from_df(path, df[20000:], df_val, bs=800, tokenizer=tok, 
                              chunksize=10000, text_cols=0, label_cols=1, max_vocab=80000)
    np.save(os.path.join(outpath,'human_vocab_3m1s.npy'), data.vocab.itos)
    voc = np.load(os.path.join(outpath,'human_vocab_3m1s.npy'))
    model_vocab = GenomicVocab(voc)
    data = GenomicTextLMDataBunch.from_df(path, df[20000:40000], df_val, bs=800, tokenizer=tok, vocab=model_vocab, max_vocab=80000,
                              chunksize=10000, text_cols=0, label_cols=1)
    config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=0, qrnn=False, output_p=0.25, 
                          hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15, tie_weights=True, out_bias=True)
    drop_mult=0.3
    learn = get_model_LM(data, drop_mult, config)
    learn = learn.to_fp16(dynamic=True)
    learn.lr_find()
    learn.fit_one_cycle(2, 5e-3, moms=(0.8, 0.7))
    learn.save(os.path.join(outpath,'human_3m1s'))
    learn.save_encoder(os.path.join(outpath,'human_3m1s_enc'))
    learn.load(os.path.join(outpath,'human_3m1s'))
    voc = np.load(os.path.join(outpath,'human_vocab_3m1s.npy'))
    model_vocab = GenomicVocab(voc)
    count = 0
    lr = 5e-3
    for df in df_iter:
        data = GenomicTextLMDataBunch.from_df(path, df, df_val, bs=800, tokenizer=tok, vocab=model_vocab, max_vocab=80000,
                                    chunksize=20000, text_cols=0, label_cols=1)
        learn.data = data
        lr_iter = lr/1.5**count
        print(f'Learning Rate: {lr_iter}')
        learn.fit_one_cycle(1, lr, moms=(0.8,0.7))
        count += 1
    learn.save(os.path.join(outpath,'human_3m1s2'))
    learn.save_encoder(os.path.join(outpath,'human_3m1s_enc2'))
    learn.load(os.path.join(outpath,'human_3m1s2'))
    learn = learn.to_fp32()
    learn.save(os.path.join(outpath,'human_3m1s2_fp32'))

if __name__ == '__main__':
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    gpu_id = get_gpu_id_max_memory(ACCEPTABLE_AVAILABLE_MEMORY)
    if gpu_id == -1:
        print("Can't run on GPUs now because of lacking available memory!")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        #main()
        #create_genome_full_enc3()
        lm_3_mer_Stride_1()
        lm_5_mer_Stride_1()