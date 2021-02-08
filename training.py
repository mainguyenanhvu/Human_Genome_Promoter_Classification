import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from training_helper import *
from utils import *
from dataset_admin import *

model_path = 'model_files'
history_path = 'history'
path = '/home/anhkhoa/Vu_working/NLP/Human-Short-Seq/data'
extent_name = ''

ACCEPTABLE_AVAILABLE_MEMORY = 1024
OPTIMIZER="Adam"
LOSS="categorical_crossentropy"
METRICS=["accuracy"]
EPOCHS = 1000



def naive_model(dataset):
    tok = Tokenizer(GenomicTokenizer, n_cpus=32, pre_rules=[], post_rules=[], special_cases=['xxpad'])
    data_clas = GenomicTextClasDataBunch.from_df(path, dataset.train_df, dataset.valid_df, test_df=dataset.test_df, tokenizer=tok, 
                                            text_cols='Sequence', label_cols='Promoter', bs=400)

    clas_config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=0, qrnn=False, output_p=0.4, 
                       hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
    drop_mult = 0.5
    learn = get_model_clas(data_clas, drop_mult, clas_config)
    return learn, data_clas

def genomic_pretrained_model(dataset):
    voc = np.load(os.path.join(path,'human_vocab_5mer.npy'))
    model_vocab = GenomicVocab(voc)
    tok = Tokenizer(GenomicTokenizer, n_cpus=32, pre_rules=[], post_rules=[], special_cases=['xxpad'])
    data_clas = GenomicTextClasDataBunch.from_df(path,  dataset.train_df, dataset.valid_df, tokenizer=tok, vocab=model_vocab,
                                                text_cols='Sequence', label_cols='Promoter', bs=400)
    clas_config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=0, qrnn=False, output_p=0.4, 
                       hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
    drop_mult = 0.6
    learn = get_model_clas(data_clas, drop_mult, clas_config)
    return learn, data_clas

def genomic_pretrained_fine_tune_model(dataset):
    voc = np.load(os.path.join(path,'human_vocab_5mer.npy'))
    model_vocab = GenomicVocab(voc)
    tok = Tokenizer(GenomicTokenizer, n_cpus=32, pre_rules=[], post_rules=[], special_cases=['xxpad'])
    # data_clas = GenomicTextLMDataBunch.from_df(path, dataset.train_df, dataset.valid_df, bs=400, tokenizer=tok, 
    #                           chunksize=10000, text_cols='Sequence', label_cols='Promoter', vocab=model_vocab)
    # config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=0, qrnn=False, output_p=0.25, 
    #                       hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15, tie_weights=True, out_bias=True)
    # drop_mult = 0.25
    # learn = get_model_LM(data_clas, drop_mult, config)
    # learn.load(os.path.join(path,'human_genome_full3'))
    # learn.fit_one_cycle(10, 2e-3, moms=(0.8,0.7))
    # learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
    # learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
    # learn.fit_one_cycle(10, 8e-4, moms=(0.8,0.7))
    # learn.save('human_LM_short_finetune')
    # learn.save_encoder('human_LM_short_finetune_enc')

    data_clas = GenomicTextClasDataBunch.from_df(path, dataset.train_df, dataset.valid_df, tokenizer=tok, vocab=model_vocab,
                                            text_cols='Sequence', label_cols='Promoter', bs=400)
    clas_config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=0, qrnn=False, output_p=0.4, 
                       hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
    drop_mult = 0.5
    learn = get_model_clas(data_clas, drop_mult, clas_config)
    learn.load_encoder('human_LM_short_finetune_enc')
    return learn, data_clas

def model_5m1s(dataset):
    voc = np.load(os.path.join(path,'human_vocab_5m1s.npy'))
    model_vocab = GenomicVocab(voc)
    tok = Tokenizer(partial(GenomicTokenizer, ngram=5, stride=1), n_cpus=32, pre_rules=[], post_rules=[], special_cases=['xxpad'])
    data = GenomicTextLMDataBunch.from_df(path, dataset.train_df, dataset.valid_df, bs=800, tokenizer=tok, 
                              chunksize=10000, text_cols='Sequence', label_cols='Promoter', vocab=model_vocab)
    config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=0, qrnn=False, output_p=0.25, 
                          hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15, tie_weights=True, out_bias=True)
    drop_mult = 0.25                              
    learn = get_model_LM(data, drop_mult, config)
    learn = learn.to_fp16(dynamic=True)
    learn.load('human_5m1s2')
    learn.lr_find()
    learn.fit_one_cycle(10, 1e-2, moms=(0.8,0.7))
    learn.save('human_LM_short_5m1s')
    learn.save_encoder('human_LM_short_5m1s_enc')
    tok = Tokenizer(partial(GenomicTokenizer, ngram=5, stride=1), n_cpus=32, pre_rules=[], post_rules=[], special_cases=['xxpad'])
    data_clas = GenomicTextClasDataBunch.from_df(path, dataset.train_df, dataset.valid_df, tokenizer=tok, vocab=model_vocab,
                                                text_cols='Sequence', label_cols='Promoter', bs=400)
    clas_config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=0, qrnn=False, output_p=0.4, 
                       hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
    drop_mult = 0.5
    learn = get_model_clas(data_clas, drop_mult, clas_config)
    learn.load_encoder('human_LM_short_5m1s_enc')
    learn = learn.to_fp16(dynamic=True)
    learn.freeze()
    return learn, data_clas

def model_3m1s(dataset):
    voc = np.load(os.path.join(path,'human_vocab_3m1s.npy'))
    model_vocab = GenomicVocab(voc)
    tok = Tokenizer(partial(GenomicTokenizer, ngram=3, stride=1), n_cpus=32, pre_rules=[], post_rules=[], special_cases=['xxpad'])
    data = GenomicTextLMDataBunch.from_df(path, dataset.train_df, dataset.valid_df, bs=800, tokenizer=tok, 
                              chunksize=30000, text_cols='Sequence', label_cols='Promoter', vocab=model_vocab)
    config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=0, qrnn=False, output_p=0.25, 
                          hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15, tie_weights=True, out_bias=True)
    drop_mult = 0.25
    learn = get_model_LM(data, drop_mult, config)
    learn = learn.to_fp16(dynamic=True)
    learn.load('human_3m1s2')
    learn.lr_find()
    learn.fit_one_cycle(10, 1e-2, moms=(0.8,0.7))
    learn.save('human_LM_short_3m1s')
    learn.save_encoder('human_LM_short_3m1s_enc')

    tok = Tokenizer(partial(GenomicTokenizer, ngram=3, stride=1), n_cpus=32, pre_rules=[], post_rules=[], special_cases=['xxpad'])
    data_clas = GenomicTextClasDataBunch.from_df(path, dataset.train_df, dataset.valid_df, tokenizer=tok, vocab=model_vocab,
                                            text_cols='Sequence', label_cols='Promoter', bs=400, chunksize=50000)
    clas_config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=0, qrnn=False, output_p=0.4, 
                       hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
    drop_mult = 0.5
    learn = get_model_clas(data_clas, drop_mult, clas_config)
    learn.load_encoder('human_LM_short_3m1s_enc')
    learn = learn.to_fp16(dynamic=True)
    return learn, data_clas

def main(options):
    mkdir_if_missing(options.save_path)
    #mkdir_if_missing(os.path.join(options.save_path,model_path))
    mkdir_if_missing(os.path.join(options.save_path,history_path))

    dataset = Dataset(options)
    
    learn, data_clas = model_dic[options.model_name](dataset)
    
    print(learn.unfreeze())
    print(learn.lr_find())
    graph = learn.recorder.plot(suggestion=True,return_fig=True)
    graph.savefig(os.path.join(options.save_path,history_path+'/'+options.model_name+'_checkloss.png'))

    min_grad_lr = learn.recorder.min_grad_lr
    learn.fit_one_cycle(EPOCHS, min_grad_lr, moms=(0.8,0.7))

    graph = learn.recorder.plot_losses(return_fig=True)
    graph.savefig(os.path.join(options.save_path,history_path+'/'+options.model_name+'_loss.png'))
    learn.save(options.model_name)
    learn.data = data_clas
    get_scores(os.path.join(options.save_path,history_path+'/'+options.model_name+'_test.csv'),learn)


if __name__ == '__main__':
    model_dic = {'hum_prom_short': naive_model,
    'human_short_human_pretrain': genomic_pretrained_model, 
    'human_short_human_pretrain_finetune':genomic_pretrained_fine_tune_model, 
    'human_short_human_pretrain_finetune_3m1s': model_3m1s,
    'human_short_human_pretrain_finetune_5m1s': model_5m1s}
    options, args = create_training_opt_parser()
    gpu_id = get_gpu_id_max_memory(ACCEPTABLE_AVAILABLE_MEMORY)
    if gpu_id == -1:
        print("Can't run on GPUs now because of lacking available memory!")
    else:
        #os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        main(options)