import os
#import cudf as pd
#import cupy as np
import pandas as pd 
import numpy as np

class Dataset():
    def __init__(self,opt):
        self.path = opt.dataset_path#'/home/anhkhoa/Vu_working/NLP/Human-Short-Seq/data/human_promoters_short.csv'

        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.batch_size = opt.batch_size

        self.create_train_validation_test()

    def create_train_validation_test(self):
        classification_df = pd.read_csv(self.path)
        self.train_df = classification_df[classification_df.set == 'train']
        self.valid_df = classification_df[classification_df.set == 'valid']
        self.test_df = classification_df[classification_df.set == 'test']