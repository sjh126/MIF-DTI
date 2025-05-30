# -*- coding:utf-8 -*-


class hyperparameter():
    def __init__(self):
        self.Learning_rate = 1e-4
        self.Epoch = 200
        self.Batch_size = 128
        self.Patience = 50
        self.decay_interval = 10
        self.lr_decay = 0.5
        self.weight_decay = 1e-4
        self.embed_dim = 64
        self.protein_kernel = [4, 8, 12]
        self.drug_kernel = [4, 6, 8]
        self.conv = 50
        self.char_dim = 64
        self.loss_epsilon = 1