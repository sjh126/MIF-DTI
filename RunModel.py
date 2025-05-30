# -*- coding:utf-8 -*-

import os
import random
import joblib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from torch.utils.data import DataLoader
from tqdm import tqdm

from config import hyperparameter
from model import MIFDTI
from utils.DataPrepare import get_kfold_data, shuffle_dataset
from utils.DataSetsFunction import CustomDataSet, collate_fn
from utils.EarlyStoping import EarlyStopping
from LossFunction import CELoss, PolyLoss
from utils.TestModel import test_model
from utils.ShowResult import show_result
from utils import protein_init, ligand_init, ProteinMoleculeDataset
import torch_geometric.loader as pyg_loader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_MIF_model(SEED, DATASET, MODEL, K_Fold, LOSS, device):
    '''set random seed'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''init hyperparameters'''
    hp = hyperparameter()

    '''load dataset from text file'''
    print("Train in " + DATASET)
    print("load data")
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")

    '''set loss function weight'''
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(device)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(device)
    else:
        weight_loss = None

    '''shuffle data'''
    print("data shuffle")
    data_list = shuffle_dataset(data_list, SEED)

    '''split dataset to train&validation set and test set'''
    split_pos = len(data_list) - int(len(data_list) * 0.2)
    train_data_list = data_list[0:split_pos]
    test_data_list = data_list[split_pos:-1]
    print('Number of Train&Val set: {}'.format(len(train_data_list)))
    print('Number of Test set: {}'.format(len(test_data_list)))

    '''Data Preparation'''
    protein_path = f'./DataSets/Preprocessed/{DATASET}-protein.pkl'
    if os.path.exists(protein_path):
        print('Loading Protein Graph data...')
        protein_dict = joblib.load(protein_path)
    else:
        print('Initialising Protein Sequence to Protein Graph...')
        protein_seqs = list(set([item.split(' ')[-2] for item in data_list]))
        protein_dict = protein_init(protein_seqs)
        joblib.dump(protein_dict,protein_path)

    ligand_path = f'./DataSets/Preprocessed/{DATASET}-ligand-hi.pkl'
    if os.path.exists(ligand_path):
        print('Loading Ligand Graph data...')
        ligand_dict = joblib.load(ligand_path)
    else:
        print('Initialising Ligand SMILES to Ligand Graph...')
        ligand_smiles = list(set([item.split(' ')[-3] for item in data_list]))
        ligand_dict = ligand_init(ligand_smiles, mode='BRICS')
        joblib.dump(ligand_dict,ligand_path)

    torch.cuda.empty_cache()

    '''metrics'''
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)

        train_dataset, valid_dataset = get_kfold_data(i_fold, train_data_list, k=K_Fold)
        train_dataset = ProteinMoleculeDataset(train_dataset, ligand_dict, protein_dict, device=device)
        valid_dataset = ProteinMoleculeDataset(valid_dataset, ligand_dict, protein_dict, device=device)
        test_dataset = ProteinMoleculeDataset(test_data_list, ligand_dict, protein_dict, device=device)
        train_size = len(train_dataset)

        train_loader = pyg_loader.DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)
        valid_loader = pyg_loader.DataLoader(valid_dataset, batch_size=hp.Batch_size,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)
        test_loader = pyg_loader.DataLoader(test_dataset, batch_size=hp.Batch_size,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)
                                    
        """ create model"""
        model = MODEL(device=device)

        """Initialize weights"""
        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        """create optimizer and scheduler"""
        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)

        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
                                                step_size_up=train_size // hp.Batch_size)
        # if LOSS == 'PolyLoss':
        #     Loss = PolyLoss(weight_loss=weight_loss,
        #                     DEVICE=device, epsilon=hp.loss_epsilon)
        # else:
        Loss = CELoss(weight_CE=weight_loss, DEVICE=device)

        """Output files"""
        save_path = "./" + DATASET + "/{}".format(i_fold+1)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_results = save_path + '/' + 'The_results_of_whole_dataset.txt'

        early_stopping = EarlyStopping(
            savepath=save_path, patience=hp.Patience, verbose=True, delta=0)

        """Start training."""
        print('Training...')
        for epoch in range(1, hp.Epoch + 1):
            if early_stopping.early_stop == True:
                break

            """train"""
            train_losses_in_epoch = []
            model.train()
            for data in train_loader:
                optimizer.zero_grad()

                data = data.to(device)
                predicted_y= model(data)
                train_loss = Loss(predicted_y, data.cls_y)
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss

            """valid"""
            valid_losses_in_epoch = []
            model.eval()
            Y, P, S = [], [], []
            with torch.no_grad():
                for data in valid_loader:

                    data = data.to(device)
                    valid_scores = model(data)
                    
                    valid_labels = data.cls_y
                    valid_loss = Loss(valid_scores, valid_labels)
                    valid_losses_in_epoch.append(valid_loss.item())
                    valid_labels = valid_labels.to('cpu').data.numpy()
                    valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
                    valid_predictions = np.argmax(valid_scores, axis=1)
                    valid_scores = valid_scores[:, 1]

                    Y.extend(valid_labels)
                    P.extend(valid_predictions)
                    S.extend(valid_scores)

            Precision_dev = precision_score(Y, P)
            Reacll_dev = recall_score(Y, P)
            Accuracy_dev = accuracy_score(Y, P)
            AUC_dev = roc_auc_score(Y, S)
            tpr, fpr, _ = precision_recall_curve(Y, S)
            PRC_dev = auc(fpr, tpr)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)

            epoch_len = len(str(hp.Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_AUC: {AUC_dev:.5f} ' +
                         f'valid_PRC: {PRC_dev:.5f} ' +
                         f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                         f'valid_Precision: {Precision_dev:.5f} ' +
                         f'valid_Reacll: {Reacll_dev:.5f} ')
            print(print_msg)

            '''save checkpoint and make decision when early stop'''
            early_stopping(Accuracy_dev, model, epoch)

        '''load best checkpoint'''
        model.load_state_dict(torch.load(early_stopping.savepath + f'/valid_best_checkpoint-{device}.pth', weights_only=True))

        '''test model'''
        trainset_test_stable_results, _, _, _, _, _ = test_model(
            model, train_loader, save_path, DATASET, Loss, device, dataset_class="Train", FOLD_NUM=1, MIF=True)
        validset_test_stable_results, _, _, _, _, _ = test_model(
            model, valid_loader, save_path, DATASET, Loss, device, dataset_class="Valid", FOLD_NUM=1, MIF=True)
        testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_loader, save_path, DATASET, Loss, device, dataset_class="Test", FOLD_NUM=1, MIF=True)
        AUC_List_stable.append(AUC_test)
        Accuracy_List_stable.append(Accuracy_test)
        AUPR_List_stable.append(PRC_test)
        Recall_List_stable.append(Recall_test)
        Precision_List_stable.append(Precision_test)
        with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:
            f.write("Test the stable model" + '\n')
            f.write(trainset_test_stable_results + '\n')
            f.write(validset_test_stable_results + '\n')
            f.write(testset_test_stable_results + '\n')

    show_result(DATASET, Accuracy_List_stable, Precision_List_stable,
                Recall_List_stable, AUC_List_stable, AUPR_List_stable, Ensemble=False)
    

def ensemble_run_MIF_model(SEED, DATASET, K_Fold, device):

    '''set random seed'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''init hyperparameters'''
    hp = hyperparameter()

    '''load dataset from text file'''
    assert DATASET in ["DrugBank", "BIOSNAP", "Davis"]
    print("Train in " + DATASET)
    print("load data")
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")

    '''set loss function weight'''
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(device)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(device)
    else:
        weight_loss = None

    '''shuffle data'''
    print("data shuffle")
    data_list = shuffle_dataset(data_list, SEED)

    '''split dataset to train&validation set and test set'''
    split_pos = len(data_list) - int(len(data_list) * 0.2)
    test_data_list = data_list[split_pos:-1]
    print('Number of Test set: {}'.format(len(test_data_list)))

    save_path = f"./{DATASET}/ensemble"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    '''Data Preparation'''
    protein_path = f'./DataSets/Preprocessed/{DATASET}-protein.pkl'
    if os.path.exists(protein_path):
        print('Loading Protein Graph data...')
        protein_dict = joblib.load(protein_path)
    else:
        print('Initialising Protein Sequence to Protein Graph...')
        protein_seqs = list(set([item.split(' ')[-2] for item in data_list]))
        protein_dict = protein_init(protein_seqs)
        joblib.dump(protein_dict,protein_path)

    ligand_path = f'./DataSets/Preprocessed/{DATASET}-ligand-hi.pkl'
    if os.path.exists(ligand_path):
        print('Loading Ligand Graph data...')
        ligand_dict = joblib.load(ligand_path)
    else:
        print('Initialising Ligand SMILES to Ligand Graph...')
        ligand_smiles = list(set([item.split(' ')[-3] for item in data_list]))
        ligand_dict = ligand_init(ligand_smiles, mode='BRICS')
        joblib.dump(ligand_dict,ligand_path)

    torch.cuda.empty_cache()  
    
          
    

    test_dataset = ProteinMoleculeDataset(test_data_list, ligand_dict, protein_dict, device=device)
    
    # test_dataset_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
    #                                  collate_fn=collate_fn, drop_last=True)
    test_dataset_loader = pyg_loader.DataLoader(test_dataset, batch_size=1,  shuffle=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'], drop_last=True)

    model = []
    for i in range(K_Fold):
        model.append(MIFDTI().to(device))
        '''MIF-DTI K-Fold train process is necessary'''
        try:
            model[i].load_state_dict(torch.load(
                f'./{DATASET}/{i+1}' + f'/valid_best_checkpoint-{device}.pth', map_location=torch.device(device)))   #加载对应权重
        except FileNotFoundError as e:
            print('-'* 25 + 'ERROR' + '-'*25)
            error_msg = 'Load pretrained model error: \n' + \
                        str(e) + \
                        '\n' + 'MIFDTI K-Fold train process is necessary'
            print(error_msg)
            print('-'* 55)
            exit(1)

    Loss = PolyLoss(weight_loss=weight_loss,
                    DEVICE=device, epsilon=hp.loss_epsilon)

#   testdataset_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
#       model, test_dataset_loader, save_path, DATASET, Loss, device, dataset_class="Test", save=True, FOLD_NUM=K_Fold)
    
    testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_dataset_loader, save_path, DATASET, Loss, device, dataset_class="Test", FOLD_NUM=K_Fold, MIF=True)
    
    show_result(DATASET, Accuracy_test, Precision_test,
                Recall_test, AUC_test, PRC_test, Ensemble=True)
    