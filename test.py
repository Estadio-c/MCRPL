from utils import  *
from datasets_all import *
import os
import time
import torch
from torch import nn
import pandas as pd
import numpy as np
import torch.optim as optim
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import copy
import argparse

def train_model_one(model,seq_dataloader_test_A,args,len_):
    with torch.no_grad():
        
        pred_list_all = None
        true_list = None
        for idx,(seq_a, target_a) in enumerate(seq_dataloader_test_A):
            if torch.cuda.is_available():
                seq_a=seq_a.to(args.device)
                target_a=target_a.to(args.device) 
            
            logits , prompt_a, prompt_b = model(idx,seq_a)

            if idx == 0:
                pred_list_all = logits
                true_list = target_a
            else:
                pred_list_all = torch.cat([pred_list_all,logits],dim=0)
                true_list = torch.cat([true_list,target_a],dim=0)
                
        get_full_sort_score(0, pred_list_all, true_list)
           


    # return best_model_wts
if __name__=='__main__':
    seed = 608
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='HVIDEO')
    # parser.add_argument('--train_dir', required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--A_size', default=3389, type=int)
    parser.add_argument('--B_size', default=16431, type=int)
    parser.add_argument('--all_size', default=(16431+3389), type=int)

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--optimizer_all', default=True, type=bool)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--maxlen', default=15, type=int)  #50
    parser.add_argument('--min_len', default=0, type=int)
    parser.add_argument('--lr_decline', default=False, type=bool)
    parser.add_argument('--Lambda', default=0.002, type=float)

    parser.add_argument('--d_k', default=100, type=int)
    parser.add_argument('--n_heads', default=1, type=int)
    parser.add_argument('--d_v', default=100, type=int)
    parser.add_argument('--d_ff', default=2048, type=int)
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--Strategy', default='default', type=str)
    

    parser.add_argument('--hidden_units', default=100, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=201, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.25, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default=torch.device('cuda:0'))
    parser.add_argument('--state_dict_path', default=None, type=str)
    args = parser.parse_args()
    dataset=TVdatasets_all('finalcontruth_info/Elist.txt','finalcontruth_info/Vlist.txt','finalcontruth_info/A_weak.txt','finalcontruth_info/B_strong.txt',args,domain='all',offsets=args.A_size)
    # usernum=dataset.usernum
    usernum=None


    dataset_test=TVdatasets_all('finalcontruth_info/Elist.txt','finalcontruth_info/Vlist.txt','finalcontruth_info/A_weak_test.txt','finalcontruth_info/B_strong_test.txt',args,domain='A',offsets=args.A_size)    

    bce_criterion = torch.nn.CrossEntropyLoss().to(args.device)

    data_loader_train_A = DataLoader(dataset,batch_size=128,shuffle=True)
    data_loader_test_A = DataLoader(dataset_test,batch_size=128,shuffle=True)


   
    from model import mcrpl
    model= mcrpl(args,args.all_size).to(args.device)
    

    path='model/mcrpl_one.pth'
    model.load_state_dict(torch.load(path,map_location=torch.device(args.device)),strict=False)

    print('成功加载模型')

    train_model_one(model,data_loader_test_A,args,len(dataset_test))



