from utils import evaluate , get_eval
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

def train_model(model,seq_dataloader_test,args,len_):
   
    model.eval()
    with torch.no_grad():
        r3_b = 0
        m3_b = 0
        r5_b = 0
        m5_b = 0
        r10_b = 0
        m10_b = 0
        m20_b = 0
        for idx,(seq,target) in enumerate(seq_dataloader_test):
            if torch.cuda.is_available():
                seq=seq.to(args.device)
                target=target.to(args.device) 
            logits_B=model(idx,seq)
            recall,mrr = get_eval(logits_B, target, [5,10,20])
            r5_b += recall[0]
            m5_b += mrr[0]
            r10_b += recall[1]
            m10_b += mrr[1]
            r20_b += recall[2]
            m20_b += mrr[2]
        print('Recall3_b: {:.5f}; Mrr3: {:.5f}'.format(r3_b/len_,m3_b/len_))
        print('Recall5_b: {:.5f}; Mrr5: {:.5f}'.format(r5_b/len_,m5_b/len_))
        print('Recall10_b: {:.5f}; Mrr10: {:.5f}'.format(r10_b/len_,m10_b/len_))
        print('Mrr20: {:.5f}'.format(m20_b/len_))
          

if __name__=='__main__':
    # seed = 608
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m')
    # parser.add_argument('--train_dir', required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--A_size', default=3389, type=int)
    parser.add_argument('--B_size', default=16431, type=int)
    parser.add_argument('--all_size', default=(16431+3389), type=int)

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--optimizer_all', default=True, type=bool)
    parser.add_argument('--epoch', default=60, type=int)
    parser.add_argument('--maxlen', default=15, type=int)  #50
    parser.add_argument('--min_len', default=0, type=int)
    parser.add_argument('--lr_decline', default=False, type=bool)


    parser.add_argument('--d_k', default=100, type=int)
    parser.add_argument('--n_heads', default=1, type=int)
    parser.add_argument('--d_v', default=100, type=int)
    parser.add_argument('--d_ff', default=2048, type=int)
    parser.add_argument('--n_layers', default=1, type=int)


    parser.add_argument('--hidden_units', default=100, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=201, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default=torch.device('cuda:0'))
    parser.add_argument('--state_dict_path', default=None, type=str)
    args = parser.parse_args()

    dataset_test=TVdatasets_all('finalcontruth_info/Elist.txt','finalcontruth_info/Vlist.txt','finalcontruth_info/A_weak_test.txt','finalcontruth_info/B_strong_test.txt',args,domain='A',offsets=args.A_size)    

    bce_criterion = torch.nn.CrossEntropyLoss().to(args.device)

    data_loader_test_A = DataLoader(dataset_test,batch_size=128,shuffle=True)

  
    from model import mcrpl

    model= mcrpl(args,args.all_size).to(args.device)


    path='model/SASrec.pth'
    model_state=torch.load(path,map_location=torch.device(args.device))

    model.load_state_dict(torch.load(path,map_location=torch.device(args.device)),strict=False)
    
    train_model(model,data_loader_test_A,args,len(dataset_test))
