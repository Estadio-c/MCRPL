from utils import  get_eval
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
class DiffLoss(nn.Module):         

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss




def train_model_all(model,seq_dataloader_train,seq_dataloader_test,optimizer_list,bce_criterion,num_epochs,args,len_):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)

        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            if phase == 'train':
                # user_list=[]
                for idx,(seq , target) in enumerate(seq_dataloader_train):
                    # with torch.set_grad_enabled(phase == 'train'):
                    if torch.cuda.is_available():
                        seq=seq.to(args.device)
                        target=target.to(args.device)
                    logits , prompt_a, prompt_b  = model(idx,seq)
                    optimizer_list[0].zero_grad()
                
                    loss = bce_criterion(logits, (target-1))
                    # for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                    loss.backward()
                    
                    optimizer.step() 
                    if len(optimizer_list) == 2:
                        lr_scheduler.step()    
            # if phase == 'val' and ((epoch+1 )% 20 == 0) :
            if phase == 'val' :
                with torch.no_grad():
                    r3_b=0
                    m3_b=0
                    r5_b = 0
                    m5_b = 0
                    r10_b = 0
                    m10_b = 0
                    m20_b = 0
                    for idx,(seq,target) in enumerate(seq_dataloader_test):
                        if torch.cuda.is_available():
                            seq=seq.to(args.device)
                            target=target.to(args.device) 
                     
                        logits , prompt_a, prompt_b = model(idx,seq)
                        recall,mrr = get_eval(logits, target, [3,5,10,20])
                        r3_b += recall[0]
                        m3_b += mrr[0]
                        r5_b += recall[1]
                        m5_b += mrr[1]
                        r10_b += recall[2]
                        m10_b += mrr[2]
                        r20_b += recall[3]
                        m20_b += mrr[3]
                    # if (r10_b+r20_b)/2>best_cirtion :
                    #     best_cirtion=(r10_b+r20_b)/2
                    #     torch.save(model.state_dict(),'model/{},{},{},qianyi={},model_best.pth'.format(args.alpha,args.Beta,args.Gamma,str(args.qianyi)))
                    print('Recall3_b: {:.5f}; Mrr3: {:.5f}'.format(r3_b/len_,m3_b/len_))
                    print('Recall5_b: {:.5f}; Mrr5: {:.5f}'.format(r5_b/len_,m5_b/len_))
                    print('Recall10_b: {:.5f}; Mrr10: {:.5f}'.format(r10_b/len_,m10_b/len_))
                    print('Mrr20: {:.5f}'.format(m20_b/len_))
                # if (epoch+1 ) == 20:
                #     torch.save(model.state_dict(),'model/SAS.pth')
    torch.save(model.state_dict(),'model/mcrpl_one.pth')               


def train_model_one(model,seq_dataloader_train_A,seq_dataloader_test_A,optimizer_list,bce_criterion,num_epochs,args,len_):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)

        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            if phase == 'train':
                # user_list=[]
                for idx,(seq_a, target_a) in enumerate(seq_dataloader_train_A):
                    # with torch.set_grad_enabled(phase == 'train'):
                    if torch.cuda.is_available():
                        seq_a=seq_a.to(args.device)
                        target_a=target_a.to(args.device)
                  
                    # for_weak  = model.log2feats(seq_b)
                    
                    logits , prompt_a, prompt_b = model(idx,seq_a)
                    loss_diff = cal_diff(prompt_a,prompt_b)
                    optimizer_list[0].zero_grad()
                    # if phase == 'train':
                    loss = bce_criterion(logits, (target_a-1))
                    loss = loss + (args.Lambda) * loss_diff
                    # for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                    loss.backward()
                    optimizer.step() 
                    if len(optimizer_list) == 2:
                        lr_scheduler.step()    
            # if phase == 'val' and ((epoch+1 )% 20 == 0) :
            if phase == 'val' :
                with torch.no_grad():
                    r3_b=0
                    m3_b=0
                    r5_b = 0
                    m5_b = 0
                    r10_b = 0
                    m10_b = 0
                    m20_b = 0
                    for idx,(seq_a, target_a) in enumerate(seq_dataloader_test_A):
                        if torch.cuda.is_available():
                            seq_a=seq_a.to(args.device)
                            target_a=target_a.to(args.device) 
                     
                        logits , prompt_a, prompt_b = model(idx,seq_a)

                        recall,mrr = get_eval(logits, target_a, [3,5,10,20])
                        r3_b += recall[0]
                        m3_b += mrr[0]
                        r5_b += recall[1]
                        m5_b += mrr[1]
                        r10_b += recall[2]
                        m10_b += mrr[2]
                        r20_b += recall[3]
                        m20_b += mrr[3]
                    # if (r10_b+r20_b)/2>best_cirtion :
                    #     best_cirtion=(r10_b+r20_b)/2
                    #     torch.save(model.state_dict(),'model/{},{},{},qianyi={},model_best.pth'.format(args.alpha,args.Beta,args.Gamma,str(args.qianyi)))
                    print('Recall3_b: {:.5f}; Mrr3: {:.5f}'.format(r3_b/len_,m3_b/len_))
                    print('Recall5_b: {:.5f}; Mrr5: {:.5f}'.format(r5_b/len_,m5_b/len_))
                    print('Recall10_b: {:.5f}; Mrr10: {:.5f}'.format(r10_b/len_,m10_b/len_))
                    print('Mrr20: {:.5f}'.format(m20_b/len_))
                # if (epoch+1 ) == 20:
                #     torch.save(model.state_dict(),'model/SAS.pth')
    torch.save(model.state_dict(),'model/mcrpl_two.pth')               


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

    cal_diff=DiffLoss().to(args.device)

    dataset_test=TVdatasets_all('finalcontruth_info/Elist.txt','finalcontruth_info/Vlist.txt','finalcontruth_info/A_weak_test.txt','finalcontruth_info/B_strong_test.txt',args,domain='A',offsets=args.A_size)    

    bce_criterion = torch.nn.CrossEntropyLoss().to(args.device)

    data_loader_train_A = DataLoader(dataset,batch_size=128,shuffle=True)
    data_loader_test_A = DataLoader(dataset_test,batch_size=128,shuffle=True)

 
    # from SASRec import SASRec
    # model = SASRec(args,args.A_size).to(args.device)

   
    from model import mcrpl
    model= mcrpl(args,args.all_size).to(args.device)


    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers


    optimizer_list=[]
    lr = 0.001
    # # optimizer = optim.Adam(list(model.prompt_common.parameters())+list(model.prompt_user.parameters()), lr=lr,betas=(0.9, 0.98))  #betas=(0.9, 0.98)
    if args.lr_decline == False :
        optimizer = optim.Adam(model.parameters(), lr=lr,betas=(0.9, 0.98))
        optimizer_list.append(optimizer)
    else:
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters()],
                    "weight_decay": 0.01,
                }
            ]
        optimizer = AdamW(optimizer_grouped_parameters,
                    lr=1e-3, eps=1e-6)
        t_total = (len(dataset) // args.batch_size + 1) * args.epoch
        warmup_ratio = 0.1
        warmup_iters = int(t_total * warmup_ratio)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, warmup_iters, t_total)
        optimizer_list.append(optimizer)
        optimizer_list.append(lr_scheduler)
    
    train_model_all(model,data_loader_train_A,data_loader_test_A,optimizer_list,bce_criterion,args.epoch,args,len(dataset_test))




    path='model/mcrpl_one.pth'
    model.load_state_dict(torch.load(path,map_location=torch.device(args.device)),strict=False)


    # model.Freeze()

    dataset_a_train=TVdatasets_all('finalcontruth_info/Elist.txt','finalcontruth_info/Vlist.txt','finalcontruth_info/A_weak.txt','finalcontruth_info/B_strong.txt',args,domain='A',offsets=args.A_size)

    bce_criterion = torch.nn.CrossEntropyLoss().to(args.device)

    data_loader_train_A = DataLoader(dataset_a_train,batch_size=128,shuffle=True)


    optimizer_list=[optim.Adam(model.parameters(), lr=lr,betas=(0.9, 0.98))]

    train_model_one(model,data_loader_train_A,data_loader_test_A,optimizer_list,bce_criterion,30,args,len(dataset_test))



