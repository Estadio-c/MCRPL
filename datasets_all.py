from itertools import count
from turtle import pen
import numpy as np
import  pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader,Dataset
import random
class TVdatasets_all(Dataset):
    def __init__(self,Elist,Vlist,train_A,train_B,args,domain,offsets):
        print('Loading training set data......')
        self.Elist = Elist
        self.Vlist = Vlist
        self.train_A = train_A
        self.train_B = train_B
        self.maxlen = args.maxlen
        # self.flag_a = flag_a
        # self.flag_b = flag_b
        self.min_len = args.min_len
        self.domain=domain
        self.offsets=offsets
 

        if self.domain == 'A':
            User_a,User_target_a = self.getdict(self.Elist,self.train_A,0,0)
            # self.finally_user_train_a,self.finally_user_target_a = self.sample_seq(User_a,User_target_a,self.maxlen)
        
        elif self.domain == 'B':

            User_a,User_target_a = self.getdict(self.Vlist,self.train_B,0,offsets)

        else : 
            User_a,User_target_a = self.getdict(self.Elist,self.train_A,0,0)
            # print(max(list(User_a.keys())))
            start_id=len(User_a)
            User_b,User_target_b = self.getdict(self.Vlist,self.train_B,start_id,offsets)
            # print(min(list(User_b.keys())))

            User_a,User_target_a= { **User_a, **User_b }   ,  { **User_target_a, **User_target_b }


        self.finally_user_train_a,self.finally_user_target_a = self.sample_seq(User_a,User_target_a,self.maxlen)    

    
    def __getitem__(self, index):

        return self.finally_user_train_a[index],self.finally_user_target_a[index]

    def __len__(self):

        return len(list(self.finally_user_train_a.keys()))

    def getdict(self,E_list,train,start_id,offsets):
        User_all = defaultdict(list)
        User_target = defaultdict(list)
        line_id=start_id
        with open(train, 'r') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                # Index starts from 1 in order to remove users
                for item in line[1:]:         
                    # User_all[line[0]].append(item)
                    User_all[line_id].append(item)
                line_id+=1   
            f.close()
        for user in list(User_all.keys()):
            User_target[user]=User_all[user][-1]
            User_all[user]=User_all[user][:-1]
            if len(User_all[user]) < self.min_len:
                del User_all[user]
                del User_target[user]
        def id_dict(fname):                                       
            itemdict = {}
            with open(fname,'r') as f:
                items =  f.readlines()                 
            for item in items:
                item = item.strip().split('\t')
                itemdict[item[1]] = int(item[0])+1+offsets
            return itemdict
        item_A = id_dict(E_list)
        User_a= defaultdict(list)
        for k,v in User_all.items():
            for item in User_all[k]:
                User_a[k].append(item_A[item])
            User_target[k]=np.array(item_A[User_target[k]])
        return User_a,User_target
    def sample_seq(self,user_train_a,user_target,maxlen):
        new_user_id=0
        finally_user_train_a={}
        finally_user_target_a={}
        for user in user_train_a.keys():
            seq = np.zeros([maxlen], dtype=np.int32)
            idx=maxlen-1
            for i in reversed(user_train_a[user]):
                seq[idx]=i
                idx-=1
                if idx==-1:
                    break
            finally_user_train_a[new_user_id] = seq
            finally_user_target_a[new_user_id] = user_target[user]
            new_user_id+=1 
        return finally_user_train_a,finally_user_target_a 
    

if __name__ == '__main__':

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
    parser.add_argument('--maxlen', default=50, type=int)  #50
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
    parser.add_argument('--num_heads', default=2, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    # parser.add_argument('--device', default=torch.device('cuda:0'))
    parser.add_argument('--state_dict_path', default=None, type=str)
    args = parser.parse_args()

    dataset=TVdatasets_all('finalcontruth_info/Elist.txt','finalcontruth_info/Vlist.txt','finalcontruth_info/A_weak.txt','finalcontruth_info/B_strong.txt',args,domain='all',offsets=args.A_size)
    # print(len(dataset))
    # print(dataset[5])
    
    # dataset=TVdatasets_A('finalcontruth_info/Elist.txt','finalcontruth_info/traindata_sess.txt',args,'E')
    # # print(dataset[0])
    # data_loader_train_A = DataLoader(dataset,batch_size=3,shuffle=False)
    # data_loader_train_A=iter(data_loader_train_A)
    # print(next(data_loader_train_A))
