import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import pandas as pd

def make_all_id(A_weak,A_weak_test,A_id_txt):
    movie_id=defaultdict(list)
    with open(A_weak,'r') as f:
        for line in f.readlines():
            line=line.strip().split('\t')
            for item in line[1:]:
                movie_id[item].append(1)   
    with open(A_weak_test,'r') as f:
        for line in f.readlines():
            line=line.strip().split('\t')
            for item in line[1:]:
                movie_id[item].append(1)    
    for item in list(movie_id.keys()):
        movie_id[item]=len(movie_id[item])
    paixu_movie=sorted(movie_id.items(),key=lambda  x:x[1],reverse=True)
    with open(A_id_txt, 'w+') as f:
        id=0
        for temp in paixu_movie:
            f.write(str(id))
            id+=1
            f.write('\t')
            f.write(str(temp[0]))
            f.write('\t')
            f.write(str(temp[1]))
            f.write('\n')
        f.close()
   
# make_all_id('A_weak.txt','A_weak_test.txt','Elist.txt')
def make_one_domain(A_weak,B_strong,train_net,A_flag):
    train_A_weak= defaultdict(list)
    train_B_strong= defaultdict(list)
    train_all= defaultdict(list)
    train_B_strong_shuffe=defaultdict(list)
    with open(A_weak, 'r') as f:
        line_id=0
        for line in f.readlines():
            line = line.strip().split('\t')

            for item in line[1:]:
                if item[0] == A_flag:         
                    train_A_weak[str(line_id)].append(item)
              

            line_id+=1
        f.close()
    with open(B_strong, 'r') as f:
        line_id=0
        for line in f.readlines():
            line = line.strip().split('\t')

            for item in line[1:]:
                if item[0] != A_flag:         
                    train_B_strong[str(line_id)].append(item)
              

            line_id+=1
        f.close()

    idx=np.arange(len(train_A_weak))
    np.random.shuffle(idx)
    id_B=0
    for id in idx:
        train_B_strong_shuffe[str(id)]=train_B_strong[str(id_B)]
        id_B+=1

    for user in train_A_weak.keys():
        train_all[user]=train_A_weak[user][:-1] + train_B_strong_shuffe[user][:-1]
        train_all[user].append(train_A_weak[user][-1])
     
        train_all[user].append(train_B_strong_shuffe[user][-1])
        
    with open(train_net, 'w+') as f:
        for user in train_all.keys():
            f.write(user)
            f.write('\t')
            for item in train_all[user]:
                f.write(item)
                f.write('\t')
            f.write('\n')
        f.close()
def make_one_weak(train_all,A_weak,B_strong,A_flag,min_len):
    train_A_weak= defaultdict(list)
    train_B_strong= defaultdict(list)
    with open(train_all, 'r') as f:
        line_id=0
        for line in f.readlines():
            line = line.strip().split('\t')

            for item in line[1:]:
                if item[0] == A_flag:         
                    train_A_weak[str(line_id)].append(item)
                else:
                    train_B_strong[str(line_id)].append(item)

            line_id+=1
        f.close()

    for user in list(train_A_weak.keys()):
        # user_length = len(train_A_weak[user])
        if len(train_A_weak[user]) < min_len:
            del train_A_weak[user]
        rand_len=random.randint(min_len,8)
        train_A_weak[user]=train_A_weak[user][-rand_len:]
    
    with open(A_weak, 'w+') as f:
        for user in train_A_weak.keys():
            f.write(user)
            f.write('\t')
            for item in train_A_weak[user]:
                f.write(item)
                f.write('\t')
            f.write('\n')
        f.close()

    with open(B_strong, 'w+') as f:
        for user in train_B_strong.keys():
            f.write(user)
            f.write('\t')
            for item in train_B_strong[user]:
                f.write(item)
                f.write('\t')
            f.write('\n')
        f.close()