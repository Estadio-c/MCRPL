import numpy as np
import torch
from torch import nn
import math
from torch.nn import functional as F
import copy
# prompt Aggregation Strategy 1
class Promptattention_a(nn.Module):
    def __init__(self,dk):
        super(Promptattention_a, self).__init__()
        self.dk=dk
    def forward(self, Q, K, V, mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dk) # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) 
        return context, attn
    
class AttentionLayer_a(nn.Module):
    def __init__(self,args):
        super(AttentionLayer_a, self).__init__()
        self.args=args
        self.d_model=args.hidden_units
        self.dk=args.d_k
        self.n_heads=args.n_heads
        self.dv=args.d_v
        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=False)
        self.fc = nn.Linear(self.d_model, self.d_model, bias=False)
    def forward(self, input):

        residual, batch_size = input, input.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input)  # Q: [batch_size, len, prompt, d_model]
        K = self.W_K(input) 
        V = self.W_V(input)  
        attn_shape = [input.size(0), input.size(2), input.size(2)]
        subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
        subsequence_mask = torch.from_numpy(subsequence_mask).byte()
        subsequence_mask=subsequence_mask.unsqueeze(1).expand(-1, input.size(1), -1, -1).to(self.args.device)
 
        context, attn = Promptattention_a(self.dk)(Q, K, V,subsequence_mask)
        output = self.fc(context) 
        return nn.LayerNorm(self.d_model).to(self.args.device)(output + residual)

class PromptLearner_a(nn.Module):
    def __init__(self, args,item_num):
        super().__init__()
        self.args=args
        
        emb_num = 2   
        emb_num_S =2  
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.src_emb = nn.Embedding(item_num+1, args.hidden_units)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.context_embedding_E = nn.Embedding(emb_num, args.hidden_units)
        self.context_embedding_s_E = nn.Embedding(emb_num_S, args.hidden_units) #share
        self.attention_E = AttentionLayer_a(args)
        embedding_E = self.context_embedding_E(torch.LongTensor(list(range(emb_num))))
        embedding_S_E = self.context_embedding_s_E(torch.LongTensor(list(range(emb_num_S))))
        ctx_vectors_E = embedding_E
        ctx_vectors_S_E = embedding_S_E
        self.ctx_E = nn.Parameter(ctx_vectors_E)  
        self.ctx_S_E = nn.Parameter(ctx_vectors_S_E)


    def forward(self, seq ):
 
        seq_feat=self.src_emb(seq)
        positions = np.tile(np.array(range(seq.shape[1])), [seq.shape[0], 1])
        seq_feat += self.pos_emb(torch.LongTensor(positions).to(self.args.device))
        seq_feat = self.emb_dropout(seq_feat)
        ctx_E = self.ctx_E 
        ctx_S_E = self.ctx_S_E 
        ctx_E_1 = ctx_E 

        if ctx_S_E.dim() == 2:
            ctx_E = ctx_E_1.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1] ,-1, -1)  
   
            ctx_S_E = ctx_S_E.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1] ,-1, -1)  

        ctx_prefix_E = self.getPrompts(seq_feat.unsqueeze(2), ctx_E, ctx_S_E ) # 128 15 8 100

        prompts_E = self.attention_E(ctx_prefix_E)[:,:,-1,:]
        return prompts_E

    def getPrompts(self, prefix, ctx,ctx_S): #ctx_S, suffix=None):#
    
        prompts = torch.cat(
            [
                ctx_S, 
                ctx,  
                prefix 
            ],
            dim=2,
        )
        return prompts


# prompt Aggregation Strategy 2

class Promptattention_b(nn.Module):
    def __init__(self,dk):
        super(Promptattention_b, self).__init__()
        self.dk=dk
    def forward(self, Q, K, V, mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dk) # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) 
        return context, attn
    
class AttentionLayer_b(nn.Module):
    def __init__(self,args):
        super(AttentionLayer_b, self).__init__()
        self.args=args
        self.d_model=args.hidden_units
        self.dk=args.d_k
        self.n_heads=args.n_heads
        self.dv=args.d_v
        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=False)
        self.fc = nn.Linear(self.d_model, self.d_model, bias=False)
    def forward(self, input):

        residual, batch_size = input, input.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input)  # Q: [batch_size, len, prompt, d_model]
        K = self.W_K(input) 
        V = self.W_V(input)  
        attn_shape = [input.size(0), input.size(2), input.size(2)]
        subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
        subsequence_mask = torch.from_numpy(subsequence_mask).byte()
        subsequence_mask=subsequence_mask.unsqueeze(1).expand(-1, input.size(1), -1, -1).to(self.args.device)
 
        context, attn = Promptattention_b(self.dk)(Q, K, V,subsequence_mask)
        output = self.fc(context) 
        return nn.LayerNorm(self.d_model).to(self.args.device)(output + residual)

class PromptLearner_b(nn.Module):
    def __init__(self, args,item_num):
        super().__init__()
        self.args=args
        
        emb_num = 2   
        emb_num_S =2  
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.src_emb = nn.Embedding(item_num+1, args.hidden_units)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.context_embedding_E = nn.Embedding(emb_num, args.hidden_units)
        self.context_embedding_s_E = nn.Embedding(emb_num_S, args.hidden_units) #share
        self.attention_E = AttentionLayer_b(args)
        embedding_E = self.context_embedding_E(torch.LongTensor(list(range(emb_num))))
        embedding_S_E = self.context_embedding_s_E(torch.LongTensor(list(range(emb_num_S))))
        ctx_vectors_E = embedding_E
        ctx_vectors_S_E = embedding_S_E
        self.ctx_E = nn.Parameter(ctx_vectors_E)  
        self.ctx_S_E = nn.Parameter(ctx_vectors_S_E)


    def forward(self, seq ):
 
        seq_feat=self.src_emb(seq)
        positions = np.tile(np.array(range(seq.shape[1])), [seq.shape[0], 1])
        seq_feat += self.pos_emb(torch.LongTensor(positions).to(self.args.device))
        seq_feat = self.emb_dropout(seq_feat)
        ctx_E = self.ctx_E 
        ctx_S_E = self.ctx_S_E 
        ctx_E_1 = ctx_E 

        if ctx_S_E.dim() == 2:
            ctx_E = ctx_E_1.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1] ,-1, -1)  
   
            ctx_S_E = ctx_S_E.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1] ,-1, -1)  

        ctx_prefix_E = self.getPrompts(seq_feat.unsqueeze(2), ctx_E, ctx_S_E ) # 128 15 8 100
        number = ctx_prefix_E.shape[2]
        prompts_E = self.attention_E(ctx_prefix_E).sum(2)/number
        return prompts_E

    def getPrompts(self, prefix, ctx,ctx_S): 
    
        prompts = torch.cat(
            [
                ctx_S, 
                ctx,  
                prefix 
            ],
            dim=2,
        )
        return prompts

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
  
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)   
    return pad_attn_mask.expand(batch_size, len_q, len_k) 
def get_attn_subsequence_mask(seq):

    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self,dk):
        super(ScaledDotProductAttention, self).__init__()
        self.dk=dk
    def forward(self, Q, K, V, attn_mask):

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dk) 
        scores.masked_fill_(attn_mask, -1e9) 
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) 

        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,args,d_model,d_k,n_heads,d_v):
        super(MultiHeadAttention, self).__init__()
        self.args=args
        self.d_model=d_model
        self.dk=d_k
        self.n_heads=n_heads
        self.dv=d_v
        self.W_Q = nn.Linear(self.d_model, self.dk * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.dk * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.dv * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.dv, self.d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):

        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.dk).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.dk).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.dv).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.dk)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,(self.n_heads) * (self.dv)) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        
        return nn.LayerNorm(self.d_model).to(self.args.device)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,args,d_model,d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.args=args
        self.d_model=d_model
        self.d_ff=d_ff
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )
    def forward(self, inputs):

        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(self.args.device)(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):

    def __init__(self,args,d_model,d_k,n_heads,d_v,d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args,d_model,d_k,n_heads,d_v)
        self.pos_ffn = PoswiseFeedForwardNet(args,d_model,d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):

        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn
    
class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0.0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )
    def forward(self, x, mask=None):
    
        out = self.linear(x)
        if mask is not None:  
            out = out.masked_fill(mask, -100000)  
            weight = F.softmax(out, dim=1)
            return weight
        else:
            weight = F.softmax(out, dim=2) 
        return weight 

class PromptLearner(nn.Module):
    def __init__(self, args,item_num):
        super().__init__()
        self.args=args
        emb_num = 2   
        emb_num_S =2  
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.src_emb = nn.Embedding(item_num+1, args.hidden_units)
     
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE

        self.context_embedding_E = nn.Embedding(emb_num, args.hidden_units)
      
        self.context_embedding_s_E = nn.Embedding(emb_num_S, args.hidden_units) #share

        drop_out = 0.25

        self.attention_E = AttentionLayer(2 * args.hidden_units, drop_out)
   

        embedding_E = self.context_embedding_E(torch.LongTensor(list(range(emb_num))))
 
        embedding_S_E = self.context_embedding_s_E(torch.LongTensor(list(range(emb_num_S))))
   

        ctx_vectors_E = embedding_E

        ctx_vectors_S_E = embedding_S_E
    
        self.ctx_E = nn.Parameter(ctx_vectors_E) 
  
        self.ctx_S_E = nn.Parameter(ctx_vectors_S_E)

    def forward(self, seq ):
  
        seq_feat=self.src_emb(seq)
        # seq_feat *= self.src_emb.embedding_dim ** 0.5
        # seq_feat = self.pos_emb(seq_feat.transpose(0, 1)).transpose(0, 1)
        positions = np.tile(np.array(range(seq.shape[1])), [seq.shape[0], 1])
        seq_feat += self.pos_emb(torch.LongTensor(positions).to(self.args.device))

        seq_feat = self.emb_dropout(seq_feat)

        ctx_E = self.ctx_E 
    
        ctx_S_E = self.ctx_S_E 

     
        ctx_E_1 = ctx_E 
   

        if ctx_S_E.dim() == 2:
            ctx_E = ctx_E_1.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1] ,-1, -1)  
       
            ctx_S_E = ctx_S_E.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1] ,-1, -1)  
       
        ctx_prefix_E = self.getPrompts(seq_feat.unsqueeze(2), ctx_E, ctx_S_E ) # 128 15 8 100

        item_embedding = seq_feat.unsqueeze(2).expand(-1, -1 ,ctx_prefix_E.shape[2],-1)
        prompt_item = torch.cat((ctx_prefix_E, item_embedding), dim=3)
        at_wt = self.attention_E(prompt_item)
 

        prompts_E = torch.matmul(at_wt.permute(0, 1, 3, 2), ctx_prefix_E).squeeze() 

        return prompts_E

    def getPrompts(self, prefix, ctx,ctx_S): #ctx_S, suffix=None):#
    
        prompts = torch.cat(
            [
                ctx_S, 
                ctx,  
                prefix 
            ],
            dim=2,
        )
        return prompts


class mcrpl(nn.Module):
    def __init__(self,args,item_num):
        super(mcrpl, self).__init__()
        self.args=args
        self.class_=nn.Linear(args.hidden_units,args.all_size)
        # self.src_emb = nn.Embedding(item_num+1, args.hidden_units)
        # self.pos_emb = PositionalEncoding(args.hidden_units)
        self.layers = nn.ModuleList([EncoderLayer(self.args,args.hidden_units,args.d_k,args.n_heads,args.d_v,args.d_ff) for _ in range(args.n_layers)])
        if args.Strategy == 'default' :
            self.prompt=PromptLearner(args,item_num)
        elif args.Strategy == 'a':
            self.prompt=PromptLearner_a(args,item_num)
        elif args.Strategy == 'b':
            self.prompt=PromptLearner_b(args,item_num)
    def phase_one(self, user , log_seqs):
  
        enc_outputs = self.prompt(log_seqs)

        enc_self_attn_mask = get_attn_pad_mask(log_seqs, log_seqs) # [batch_size, src_len, src_len]

        enc_attn_mask=get_attn_subsequence_mask(log_seqs).to(self.args.device)

        all_mask=torch.gt((enc_self_attn_mask + enc_attn_mask), 0).to(self.args.device)
        
        enc_self_attns = []
        for layer in self.layers:
            
            enc_outputs, enc_self_attn = layer(enc_outputs, all_mask)
    
            enc_self_attns.append(enc_self_attn)
   
        logits=self.class_(enc_outputs[:,-1,:])
        return logits , self.prompt.ctx_E ,  self.prompt.ctx_S_E
    

    def forward(self,user,log_seqs):
        logits=self.phase_one(user,log_seqs)
        return logits
    
    def Freeze_a(self):#tune prompt + head
        for param in self.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():

            if "ctx_S_E" in name:
                param.requires_grad = True
            if "class_" in name:
                param.requires_grad = True
        self.prompt.src_emb.requires_grad = True
        self.prompt.pos_emb.requires_grad = True

    def Freeze_b(self):#tune prompt + head
        for param in self.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            #print(name)
            if "ctx" in name:
                param.requires_grad = True
            if "class_" in name:
                param.requires_grad = True
        self.prompt.src_emb.requires_grad = True
        # self.prompt.pos_emb.requires_grad = True

    def Freeze_c(self):#tune prompt + head
        for param in self.parameters():
            param.requires_grad = False
            
        for name, param in self.named_parameters():
            #print(name)
            if "layers" in name:
                param.requires_grad = True
            if "class_" in name:
                param.requires_grad = True
        self.prompt.src_emb.requires_grad = True
        # self.prompt.pos_emb.requires_grad = True
        
    def Freeze_d(self):#tune prompt + head
        for param in self.parameters():
            param.requires_grad = False
            
        for name, param in self.named_parameters():
            #print(name)
            if "ctx" in name:
                param.requires_grad = True
            if "class_" in name:
                param.requires_grad = True
        self.prompt.src_emb.requires_grad = True
        # self.prompt.pos_emb.requires_grad = True

if __name__ == '__main__':
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
    parser.add_argument('--Lambda', default=0.0001, type=float)

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
    model = mcrpl(args,args.all_size)
    for name, param in model.named_parameters():
        print(name)


