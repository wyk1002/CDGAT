import torch
import torch.nn as nn
import math
import scipy.sparse as sp
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import SAGEConv
import numpy as np
import math, copy
from pop_class import *

class ControllerModel():
    def __init__(self, lr, epoches, n_var, n_obj, fname, pop_num,history_length,batch_size=2,dropout=0.5,window_size=None):
        self.epoches=epoches
        self.n_var=n_var
        self.n_obj=n_obj
        self.fname=fname
        self.criterion = nn.MSELoss()

        self.batch_size=batch_size
        self.history_length=history_length

        if window_size:
            assert window_size<=history_length-batch_size
            self.window_size=window_size
        else:
            self.window_size=history_length-batch_size

        self.model = MPGAT(n_var + n_obj, n_var * 32, n_var+ n_obj, 3,self.window_size, dropout=dropout, pop_size=pop_num).cuda()
        # self.model = RelativeTransformerModel(d_i=n_var,d_o=n_obj,head=1,dropout=0.5,layer_num=2)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.98)
        self.weight=None

        self.pop_num=pop_num
        self.relation=[]


    def findominatImdex1(self, pop,n_var,n_obj):  #基于支配关系构造的有向图
        N_particle = pop.shape[0]
        start=[]
        end=[]
        for i in range(N_particle):
            for j in range(N_particle):
                dom_less = 0
                dom_equal = 0
                dom_more = 0
                for k in range (n_obj):
                    if (pop[i,n_var + k] < pop[j,n_var + k]):
                        dom_less = dom_less + 1
                    elif (pop[i,n_var + k] == pop[j,n_var + k]):
                        dom_equal = dom_equal + 1
                    else:
                        dom_more = dom_more + 1
                if (dom_more == 0 and dom_equal != n_obj):
                    start.append(j)
                    end.append(i)
        array1 = np.array(start)
        array2 = np.array(end)
        DR = np.concatenate([array1.reshape(1, -1), array2.reshape(1, -1)], axis=0)
        return DR

    def findominatImdex2(self, pop,n_var,n_obj):
        N_particle = pop.shape[0]
        NicheSize = math.ceil(0.05*N_particle)
        start=[]
        end=[]
        temp_index=[]
        for i in range(N_particle):
            #根据距离选择
            point=pop[i,0:n_var]
            array=pop[:,0:n_var]
            distances = np.linalg.norm(array - point, axis=1)
            NicheIndex = np.argsort(distances)[:NicheSize]
            for j in NicheIndex:
                start.append(i)
                end.append(j)
        array1 = np.array(start)
        array2 = np.array(end)
        DR = np.concatenate([array1.reshape(1, -1), array2.reshape(1, -1)], axis=0)
        return DR

    def findominatImdex3(self, pop,n_var,n_obj):  #基于距离构造无向图
        N_particle = pop.shape[0]
        start=[]
        end=[]
        dominated_index=[]
        for i in range(N_particle):
            for j in range(N_particle):
                dom_less = 0
                dom_equal = 0
                dom_more = 0
                for k in range (n_obj):
                    if (pop[i,n_var + k] < pop[j,n_var + k]):
                        dom_less = dom_less + 1
                    elif (pop[i,n_var + k] == pop[j,n_var + k]):
                        dom_equal = dom_equal + 1
                    else:
                        dom_more = dom_more + 1
                if (dom_less == 0 and dom_equal != n_obj): #j支配i
                     dominated_index.append(j)
            if(len(dominated_index)==0):
                point=pop[i,0:n_var]
                array=pop[:,0:n_var]
                NicheSize = math.ceil(0.05*N_particle)
                distances = np.linalg.norm(array - point, axis=1)
                NicheIndex = np.argsort(distances)[::-1][:NicheSize]
                for j in NicheIndex:
                    start.append(i)
                    end.append(j)
            else:
                point=pop[i,0:n_var]
                array=pop[dominated_index,0:n_var]
                NicheSize = math.ceil(0.5*len(dominated_index))
                distances = np.linalg.norm(array - point, axis=1)
                NicheIndex = np.argsort(distances)[::-1][:NicheSize]
                for j in NicheIndex:
                    start.append(i)
                    end.append(dominated_index[j])
        array1 = np.array(start)
        array2 = np.array(end)
        DR = np.concatenate([array1.reshape(1, -1), array2.reshape(1, -1)], axis=0)
        return DR

    def find_niche_neighbors(self, pop, n_var, niche_ratio=0.05):
        """
        基于(决策,目标)空间的K近邻，构建个体间的邻接关系图。

        参数:
            pop: 种群，shape=(N, n_var + ...)
            n_var: 决策变量维度
            niche_ratio: 小生境比例，默认5%

        返回:
            DR: (2, M) 数组，0行为起点，1行为终点，表示 i -> j 是邻居
        """
        N_particle = pop.shape[0]
        NicheSize = max(2, int(niche_ratio * N_particle))  # 至少保留2个邻居

        # 提取决策变量并归一化（避免量纲影响）
        X = pop[:, :n_var]
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)  # Min-Max 归一化

        # 使用 KNN 快速查找最近邻
        nbrs = NearestNeighbors(n_neighbors=NicheSize, metric='euclidean').fit(X)
        _, indices = nbrs.kneighbors(X)  # indices.shape = (N, k)

        # 构建边列表：i -> j 表示 j 是 i 的邻居
        adj=np.zeros(shape=(N_particle,N_particle))
        start = []
        end = []
        for i in range(N_particle):
            adj[i,indices[i]]=1
            # for j in indices[i]:
            #     if i != j:  # 可选：去除自环
            #         start.append(i)
            #         end.append(j)


        # # 转为 numpy 数组
        # DR = np.array([start, end])
        return adj

    def get_trainBatch(self, b, w, train_data):
        X = np.zeros(shape=(b, w, self.pop_num, self.n_var+self.n_obj))
        GT=np.zeros(shape=(b, self.pop_num, self.n_var+self.n_obj))
        adj=np.zeros(shape=(b,w,self.pop_num,self.pop_num))
        for i in range(b):
            for j in range(w):
                X[i,j,:,:]=train_data[self.history_length-w-b+i+j].pop
            GT[i,:,:]=train_data[self.history_length-b+i].pop
            # a=self.relation[self.history_length-w-b+i:self.history_length-b+i]
            # b=np.asmatrix(a)
            adj[i,:,:,:]=np.array(self.relation[self.history_length-w-b+i:self.history_length-b+i])
        return X, GT, adj

    def get_testBatch(self,w,test_data):
        X = np.zeros(shape=(1, w, self.pop_num, self.n_var + self.n_obj))
        for i in range(w):
            X[0,i,:,:]=test_data[self.history_length-w+i].pop
        return X

    def train(self, train_data):
        '''
        train_data=[step,population]
        '''
        self.model.train()

        for i in range (len(train_data)):
            DR=self.find_niche_neighbors(train_data[i].pop,self.n_var,niche_ratio=0.05)
            # DR=self.findominatImdex2(train_data[i].pop,self.n_var,self.n_obj)
            self.relation.append(DR)

        node_train, node_GT, adjs = self.get_trainBatch(self.batch_size, self.window_size, train_data)
        node_train = torch.from_numpy(node_train).to(torch.float32).cuda()
        node_GT = torch.from_numpy(node_GT).to(torch.float32).cuda()
        # node_train = node_train.cuda()
        # node_GT = node_GT.cuda()
        adj_train = torch.from_numpy(adjs).to(torch.int).cuda()
        # adj_train=adj_train.cuda()


        for epoch in range(self.epoches):
            self.optimizer.zero_grad()
            predict_result=self.model(node_train,adj_train)
            predict_result.to(torch.float32)
            loss = self.criterion(predict_result, node_GT)
            total_loss = loss.item()
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1,self.epoches,total_loss))

    def predict_offspring(self, train_data):
        self.model.eval()
        X=self.get_testBatch(self.window_size, train_data)
        DR_train=np.array(self.relation[-self.window_size:])
        DR_train=torch.from_numpy(DR_train).to(torch.int).cuda()

        X=torch.from_numpy(X).to(torch.float32).cuda()
        predict_result= self.model(X, DR_train)
        EXA=predict_result.reshape(self.pop_num,self.n_obj+self.n_var)
        return EXA


class MPGAT(nn.Module):
    def __init__(self,des_dim, feature_dim, obj_dim, GAT_layers,window_size,dropout,pop_size):
        super(MPGAT, self).__init__()
        if des_dim!=feature_dim:
            self.dimChange = nn.Linear(des_dim, feature_dim)
        else:
            self.dimChange =None
        self.RelativeGATLayer=RelativeGAT(feature_dim,dropout=dropout,MAX_Length=pop_size,GAT_layers=GAT_layers)

        self.timeLinear=nn.Linear(window_size,1)

        if feature_dim!=obj_dim:
            self.outputLayer=nn.Linear(feature_dim,obj_dim)
        else:
            self.outputLayer = None

    def forward(self,inputs,GraphAdjacencyMatrix):
        '''
        inputs=[b,w,p,d]
        '''
        if self.dimChange:
            inputs=self.dimChange(inputs)
        inputs=self.RelativeGATLayer(inputs,GraphAdjacencyMatrix)
        if self.outputLayer:
            inputs=self.outputLayer(inputs)
        return inputs

class RelativeGAT(nn.Module):
    def __init__(self,dim,MAX_Length,GAT_layers,dropout):
        super(RelativeGAT, self).__init__()
        self.MAX_Length=MAX_Length
        self.GAT_layers=GAT_layers

        pos_emb = self.getPOS(2*MAX_Length + 1,dim,None)
        self.register_buffer('pos_emb', pos_emb)

        self.layers = clones(RelativeGatBlock(dim, 1, dropout, scale=True), GAT_layers)
        
    def getPOS(self,token_length, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-token_length // 2, token_length // 2, dtype=torch.float).unsqueeze(
            1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(token_length, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(token_length, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, inputs, GraphAdjacencyMatrix):
        b,w,l,d=inputs.shape
        positions = torch.arange(-l, l).to(inputs.device).long() + self.MAX_Length  # 2*seq_len
        pos_emb = self.pos_emb.index_select(0, positions.long()).detach()

        for i_layer in range(self.GAT_layers):
            inputs=self.layers[i_layer](inputs,GraphAdjacencyMatrix,pos_emb)
        outputs=torch.max(inputs, dim=1, keepdim=False)[0]
        return outputs


class RelativeGatBlock(nn.Module):
    def __init__(self,dim,n_head,dropout,scale=False):
        super(RelativeGatBlock, self).__init__()
        self.d_m = dim
        self.qkv_linear = nn.Linear(dim, dim * 3, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

        if scale:
            self.scale = math.sqrt(dim)
        else:
            self.scale = 1

        self.u = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, dim // n_head)))
        self.v = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, dim // n_head)))

        self.layer_normal_1 = nn.LayerNorm(dim)
        self.layer_normal_2 = nn.LayerNorm(dim)
        self.FNN = FFN(dim, dropout)

    def forward(self,inputs,GraphAdjacencyMatrix,pos_emb):
        batch_size, window_size,max_len, d_i=inputs.shape
        qkv = self.qkv_linear(inputs)  # batch_size x max_len x d_model3
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        # q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        # k = k.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        # v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)  # b x n x l x d

        rw_head_q = q + self.u[:, None]
        AC = torch.einsum('bnqd,bnkd->bnqk', [rw_head_q, k])  # b x n x l x d

        D_ = torch.einsum('nd,ld->nl', self.v, pos_emb)[None, :, None]  # head x 2max_len, 每个head对位置的bias

        B_ = torch.einsum('bnqd,ld->bnql', q, pos_emb)  # bsz x head  x max_len x 2max_len，每个query对每个shift的偏移
        E_ = torch.einsum('bnqd,ld->bnql', k, pos_emb)  # bsz x head x max_len x 2max_len, key对relative的bias
        BD = B_ + D_  # bsz x head x max_len x 2max_len, 要转换为bsz x head x max_len x max_len
        BDE = self._shift(BD) + self._transpose_shift(E_)
        attn = AC + BDE

        attn = attn / self.scale

        att = F.softmax(attn.masked_fill(GraphAdjacencyMatrix == False, -1e9), dim=-1)  # (b,h,l,l)
        attn = self.dropout_layer(att)
        v = torch.matmul(attn, v)  # b x  l x d

        sum = self.layer_normal_1(self.dropout_layer(v) + inputs)  # b x  l x d_o
        out = self.layer_normal_2(self.FNN(sum) + sum)  # b x  l x d_o

        return out

    def _shift(self, BD):
        """
        类似
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2

        转换为
        0   1  2
        -1  0  1
        -2 -1  0

        :param BD: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = BD.size()
        zero_pad = BD.new_zeros(bsz, n_head, max_len, 1)
        BD = torch.cat([BD, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)  # bsz x n_head x (2max_len+1) x max_len
        BD = BD[:, :, :-1].view(bsz, n_head, max_len, -1)  # bsz x n_head x 2max_len x max_len
        BD = BD[:, :, :, max_len:]
        return BD

    def _transpose_shift(self, E):
        """
        类似
          -3   -2   -1   0   1   2
         -30  -20  -10  00  10  20
        -300 -200 -100 000 100 200

        转换为
          0  -10   -200
          1   00   -100
          2   10    000


        :param E: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = E.size()
        zero_pad = E.new_zeros(bsz, n_head, max_len, 1)
        # bsz x n_head x -1 x (max_len+1)
        E = torch.cat([E, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)
        indice = (torch.arange(max_len) * 2 + 1).to(E.device)
        E = E.index_select(index=indice, dim=-2).transpose(-1, -2)  # bsz x n_head x max_len x max_len

        return E



class FFN(nn.Module):
    def __init__(self,d_m,dropout):
        super().__init__()
        self.w_1=nn.Linear(d_m,d_m)
        self.w_2=nn.Linear(d_m,d_m)
        self.dropout=nn.Dropout(dropout)
        self.gelu=nn.GELU()

    def forward(self,x):
        return self.dropout(self.w_2(self.dropout(self.gelu(self.w_1(x)))))

class RelativeEmbedding(nn.Module):
    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen].
        """

        bsz, seq_len = input.size()
        max_pos = self.padding_idx + seq_len
        if max_pos > self.origin_shift:
            # recompute/expand embeddings if needed
            weights = self.get_embedding(
                max_pos * 2,
                self.embedding_dim,
                self.padding_idx,
            )
            weights = weights.to(self._float_tensor)
            del self.weights
            self.origin_shift = weights.size(0) // 2
            self.register_buffer('weights', weights)

        positions = torch.arange(-seq_len, seq_len).to(input.device).long() + self.origin_shift  # 2*seq_len
        embed = self.weights.index_select(0, positions.long()).detach()
        return embed

class RelativeSinusoidalPositionalEmbedding(RelativeEmbedding):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=50):
        """

        :param embedding_dim: 每个位置的dimension
        :param padding_idx:
        :param init_size:
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size % 2 == 0
        weights = self.get_embedding(
            init_size + 1,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('weights', weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-num_embeddings // 2, num_embeddings // 2, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        self.origin_shift = num_embeddings // 2 + 1
        return emb

class RelativeTransformerModel(nn.Module):
    def __init__(self, d_i, d_o, head, dropout, layer_num):
        super().__init__()
        self.layers = clones(RelativeMultiHeadAttn(d_i, d_o, head, dropout, scale=True), layer_num)

    def forward(self, x, lens_mask, self_mask):
        for layer in self.layers:
            x = layer(x, lens_mask, self_mask)
        return x

class RelativeMultiHeadAttn(nn.Module):
    def __init__(self, d_i, d_o, n_head, dropout, r_w_bias=None, r_r_bias=None, scale=True):
        """

        :param int d_model:
        :param int n_head:
        :param dropout: 对attention map的dropout
        :param r_w_bias: n_head x head_dim or None, 如果为dim
        :param r_r_bias: n_head x head_dim or None,
        :param scale:
        :param rel_pos_embed:
        """
        super().__init__()
        assert d_o % n_head == 0
        self.d_m = d_o
        if d_i != d_o:
            self.reshape = nn.Linear(d_i, d_o, bias=False)
        self.qkv_linear = nn.Linear(d_o, d_o * 3, bias=False)
        self.n_head = n_head
        self.head_dim = d_o // n_head
        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_o // n_head, 0, 1200)

        if scale:
            self.scale = math.sqrt(d_o // n_head)
        else:
            self.scale = 1

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_o // n_head)))
            self.r_w_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_o // n_head)))
        else:
            self.r_r_bias = r_r_bias  # r_r_bias就是v
            self.r_w_bias = r_w_bias  # r_w_bias就是u

        self.layer_normal_1 = nn.LayerNorm(d_o)
        self.layer_normal_2 = nn.LayerNorm(d_o)
        self.FNN = FFN(d_o, dropout)

    def forward(self, x, lens_mask, self_mask):
        """

        :param x: batch_size x max_len x d_model
        :param mask: batch_size x max_len
        :return:
        """

        batch_size, max_len, d_i = x.size()
        if d_i != self.d_m:
            x = self.reshape(x)
        pos_embed = self.pos_embed(lens_mask)  # l x head_dim

        qkv = self.qkv_linear(x)  # batch_size x max_len x d_model3
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)  # b x n x l x d

        rw_head_q = q + self.r_r_bias[:, None]
        AC = torch.einsum('bnqd,bnkd->bnqk', [rw_head_q, k])  # b x n x l x d, n是head

        D_ = torch.einsum('nd,ld->nl', self.r_w_bias, pos_embed)[None, :, None]  # head x 2max_len, 每个head对位置的bias
        B_ = torch.einsum('bnqd,ld->bnql', q, pos_embed)  # bsz x head  x max_len x 2max_len，每个query对每个shift的偏移
        E_ = torch.einsum('bnqd,ld->bnql', k, pos_embed)  # bsz x head x max_len x 2max_len, key对relative的bias
        BD = B_ + D_  # bsz x head x max_len x 2max_len, 要转换为bsz x head x max_len x max_len
        BDE = self._shift(BD) + self._transpose_shift(E_)
        attn = AC + BDE

        attn = attn / self.scale

        mask_expand = self_mask.unsqueeze(1).contiguous()  # (b,1,l,l)
        # print(mask_expand.shape,attn.shape,self_mask.shape,lens_mask.shape)
        att = F.softmax(attn.masked_fill(mask_expand == False, -1e9), dim=-1)  # (b,h,l,l)
        att_fixed = mask_expand.long() * att  # (b,h,l,l)

        # attn = attn.masked_fill(mask[:, None, None, :].eq(0), float('-inf'))
        # att_fixed = F.softmax(attn, dim=-1)

        attn = self.dropout_layer(att_fixed)
        v = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, max_len, self.d_m)  # b x  l x d

        sum = self.layer_normal_1(self.dropout_layer(v) + x)  # b x  l x d_o
        out = self.layer_normal_2(self.FNN(sum) + sum)  # b x  l x d_o

        return out

    def _shift(self, BD):
        """
        类似
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2

        转换为
        0   1  2
        -1  0  1
        -2 -1  0

        :param BD: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = BD.size()
        zero_pad = BD.new_zeros(bsz, n_head, max_len, 1)
        BD = torch.cat([BD, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)  # bsz x n_head x (2max_len+1) x max_len
        BD = BD[:, :, :-1].view(bsz, n_head, max_len, -1)  # bsz x n_head x 2max_len x max_len
        BD = BD[:, :, :, max_len:]
        return BD

    def _transpose_shift(self, E):
        """
        类似
          -3   -2   -1   0   1   2
         -30  -20  -10  00  10  20
        -300 -200 -100 000 100 200

        转换为
          0  -10   -200
          1   00   -100
          2   10    000


        :param E: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = E.size()
        zero_pad = E.new_zeros(bsz, n_head, max_len, 1)
        # bsz x n_head x -1 x (max_len+1)
        E = torch.cat([E, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)
        indice = (torch.arange(max_len) * 2 + 1).to(E.device)
        E = E.index_select(index=indice, dim=-2).transpose(-1, -2)  # bsz x n_head x max_len x max_len

        return E

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def modelTest():
    runs = 20  # 运行次数
    N_function = 33
    eps = 0.1
    p = 0.5
    history_length = 8
    batch_size = 2
    epoch = 300
    Popsize=20
    pi_value = np.pi
    fname='test'
    lr = 0.01
    n_var=2
    n_obj=2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn(size=[8, Popsize, n_var])
    GraphAdjacencyMatrix=torch.randint(low=0,high=2,size=[8,Popsize,Popsize])
    # model=ControllerModel(lr, epoch, n_var, n_obj, fname, Popsize,device)
    model=MPGAT(n_var, n_var*32, n_obj, 3, dropout=0.5,pop_size=Popsize)
    out=model(inputs,GraphAdjacencyMatrix)
    print(out.shape)

if __name__ == '__main__':
    modelTest()


