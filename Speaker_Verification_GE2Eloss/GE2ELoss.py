import os
import time
import argparse
from datetime import datetime
from numpy.core.fromnumeric import shape

#import test_global_storage
import oneflow as flow
import numpy as np
import oneflow.typing as tp
#flow.config.gpu_device_num(8)

def get_utterance_centroids(embeddings):
    """
    Returns the centroids for each utterance of a speaker, where
    the utterance centroid is the speaker centroid without considering
    this utterance

    Shape of embeddings should be:
        (speaker_ct, utterance_per_speaker_ct, embedding_size)
    """
    sum_centroids = flow.math.reduce_sum(embeddings,axis=1)
    # we want to subtract out each utterance, prior to calculating the
    # the utterance centroid
    sum_centroids = flow.reshape(sum_centroids,[sum_centroids.shape[0], 1, sum_centroids.shape[-1]])  

    # we want the mean but not including the utterance itself, so -1
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids

def cosine_similarity(x1,x2,dim=1):
    multiply_numerator= flow.math.reduce_sum(x1*x2,axis=dim)
    denominator = flow.math.sqrt(flow.math.reduce_sum(x1*x1,axis=dim)) * flow.math.sqrt(flow.math.reduce_sum(x2*x2,axis=dim))
    return multiply_numerator/denominator
def get_centroids(embeddings):
    return flow.math.reduce_mean(embeddings,axis=1)
def get_cossim(embeddings,centroids):
    num_utterances = embeddings.shape[1]
    utterance_centroids = get_utterance_centroids(embeddings)

    utterance_centroids_flat = flow.reshape(utterance_centroids,[utterance_centroids.shape[0] * utterance_centroids.shape[1],-1])

    embeddings_flat = flow.reshape(embeddings,[embeddings.shape[0] * num_utterances,-1]) 
    
    cos_same = cosine_similarity(embeddings_flat, utterance_centroids_flat,dim=1)

    centroids_list = []
    for i in range(num_utterances * embeddings.shape[0]):
        centroids_list.append(centroids)

    centroids_expand = flow.concat(centroids_list,axis=0)
    
    embeddings_flat_list = []
    embeddings_flat=flow.reshape(embeddings_flat, [embeddings_flat.shape[0],1,embeddings_flat.shape[1]])
    for i in range(embeddings.shape[0]):
        embeddings_flat_list.append(embeddings_flat)
    embeddings_expand = flow.concat(embeddings_flat_list,axis=1)

    embeddings_expand = flow.reshape(embeddings_expand,[embeddings_expand.shape[0] * embeddings_expand.shape[1],embeddings_expand.shape[-1] ])
    cos_diff = cosine_similarity(embeddings_expand, centroids_expand)
    
    cos_diff = flow.reshape( cos_diff , [embeddings.shape[0],num_utterances,centroids.shape[0] ] )

    # assign the cosine distance for same speakers to the proper idx
    #same_idx = list(range(embeddings.size(0)))

    cos_same = flow.reshape(cos_same,[embeddings.shape[0],num_utterances])
    for i in range(embeddings.shape[0]):
        cos_same_sliced = flow.slice_v2(cos_same,[[i,i+1,1],[0,num_utterances,1]])
        cos_same_sliced = flow.reshape(cos_same_sliced,[1,num_utterances,1])
        cos_diff = flow.slice_update(cos_diff,cos_same_sliced,[(i,i+1,1),(0,cos_diff.shape[1],1),(i,i+1,1)])
    
    cos_diff = cos_diff + 1e-6
    return cos_diff

def GE2Eloss(embeddings):

    def calc_loss(sim_matrix):
        # softmax version loss ,there is a Contrast version loss

        neg = flow.math.log(flow.math.reduce_sum(flow.math.exp(sim_matrix),axis=2) +  1e-6 )
        
        for i in range(sim_matrix.shape[0]):
            if i ==0:
                pos_sum = flow.slice_v2(sim_matrix,[[i,i+1,1],[0,sim_matrix.shape[1],1],[i,i+1,1]]) 
            else:
                pos_sum = pos_sum + flow.slice_v2(sim_matrix,[[i,i+1,1],[0,sim_matrix.shape[1],1],[i,i+1,1]]) 
        
        print('pos_sum',pos_sum.shape)
        pos_sum = flow.math.reduce_sum(pos_sum)
        print('pos_sum',pos_sum.shape)

        neg_sum = flow.math.reduce_sum(neg)

        per_embedding_loss = -1* (pos_sum - neg_sum)

        loss = per_embedding_loss
        return loss, per_embedding_loss

    dtype = flow.float32
    weight_blob_i = flow.get_variable(
    name='input' + '-weight',
    shape=[1],
    dtype=dtype,
    trainable=True,
    initializer=flow.constant_initializer(10.0))

    bias_blob_o = flow.get_variable(
    name='iutput' + '-bias',
    shape=[1],
    dtype=dtype,
    trainable=True,
    initializer=flow.constant_initializer(-5.0))
    
    centroids = get_centroids(embeddings)
    cossim = get_cossim(embeddings, centroids)
    sim_matrix = weight_blob_i*cossim + bias_blob_o
    loss, _ = calc_loss(sim_matrix)
    return loss

def test_GE2ELoss():
    embeddings = np.random.rand(4,5,256).astype(np.float32)
    @flow.global_function()
    def GE2ELoss_Job(x: tp.Numpy.Placeholder(shape=(4, 5, 256), dtype=flow.float32)
    ) -> tp.Numpy:
        #with flow.scope.placement("gpu", "0:5"):
        loss_blob = GE2Eloss(x)
        print(loss_blob.shape)
        return loss_blob
    
    out = GE2ELoss_Job(embeddings)

    print('oneflow output',out)

    import torch
    import torch.nn as nn
    from utils import get_centroids, get_cossim, calc_loss

    class GE2ELoss(nn.Module):
        
        def __init__(self, device):
            super(GE2ELoss, self).__init__()
            self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
            self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
            self.device = device
            
        def forward(self, embeddings):
            #embeddings shape [4, 5, 256]
            torch.clamp(self.w, 1e-6)
            centroids = get_centroids(embeddings)
            cossim = get_cossim(embeddings, centroids)
            sim_matrix = self.w*cossim.to(self.device) + self.b
            loss, _ = calc_loss(sim_matrix)
            return loss

    tensor_embeddings = torch.tensor(embeddings)
    
    pytorch_out=GE2ELoss(torch.device('cuda:0'))(tensor_embeddings)
    
    print('pytorch output',pytorch_out)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    test_GE2ELoss()
    
