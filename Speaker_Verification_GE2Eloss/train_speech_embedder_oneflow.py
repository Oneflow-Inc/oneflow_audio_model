#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:49:16 2021

@author: fujiaqing
"""

import os
import random
import time
import oneflow as flow
import oneflow.typing as tp
import numpy as np
import sys
from hparam import hparam as hp
from data_load_oneflow import SpeakerDatasetTIMIT, SpeakerDatasetTIMITPreprocessed
from speech_embedder_net_oneflow import SpeechEmbedder
from GE2ELoss import GE2Eloss, get_centroids, get_cossim
import oneflow._oneflow_internal as oneflow_api
import oneflow.python.framework.session_context as session_ctx

flow.config.gpu_device_num(4)

def train(model_path):
    def GetLarsVariablesForCurrentJob():
        sess = session_ctx.GetDefaultSession()
        job_name = oneflow_api.JobBuildAndInferCtx_GetCurrentJobName()
        print('job_name',job_name)
        all_vars = list(sess.job_name2var_name2var_blob_[job_name].keys())
        return [var for var in all_vars if var!='input-weight' and var !='iutput-bias']
    
    @flow.global_function(type='train')
    def train(
        mel_db_batch: tp.Numpy.Placeholder((hp.train.N*hp.train.M,160,40), dtype=flow.float),
        ) -> tp.Numpy:

        with flow.scope.placement("gpu", "0:0-3"):
            embeddings = SpeechEmbedder(mel_db_batch, train=True)
            print('embeddings',embeddings.shape)
            embeddings = flow.reshape(embeddings,(hp.train.N, hp.train.M, embeddings.shape[1]))
            loss = GE2Eloss(embeddings)
            
        lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([],[hp.train.lr])

        sgd_opt = flow.optimizer.SGD(lr_scheduler, momentum=0.0,grad_clipping=flow.optimizer.grad_clipping.by_global_norm(3.0),variables=GetLarsVariablesForCurrentJob())
        #print(GetLarsVariablesForCurrentJob())
        sgd_opt1 = flow.optimizer.SGD(lr_scheduler, momentum=0.0,grad_clipping=flow.optimizer.grad_clipping.by_global_norm(1.0),variables=['input-weight','iutput-bias'])
        flow.optimizer.CombinedOptimizer([sgd_opt, sgd_opt1]).minimize(
                loss)
        #flow.optimizer.SGD(lr_scheduler, momentum=0.0,grad_clipping=gradient_clip).minimize(loss)
        
        return loss

    
    if hp.data.data_preprocessed:
        train_dataset = SpeakerDatasetTIMITPreprocessed()
    else:
        train_dataset = SpeakerDatasetTIMIT()
    

    if hp.train.restore:
        embedder_net.load_state_dict(torch.load(model_path))
    
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
    with open(hp.train.log_file,'w') as f:
        f.write('')      
    
    iteration = 0
    num_speaker = len(os.listdir(train_dataset.path))

    for e in range(hp.train.epochs):
        total_loss = 0
        train_index = np.arange(num_speaker)
        np.random.shuffle(train_index)
        #print(train_index) 
        for batch_id in range(num_speaker//hp.train.N): 
            for i in range(hp.train.N):
                mel_db = train_dataset[train_index[batch_id*hp.train.N+i]]
                if i==0:
                    mel_db_batch = mel_db
                else:    
                    mel_db_batch = np.concatenate((mel_db,mel_db_batch),axis=0)
            
            #print('mel_db_batch:',mel_db_batch.shape) #[4, 5, 160, 40]
            #mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.shape[2], mel_db_batch.shape[3]))
            #print('mel_db_batch:',mel_db_batch.shape) #[20, 160, 40]
            
            #get loss, call backward, step optimizer
            #print('shape before:',embeddings.shape) #[4, 5, 256]
            loss = train(mel_db_batch) #wants (Speaker, Utterances, embedding)
            #print('loss :',loss.shape)
            #loss = predict(mel_db_batch)
            
            loss = loss[0]
            total_loss = total_loss + loss
            iteration += 1
            if (batch_id + 1) % hp.train.log_interval == 0:
                print('mel_db_batch:',mel_db_batch.shape)
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(time.ctime(), e+1,
                        batch_id+1, len(train_dataset)//hp.train.N, iteration,loss, total_loss / (batch_id + 1))
                print(mesg)
                if hp.train.log_file is not None:
                    with open(hp.train.log_file,'a') as f:
                        f.write(mesg)
        print('batch_id',batch_id)            
        if hp.train.checkpoint_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
            
            ckpt_model_filename = "ckpt_epoch_" + str(e+1) + "_batch_id_" + str(batch_id+1) + "model"
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            flow.checkpoint.save(ckpt_model_path)
            

    #save model
    
    save_model_filename = "final_epoch_" + str(e + 1) + "_batch_id_" + str(batch_id + 1) + "model"
    save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
    flow.checkpoint.save(save_model_path)
    
    print("\nDone, trained model saved at", save_model_path)

def test(model_path):
    
    if hp.data.data_preprocessed:
        test_dataset = SpeakerDatasetTIMITPreprocessed()
    else:
        test_dataset = SpeakerDatasetTIMIT()
    
    @flow.global_function(type='predict')
    def predict(
        enrollment_batch: tp.Numpy.Placeholder((hp.test.N*hp.test.M//2,160,40), dtype=flow.float),
        verification_batch: tp.Numpy.Placeholder((hp.test.N*hp.test.M//2,160,40), dtype=flow.float)
        ) -> tp.Numpy:

        with flow.scope.placement("gpu", "0:0-3"):
            # first concat
            batch = flow.concat(inputs=[enrollment_batch, verification_batch],
                            axis=0)
            embeddings = SpeechEmbedder(batch, train=False)
            enrollment_embeddings = flow.slice_v2(embeddings,[[0,embeddings.shape[0]//2,1],[0,embeddings.shape[1],1]]) #SpeechEmbedder(enrollment_batch, train=False)
            verification_embeddings = flow.slice_v2(embeddings,[[embeddings.shape[0]//2,embeddings.shape[0],1],[0,embeddings.shape[1],1]]) #SpeechEmbedder(verification_batch, train=False)
            print('embeddings',embeddings.shape)#[20,256]
            enrollment_embeddings = flow.reshape(enrollment_embeddings,(hp.test.N, hp.test.M//2, embeddings.shape[1])) #[4,6//2,256]
            verification_embeddings = flow.reshape(verification_embeddings,(hp.test.N, hp.test.M//2, embeddings.shape[1])) #[4,6//2,256]

            enrollment_centroids = get_centroids(enrollment_embeddings)
            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)
        return sim_matrix

    # check_point = flow.train.CheckPoint()
    # check_point.load(model_path)
    flow.load_variables(flow.checkpoint.get(model_path))
    
    avg_EER = 0
    num_speaker = len(os.listdir(test_dataset.path))

    for e in range(hp.test.epochs):
        test_index = np.arange(num_speaker)
        np.random.shuffle(test_index) 
        print(test_index)
        batch_avg_EER = 0
        for batch_id in range(num_speaker//hp.test.N): 
            for i in range(hp.test.N):
                mel_db = test_dataset[test_index[batch_id*hp.test.N+i]]
                if i==0:
                    enrollment_batch = mel_db[0:hp.test.M//2,:,:]
                    verification_batch = mel_db[hp.test.M//2:,:,:]
                else:    
                    enrollment_batch = np.concatenate((mel_db[0:hp.test.M//2,:,:],enrollment_batch),axis=0)
                    verification_batch = np.concatenate((mel_db[hp.test.M//2:,:,:],verification_batch),axis=0)
        
            sim_matrix = predict(enrollment_batch,verification_batch)
            
            
            # calculating EER
            diff = 1; EER=0; EER_thresh = 0; EER_FAR=0; EER_FRR=0
            
            for thres in [0.01*i+0.5 for i in range(50)]:
                sim_matrix_thresh = sim_matrix>thres
                
                FAR = (sum([sim_matrix_thresh[i].sum()-sim_matrix_thresh[i,:,i].sum() for i in range(int(hp.test.N))])
                /(hp.test.N-1.0)/(float(hp.test.M/2))/hp.test.N)
    
                FRR = (sum([hp.test.M/2-sim_matrix_thresh[i,:,i].sum() for i in range(int(hp.test.N))])
                /(float(hp.test.M/2))/hp.test.N)
                
                # Save threshold when FAR = FRR (=EER)
                if diff> abs(FAR-FRR):
                    diff = abs(FAR-FRR)
                    EER = (FAR+FRR)/2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)"%(EER,EER_thresh,EER_FAR,EER_FRR))
        print('batch id',batch_id)
        avg_EER += batch_avg_EER/(batch_id+1)
    avg_EER = avg_EER / hp.test.epochs
    print("\n EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))
        
if __name__=="__main__":
    if hp.training:
        train(hp.model.model_path)
    else:
        test(hp.model.model_path)
