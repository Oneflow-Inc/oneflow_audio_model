#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 5 20:58:34 2021

@author: fujiaqing
"""

import oneflow as flow

from lstm import lstm

from hparam import hparam as hp


def norm(x):
    _square_x = flow.math.square(x, name="square")
    _sum_x = flow.math.reduce_sum(
        _square_x, axis=1, keepdims=False, name="sum"
    )
    _sqrt_x = flow.math.sqrt(_sum_x, name="sqrt")
    _sqrt_x = flow.expand_dims(_sqrt_x, axis=1)
    return _sqrt_x

def SpeechEmbedder(data, train=False):
    print(f"before lstm,data shape:{data.shape},split axis:{data.split_axis},parallel size:{data.parallel_size}")
    for i in range(hp.model.num_layer):
        data=lstm(data,hp.model.hidden,return_sequence=True,layer_index=i)

    print(f"before slice,data shape:{data.shape},split axis:{data.split_axis},parallel size:{data.parallel_size}")    
    data = flow.slice(data, [None, data.shape[1]-1, 0], [None, 1, data.shape[2]]) #actually return_sequence==False
    print(f"before reshape,data shape:{data.shape},split axis:{data.split_axis},parallel size:{data.parallel_size}")   
    data = flow.reshape(data,shape=[data.shape[0],data.shape[2]])

    initializer = flow.truncated_normal(0.1)
    print(f"before dense,data shape:{data.shape},split axis:{data.split_axis},parallel size:{data.parallel_size}")   
    data = flow.layers.dense(data, hp.model.proj, kernel_initializer=initializer, name="dense_1")  #[20,256]

    normed = norm(data)
    
    data = data / normed
    print(f"final,data shape:{data.shape},split axis:{data.split_axis},parallel size:{data.parallel_size}")   #[20,256]
    return data

    
