from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
from datetime import datetime

import test_global_storage
import oneflow as flow
import numpy as np
np.set_printoptions(suppress=True)

def _FullyConnected(input_blob,weight_blob,bias_blob):
    x = flow.matmul
    output_blob = x(input_blob, weight_blob)
    if bias_blob:
        output_blob = flow.nn.bias_add(output_blob, bias_blob)
    return output_blob


def lstm(input,units,return_sequence=False,initial_state=None,direction='forward',layer_index=0,init=flow.xavier_normal_initializer()):
    '''
       input: sequence input tensor with shape [batch_size,sequence_length,embedding size]
       units: hidden units numbers
    '''
    batch_size=input.shape[0]
    seq_len=input.shape[1]
    input_size = input.shape[2]
    

    trainable = True
    dtype = flow.float32
    with flow.scope.namespace('layer'+str(layer_index)):
        with flow.scope.namespace(direction):
            weight_blob_i = flow.get_variable(
                name='input' + '-weight',
                shape=[input_size, units],
                dtype=dtype,
                trainable=trainable,
                initializer=init)

            weight_blob_ih = flow.get_variable(
                name='input' + '-h-weight',
                shape=[units, units],
                dtype=dtype,
                trainable=trainable,
                initializer=init)

            bias_blob_i = flow.get_variable(
                name='input' + '-bias',
                shape=[units],
                dtype=dtype,
                trainable=trainable,
                initializer=flow.constant_initializer(0.0))

            weight_blob_f = flow.get_variable(
                name='forget' + '-weight',
                shape=[input_size, units],
                dtype=dtype,
                trainable=trainable,
                initializer=init)

            weight_blob_fh = flow.get_variable(
                name='forget' + '-h-weight',
                shape=[units, units],
                dtype=dtype,
                trainable=trainable,
                initializer=init)

            bias_blob_f = flow.get_variable(
                name='forget' + '-bias',
                shape=[units],
                dtype=dtype,
                trainable=trainable,
                initializer=flow.constant_initializer(0.0))

            weight_blob_c = flow.get_variable(
                name='cell' + '-weight',
                shape=[input_size, units],
                dtype=dtype,
                trainable=trainable,
                initializer=init)

            weight_blob_ch = flow.get_variable(
                name='cell' + '-h-weight',
                shape=[units, units],
                dtype=dtype,
                trainable=trainable,
                initializer=init)

            bias_blob_c = flow.get_variable(
                name='cell' + '-bias',
                shape=[units],
                dtype=dtype,
                trainable=trainable,
                initializer=flow.constant_initializer(0.0))

            weight_blob_o = flow.get_variable(
                name='output' + '-weight',
                shape=[input_size, units],
                dtype=dtype,
                trainable=trainable,
                initializer=init)

            weight_blob_oh = flow.get_variable(
                name='output' + '-h-weight',
                shape=[units, units],
                dtype=dtype,
                trainable=trainable,
                initializer=init)

            bias_blob_o = flow.get_variable(
                name='output' + '-bias',
                shape=[units],
                dtype=dtype,
                trainable=trainable,
                initializer=flow.constant_initializer(0.0))
    

    def step_function(input,states):
        
        if input.split_axis >= 0:
            hx = flow.parallel_cast(states[0], distribute=flow.distribute.split(input.split_axis))
            cx = flow.parallel_cast(states[1], distribute=flow.distribute.split(input.split_axis))
        else:
            hx=states[0]
            cx=states[1]

        x_i = _FullyConnected(input,weight_blob_i,bias_blob_i) # input gate
        mark_int=x_i
        x_f = _FullyConnected(input,weight_blob_f,bias_blob_f) # forget gate
        x_c = _FullyConnected(input,weight_blob_c,bias_blob_c) # cell state
        x_o = _FullyConnected(input,weight_blob_o,bias_blob_o) # output gate

        h_i = _FullyConnected(hx,weight_blob_ih,None)
        h_f = _FullyConnected(hx,weight_blob_fh,None)
        h_c = _FullyConnected(hx,weight_blob_ch,None)
        h_o = _FullyConnected(hx,weight_blob_oh,None)
        #print(f"before lstm,data shape:{h_i.shape},split axis:{h_i.split_axis},parallel size:{h_i.parallel_size}") 

        x_i = x_i + h_i
        x_f = x_f+h_f
        x_c = x_c+h_c
        x_o = x_o+h_o

        x_i = flow.math.sigmoid(x_i)
        x_f = flow.math.sigmoid(x_f)
        cellgate = flow.math.tanh(x_c)
        x_o = flow.math.sigmoid(x_o)

        cy = x_f * cx + x_i * cellgate

        hy = x_o * flow.math.tanh(cy)

        return hy, (hy,cy)

    if initial_state:
        states=initial_state
    else:
        states=[flow.constant(0, dtype=flow.float32, shape=[batch_size,units]),flow.constant(0, dtype=flow.float32, shape=[batch_size,units])]
        
    
    successive_outputs=[]
    successive_states= []

    for index in range(seq_len):
        #print('time step:',index)
        
        inp = flow.slice(input, [None, index, 0], [None, 1, input_size])
        #print(f"before lstm,data shape:{inp.shape},split axis:{inp.split_axis},parallel size:{inp.parallel_size}") 
        #print(inp.shape)
        inp = flow.reshape(inp, [-1, input_size])
        #print(f"before lstm,data shape:{inp.shape},split axis:{inp.split_axis},parallel size:{inp.parallel_size}") 
        #print(inp.shape)
        output, states = step_function(inp, states)

        output = flow.reshape(output,[-1,1,units])
        successive_outputs.append(output)
        successive_states.append(states)
        #print(f"before lstm,data shape:{output.shape},split axis:{output.split_axis},parallel size:{output.parallel_size}")
    last_output = successive_outputs[-1]
    new_states = successive_states[-1]
    outputs = flow.concat(successive_outputs,axis=1)
    #print(f"before lstm,data shape:{outputs.shape},split axis:{outputs.split_axis},parallel size:{outputs.parallel_size}")    


    if return_sequence:
        return outputs
    else:
        return flow.reshape(last_output,[-1,units]) 

def Blstm(input,units,return_sequence=True,initial_state=None,layer_index=0):
    # return_sequence should be True for BLSTM currently
    # default concat method : add

    forward = lstm(input,units,return_sequence=return_sequence,initial_state=initial_state,direction='forward',layer_index=layer_index)

    reverse_input = flow.reverse(input,axis=1)
    backward = lstm(reverse_input,units,return_sequence=return_sequence,initial_state=initial_state,direction='backward',layer_index=layer_index)    
    backward = flow.reverse(backward,axis=1)

    outputs = forward + backward

    return outputs 

