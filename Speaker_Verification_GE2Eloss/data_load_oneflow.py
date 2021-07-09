#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:55:52 2018

@author: harry
"""
import glob
import numpy as np
import os
import random
from random import shuffle
import librosa
import numpy as np

from hparam import hparam as hp

def mfccs_and_spec(wav_file, wav_process = False, calc_mfccs=False, calc_mag_db=False):    
    sound_file, _ = librosa.core.load(wav_file, sr=hp.data.sr)
    window_length = int(hp.data.window*hp.data.sr)
    hop_length = int(hp.data.hop*hp.data.sr)
    duration = hp.data.tisv_frame * hp.data.hop + hp.data.window
    
    # Cut silence and fix length
    if wav_process == True:
        sound_file, index = librosa.effects.trim(sound_file, frame_length=window_length, hop_length=hop_length)
        length = int(hp.data.sr * duration)
        sound_file = librosa.util.fix_length(sound_file, length)
        
    spec = librosa.stft(sound_file, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
    mag_spec = np.abs(spec)
    
    mel_basis = librosa.filters.mel(hp.data.sr, hp.data.nfft, n_mels=hp.data.nmels)
    mel_spec = np.dot(mel_basis, mag_spec)
    
    mag_db = librosa.amplitude_to_db(mag_spec)
    #db mel spectrogram
    mel_db = librosa.amplitude_to_db(mel_spec).T
    
    mfccs = None
    if calc_mfccs:
        mfccs = np.dot(librosa.filters.dct(40, mel_db.shape[0]), mel_db).T
    
    return mfccs, mel_db, mag_db

class SpeakerDatasetTIMIT():
    
    def __init__(self):

        if hp.training:
            self.path = hp.data.train_path_unprocessed
            self.utterance_number = hp.train.M
        else:
            self.path = hp.data.test_path_unprocessed
            self.utterance_number = hp.test.M
        self.speakers = glob.glob(os.path.dirname(self.path))
        shuffle(self.speakers)
        
    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):
        
        speaker = self.speakers[idx]
        wav_files = glob.glob(speaker+'/*.WAV')
        shuffle(wav_files)
        wav_files = wav_files[0:self.utterance_number]
        
        mel_dbs = []
        for f in wav_files:
            _, mel_db, _ = mfccs_and_spec(f, wav_process = True)
            mel_dbs.append(mel_db)
        return mel_dbs

class SpeakerDatasetTIMITPreprocessed():
    
    def __init__(self, shuffle=True, utter_start=0):
        
        # data path
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M
        self.file_list = os.listdir(self.path)
        self.shuffle=shuffle
        self.utter_start = utter_start
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        np_file_list = os.listdir(self.path)
        
        if self.shuffle:
            selected_file = random.sample(np_file_list, 1)[0]  # select random speaker
        else:
            selected_file = np_file_list[idx]               
        
        utters = np.load(os.path.join(self.path, selected_file))        # load utterance spectrogram of selected speaker
        if self.shuffle:
            utter_index = np.random.randint(0, utters.shape[0], self.utter_num)   # select M utterances per speaker
            utterance = utters[utter_index]       
        else:
            utterance = utters[self.utter_start: self.utter_start+self.utter_num] # utterances of a speaker [batch(M), n_mels, frames]

        utterance = utterance[:,:,:160]               # TODO implement variable length batch size

        utterance = np.transpose(utterance, axes=(0,2,1))     # transpose [batch, frames, n_mels]
        return utterance

