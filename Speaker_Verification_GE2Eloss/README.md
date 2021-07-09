# Oneflow_Speaker_Verification

oneflow implementation of speech embedding net and loss described here: https://arxiv.org/pdf/1710.10467.pdf.

The TIMIT speech corpus was used to train the model, found here: https://catalog.ldc.upenn.edu/LDC93S1,
or here, https://github.com/philipperemy/timit

[reference](https://github.com/HarryVolek/PyTorch_Speaker_Verification)

# Dependencies

* Oneflow 0.5.0
* python 3.5+
* numpy 1.15.4
* librosa 0.6.1
* PyTorch 1.8.1

# Preprocessing

Change the following config.yaml key to a regex containing all .WAV files in your downloaded TIMIT dataset. 
```yaml
unprocessed_data: './TIMIT/*/*/*/*.wav'
```
Run the preprocessing script:
```
./data_preprocess.py 
```
Two folders will be created, train_tisv and test_tisv, containing .npy files containing numpy ndarrays of speaker utterances with a 90%/10% training/testing split.

# Training

To train the speaker verification model, run:
```
./train_speech_embedder_oneflow.py 
```
with the following config.yaml key set to true:
```yaml
training: !!bool "true"
```
for testing, set the key value to:
```yaml
training: !!bool "false"
```
The log file and checkpoint save locations are controlled by the following values:
```yaml
log_file: './speech_id_checkpoint/Stats'
checkpoint_dir: './speech_id_checkpoint'
```
Only TI-SV is implemented.

# Performance

```
EER with oneflow: 0.0460
EER with pytorch: 0.0578 
```
Note : despite this code claim the EER 0.0377 , I can not reproduce the [result](https://github.com/HarryVolek/PyTorch_Speaker_Verification/issues/76)
