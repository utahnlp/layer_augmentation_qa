<p align="center"><img width="80%" src="logo.png" /></p>

Implementation of the **machine comprehension model** in our ACL 2019 paper: [Augmenting Neural Networks with First-order Logic](https://arxiv.org/abs/1906.06298)
```
@inproceedings{li2019augmenting,
      author    = {Li, Tao and Srikumar, Vivek},
      title     = {Augmenting Neural Networks with First-order Logic},
      booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
      year      = {2019}
  }
```

**For the NLI model, check out [here](https://github.com/utahnlp/layer_augmentation).**

### 0. Prerequisites
Have the following installed:
```
python 3.6+
pytorch 1.2.0
spacy 2.0.11 with en model
nltk with punkt
h5py
numpy
allennlp (install from src if to use ELMo)
```

A few other things should be downloaded and unpacked into ```./data```:
```
/squad-v1.1/train-v1.1.txt
/squad-v1.1/dev-v1.1.txt
glove.840B.300d.txt
```
SQuAD v1.1 will be the dataset to work on here. 

Also unzip the ```./data/conceptnet_rel.zip```. The files should be directly located under ```./data/``` without any intermediate folder.

## 1. Preprocessing
Preprocessing is separated into the following steps.

First extract something out of the json files. Assume the data is downloaded and unpacked into ```data/squad-v1.1/```
```
python3 squad_extraction.py --dir data/squad-v1.1/ --data train-v1.1.json --output train
python3 squad_extraction.py --dir data/squad-v1.1/ --data dev-v1.1.json --output dev
```
Then batch examples and vectorize them:
```
python3 preprocess.py --dir data/squad-v1.1/ --glove data/glove.840B.300d.txt --batchsize 15 --lowercase 0 --outputfile squad
python3 get_pretrain_vecs.py --dir data/squad-v1.1/ --glove data/glove.840B.300d.txt --dict squad.word.dict --output glove
python3 get_char_idx.py --dir data/squad-v1.1/ --dict squad.allword.dict --freq 49 --output char
```
There will be 6039 batches in train set and 284 characters extracted.

### 1.1 Caching ELMo
The following instructions are used for caching ELMo embeddings for much faster training and evaluation.\
But this process requires 200+GB cache disk space. 

```
python3 -u elmo_batch_preprocess.py --gpuid [GPUID] --src data/squad-v1.1/dev.context.txt --tgt data/squad-v1.1/dev.query.txt --batched data/squad-v1.1/squad-val.hdf5 --output data/squad-v1.1/dev
python3 -u elmo_batch_preprocess.py --gpuid [GPUID] --src data/squad-v1.1/train.context.txt --tgt data/squad-v1.1/train.query.txt --batched data/squad-v1.1/squad-train.hdf5 --output data/squad-v1.1/train
```
where ```[GPUID]``` is the target GPU id to use (one only). Set to ```-1``` to run on CPU.
The result ELMo cache embeddings will be ```data/squad-v1.1/dev.elmo.hdf5``` and ```data/squad-v1.1/train.elmo.hdf5```.


## 2. BiDAF
To train BiDAF (w/o ELMo) baseline, use this:
```
mkdir ./models

GPUID=[GPUID]
PERC=[PERC]
SEED=[SEED]
python3 -u train.py --gpuid $GPUID --dir data/squad-v1.1/ \
--train_res train.raw_context.txt,train.raw_answer.txt,train.token_span.txt \
--word_vec_size 300 --num_char 284 \
--optim adam --adam_betas 0.9,0.999 --ema 1 --use_elmo_post 0 --learning_rate 0.001 --clip 5 --epochs 20 \
--enc encoder --att biatt --reenc reencoder --reenc_rnn_layer 2 --cls boundary --acc_batch_size 50 --hidden_size 100 \
--percent $PERC --seed $SEED \
--save_file models/bidaf_adam_lr0001_perc${PERC//.}_seed${SEED} | tee models/bidaf_adam_lr0001_perc${PERC//.}_seed${SEED}.txt
```
where ```[GPUID]``` is the GPU you want to use (one only).\
```[SEED]``` is the randomness for data split and shuffling.\
```[PERC]``` is the sampling ratio (e.g. 0.1 for 10%).

To evaluate the trained model:
```
GPUID=[GPUID]
PERC=[PERC]
SEED=[SEED]
python3 -u eval.py --gpuid $GPUID --dir data/squad-v1.1/ \
--res dev.raw_context.txt,dev.raw_answer.txt,dev.token_span.txt,dev.raw_query.txt \
--num_char 284 --enc encoder --att biatt --reenc reencoder --reenc_rnn_layer 2 --cls boundary --use_elmo_post 0 \
--verbose 1 --print print \
--load_file models/bidaf_adam_lr0001_perc${PERC//.}_seed${SEED} | tee models/bidaf_adam_lr0001_perc${PERC//.}_seed${SEED}.evallog.txt
```
With full data, expect to see validation F1 around ```76```.

### Augmented Models
To train BiDAF (w/o ELMo) with augmented attention layer:
```
GPUID=[GPUID]
CONSTR_W=a8
RHO_W=[RHO_W]
PERC=[PERC]
SEED=[SEED]
python3 -u train.py --gpuid $GPUID --dir data/squad-v1.1/ \
	--train_res train.raw_context.txt,train.raw_answer.txt,train.token_span.txt,train.all_rel.json,train.content_word.json \
	--num_char 284 --ema 1 --use_elmo_post 0 --learning_rate 0.001 --clip 5 --epochs 20 \
	--enc encoder --att biatt --reenc reencoder --reenc_rnn_layer 2 --cls boundary --percent ${PERC} --div_percent 0.9 \
	--within_constr ${CONSTR_W} --rho_w ${RHO_W} \
	--seed ${SEED} --save_file models/${CONSTR_W//.}_rho${RHO_W}_dev_lr0001_perc${PERC//.}_seed${SEED} | tee models/${CONSTR_W//.}_rho${RHO_W}_dev_lr0001_perc${PERC//.}_seed${SEED}.txt
```
Constraint ```a8``` is the conservative constraint in our paper while ```a9``` is the normal constraint.\
```rho_w``` is the impact scaling factor for the constraint.

To evaluate the trained model:
```
GPUID=[GPUID]
CONSTR_W=a8
RHO_W=[RHO_W]
PERC=[PERC]
SEED=[SEED]
python3 -u eval.py --gpuid $GPUID --dir data/squad-v1.1/ \
	--res dev.raw_context.txt,dev.raw_answer.txt,dev.token_span.txt,dev.raw_query.txt,dev.all_rel.json,dev.content_word.json \
	--num_char 284 --enc encoder --att biatt --reenc reencoder --reenc_rnn_layer 2 --cls boundary --use_elmo_post 0 \
	--within_constr ${CONSTR_W} --rho_w ${RHO_W} \
	--load_file models/${CONSTR_W//.}_rho${RHO_W}_dev_lr0001_perc${PERC//.}_seed${SEED} | tee models/${CONSTR_W//.}_rho${RHO_W}_dev_lr0001_perc${PERC//.}_seed${SEED}.evallog.txt
```
The evaluation performances should match the numbers reported in our paper.

## 3. BIDAF+ELMo
Here, we show how to train the improved BIDAF+ELMo model (from the NAACL18 ELMo paper):
```
GPUID=[GPUID]
PERC=[PERC]
SEED=[SEED]
LR=0.0002
python3 -u train.py --gpuid $GPUID --dir data/squad-v1.1/ --train_data squad-train.hdf5 \
	--train_res train.context.txt,train.query.txt,train.raw_context.txt,train.raw_answer.txt,train.token_span.txt,train.elmo.hdf5 \
	--word_vec_size 300 --num_char 284 --optim adam --adam_betas 0.9,0.999 --elmo_in_size 1024 --elmo_size 1024  --clip 5 \
	--rnn_type gru --ema 1 --learning_rate $LR --dropout 0.2 --elmo_dropout 0.5 --epochs 25 --div_percent 0.9 --percent $PERC \
	--enc encoder_with_elmo --att biatt --reenc match --cls boundary_chain --hidden_size 100 --fix_elmo 1 --reenc_rnn_layer 1 \
	--acc_batch_size 50 --seed $SEED \
	--save_file models/elmo_adam_lr${LR//.}_perc${PERC//.}_seed${SEED} | tee models/elmo_adam_lr${LR//.}_perc${PERC//.}_seed${SEED}.txt
```

To evaluate the trained model:
```
GPUID=[GPUID]
PERC=[PERC]
SEED=[SEED]
LR=0.0002
python3 -u eval.py --gpuid 0 --res dev.raw_context.txt,dev.raw_query.txt,dev.raw_answer.txt,dev.token_span.txt,dev.elmo.hdf5 \
--word_vec_size 300 --num_char 284 --rnn_type gru --elmo_dropout 0.0 --char_dropout 0.0 --dropout 0.0 \
--enc encoder_with_elmo --att biatt --reenc match --reenc_rnn_layer 1 --cls boundary_chain \
--load_file ./models/elmo_adam_lr${LR//.}_perc${PERC//.}_seed${SEED} | tee ./models/elmo_adam_lr${LR//.}_perc${PERC//.}_seed${SEED}.evallog.txt
```
With full data, expect to see validation F1 of around ```85```.

### Augmented Models
To train BiDAF+ELMo with augmented attention layer:
```
GPUID=[GPUID]
PERC=[PERC]
SEED=[SEED]
LR=0.0002
CONSTR_W=a8
RHO_W=[RHO_W]
SEED=[SEED]
python3 -u train.py --gpuid $GPUID --dir data/squad-v1.1/ --train_data squad-train.hdf5 \
	--train_res train.context.txt,train.query.txt,train.raw_context.txt,train.raw_answer.txt,train.token_span.txt,train.all_rel.json,train.content_word.json,train.elmo.hdf5 \
	--word_vec_size 300 --num_char 284 --optim adam --adam_betas 0.9,0.999 --elmo_in_size 1024 --elmo_size 1024 --clip 5 \
	--rnn_type gru --ema 1 --learning_rate $LR --dropout 0.2 --elmo_dropout 0.5 --epochs 25 --div_percent 0.9 --percent ${PERC} \
	--enc encoder_with_elmo --att biatt --reenc match --cls boundary_chain --hidden_size 100 --fix_elmo 1 --reenc_rnn_layer 1 \
	--within_constr ${CONSTR_W} --rho_w ${RHO_W} --acc_batch_size 50 --seed $SEED \
	--save_file models/${CONSTR_W}_rho${RHO_W}_elmo_adam_lr${LR//.}_perc${PERC//.}_seed${SEED} | tee models/${CONSTR_W}_rho${RHO_W}_elmo_adam_lr${LR//.}_perc${PERC//.}_seed${SEED}.txt
```

To evaluate the trained model:
```
GPUID=[GPUID]
PERC=[PERC]
SEED=[SEED]
LR=0.0002
CONSTR_W=a8
RHO_W=[RHO_W]
python3 -u eval.py --gpuid $GPUID --res dev.raw_context.txt,dev.raw_query.txt,dev.raw_answer.txt,dev.token_span.txt,dev.all_rel.json,dev.content_word.json,dev.elmo.hdf5, \
	--word_vec_size 300 --num_char 284 --rnn_type gru --elmo_dropout 0.0 --char_dropout 0.0 --dropout 0.0 \
	--enc encoder_with_elmo --att biatt --reenc match --reenc_rnn_layer 1 --cls boundary_chain --dynamic_elmo 0 \
	--within_constr ${CONSTR_W} --rho_w ${RHO_W} \
	--load_file ./models/${CONSTR_W}_rho${RHO_W}_elmo_adam_lr${LR//.}_perc${PERC//.}_seed${SEED} | tee ./models/${CONSTR_W}_rho${RHO_W}_elmo_adam_lr${LR//.}_perc${PERC//.}_seed${SEED}.evallog.txt
```
The evaluation performances should match the numbers reported in our paper.


