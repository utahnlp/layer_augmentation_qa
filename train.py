import sys
from pipeline import *
import argparse
import h5py
import os
import random
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch import cuda
from holder import *
from optimizer import *
from data import *
from util import *
from ema import *
from boundary_loss import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="data/squad-v1.1/")
parser.add_argument('--train_data', help="Path to training data hdf5 file.", default="squad-train.hdf5")
parser.add_argument('--val_data', help="Path to validation data hdf5 file.", default="")
parser.add_argument('--save_file', help="Path to where model to be saved.", default="model")
parser.add_argument('--word_vecs', help="The path to word embeddings", default = "glove.hdf5")
parser.add_argument('--char_idx', help="The path to word2char index file", default = "char.idx.hdf5")
parser.add_argument('--dict', help="The path to word dictionary", default = "squad.word.dict")
parser.add_argument('--char_dict', help="The path to char dictionary", default = "char.dict.txt")
# resource specs
parser.add_argument('--train_res', help="Path to training resource files, seperated by comma.", default="")
parser.add_argument('--val_res', help="Path to validation resource files, seperated by comma.", default="")
## pipeline specs
parser.add_argument('--use_char_enc', help="Whether to use char encoding", type=int, default=1)
parser.add_argument('--char_encoder', help="The type of char encoder, cnn/rnn", default='cnn')
parser.add_argument('--char_filters', help="The list of filters for char cnn", default='5')
parser.add_argument('--num_char', help="The number of distinct chars", type=int, default=284)
parser.add_argument('--char_emb_size', help="The input char embedding dim", type=int, default=20)
parser.add_argument('--char_enc_size', help="The input char encoding dim", type=int, default=100)
parser.add_argument('--word_vec_size', help="The input word embedding dim", type=int, default=300)
parser.add_argument('--elmo_in_size', help="The input elmo dim", type=int, default=1024)
parser.add_argument('--elmo_size', help="The hidden elmo dim", type=int, default=1024)
parser.add_argument('--elmo_top_only', help="Whether to use elmo top layer only", type=int, default=0)
parser.add_argument('--use_elmo_post', help="Whether to use elmo after encoder", type=int, default=1)
#parser.add_argument('--dynamic_elmo', help="Whether to use elmo model to parse text dynamically, or use cached ELMo", type=int, default=0)
parser.add_argument('--hidden_size', help="The general hidden size of the pipeline", type=int, default=100)
parser.add_argument('--token_l', help="The maximal token length", type=int, default=16)
parser.add_argument('--elmo_dropout', help="The dropout probability on ELMO", type=float, default=0.5)
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.2)
parser.add_argument('--percent', help="The percent of training data to use", type=float, default=1.0)
parser.add_argument('--div_percent', help="The percent of training data to divide as train/val", type=float, default=0.9)
parser.add_argument('--epochs', help="The number of epoches for training", type=int, default=30)
parser.add_argument('--optim', help="The name of optimizer to use for training", default='adam')
parser.add_argument('--learning_rate', help="The learning rate for training", type=float, default=0.001)
parser.add_argument('--clip_epoch', help="The starting epoch to enable clip", type=int, default=1)
parser.add_argument('--clip', help="The norm2 threshold to clip, set it to negative to disable", type=float, default=-1.0)
parser.add_argument('--mu', help="The mu ratio used in EMA", type=float, default=0.999)
parser.add_argument('--adam_betas', help="The betas used in adam", default='0.9,0.999')
## pipeline stages
parser.add_argument('--enc', help="The type of encoder, encoder/encoder_with_elmo", default='encoder')
parser.add_argument('--att', help="The type of biattention, biattention", default='biatt')
parser.add_argument('--reenc', help="The type of reencoder, reencoder/match", default='reencoder')
parser.add_argument('--self_att', help="The type of self attention, self_att", default='self_att')
parser.add_argument('--cls', help="The type of classifier, boundary", default='boundary')
parser.add_argument('--loss', help="The type of loss, boundary", default='boundary')

parser.add_argument('--param_init_type', help="The type of parameter initialization", default='xavier_normal')
parser.add_argument('--fix_word_vecs', help="Whether to make word embeddings NOT learnable", type=int, default=1)
parser.add_argument('--fix_elmo', help="Whether to make ELMo model NOT learnable", type=int, default=1)
parser.add_argument('--elmo_cache_size', help="The size of elmo cache", type=int, default=1000)
parser.add_argument('--print_every', help="Print stats after this many batches", type=int, default=500)
parser.add_argument('--seed', help="The random seed", type=int, default=3435)
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--enc_rnn_layer', help="The number of layers of rnn encoder", type=int, default=1)
parser.add_argument('--reenc_rnn_layer', help="The number of layers of rnn reencoder", type=int, default=1)
parser.add_argument('--cls_rnn_layer', help="The number of layers of classifier rnn", type=int, default=1)
parser.add_argument('--num_cls_pass', help="The number passes in multipass classifier", type=int, default=1)
parser.add_argument('--birnn', help="Whether to use bidirectional rnn", type=int, default=1)
parser.add_argument('--rnn_type', help="The type of rnn to use (lstm or gru)", default='lstm')
parser.add_argument('--hw_layer', help="The number of highway layers to use", type=int, default=2)
parser.add_argument('--ema', help="Whether to use EMA", type=int, default=0)
parser.add_argument('--save_all_epochs', help="Whether to save models for all epochs", type=int, default=0)
parser.add_argument('--span_l', help="The maximal span length allowed for prediction", type=int, default=17)
parser.add_argument('--acc_batch_size', help="The accumulative batch size, -1 to disable", type=int, default=50)
# constraint
parser.add_argument('--rho_w', help="The weight of within layer struct attention penalty", type=float, default=1.0)
parser.add_argument('--constr_on', help="Directions of attentions to apply constraints on", default='1')
parser.add_argument('--within_constr', help="The list of att constraint layers to use, no if empty", default="")
parser.add_argument('--fix_rho', help="Whether to fix rho", type=int, default=1)


# train batch by batch, accumulate batches until the size reaches acc_batch_size
def train_epoch(opt, shared, m, optim, ema, data, epoch_id, sub_idx):
	train_loss = 0.0
	num_ex = 0
	start_time = time.time()
	train_idx1_correct = 0
	train_idx2_correct = 0
	min_grad_norm2 = 1000000000000.0
	max_grad_norm2 = 0.0
	total_em_bow = 0.0
	total_f1_bow = 0.0

	loss = None
	if opt.loss == 'boundary':
		loss = BoundaryLoss(opt, shared)
	else:
		assert(False)

	data_size = len(sub_idx)
	batch_order = torch.randperm(data_size)
	batch_order = [sub_idx[idx] for idx in batch_order]

	acc_batch_size = 0
	m.train(True)
	data.begin_pass()
	loss.begin_pass()
	m.begin_pass()
	for i in range(data_size):
		(data_name, source, target, char_source, char_target, 
			batch_ex_idx, batch_l, source_l, source_sent_l, target_l, span, res_map) = data[batch_order[i]]

		wv_idx1 = Variable(source, requires_grad=False)
		wv_idx2 = Variable(target, requires_grad=False)
		cv_idx1 = Variable(char_source, requires_grad=False)
		cv_idx2 = Variable(char_target, requires_grad=False)
		y_gold = Variable(span, requires_grad=False)

		# update network parameters
		shared.epoch = epoch_id
		m.update_context(batch_ex_idx, batch_l, source_l, source_sent_l, target_l, res_map)

		# forward pass
		output = m.forward(wv_idx1, wv_idx2, cv_idx1, cv_idx2)

		# loss
		batch_loss = loss(output, y_gold)

		# stats
		train_loss += float(batch_loss.data)
		num_ex += batch_l
		time_taken = time.time() - start_time
		acc_batch_size += batch_l

		# accumulate grads
		batch_loss.backward()

		# accumulate current batch until the rolled up batch size exceeds threshold or meet certain boundary
		if i == data_size-1 or acc_batch_size >= opt.acc_batch_size or (i+1) % opt.print_every == 0:
			grad_norm2 = optim.step(m, acc_batch_size)
			if opt.ema == 1:
				ema.step(m)

			# clear up grad
			m.zero_grad()
			acc_batch_size = 0

			# stats
			grad_norm2_avg = grad_norm2
			min_grad_norm2 = min(min_grad_norm2, grad_norm2_avg)
			max_grad_norm2 = max(max_grad_norm2, grad_norm2_avg)
			time_taken = time.time() - start_time
			loss_stats = loss.print_cur_stats()

			if (i+1) % opt.print_every == 0:
				stats = '{0}, Batch {1:.1f}k '.format(epoch_id+1, float(i+1)/1000)
				stats += 'Grad {0:.1f}/{1:.1f} '.format(min_grad_norm2, max_grad_norm2)
				stats += 'Loss {0:.4f} '.format(train_loss / num_ex)
				stats += loss.print_cur_stats()
				stats += 'Time {0:.1f}'.format(time_taken)
				print(stats)

	perf, extra_perf = loss.get_epoch_metric()

	m.end_pass()
	loss.end_pass()
	data.end_pass()

	return perf, extra_perf, train_loss / num_ex, num_ex


def train(opt, shared, m, optim, ema, train_data, val_data):
	best_val_perf = 0.0
	test_perf = 0.0
	train_perfs = []
	val_perfs = []
	extra_perfs = []

	print('{0} batches in train set'.format(train_data.size()))
	if val_data is not None:
		print('{0} batches in dev set'.format(val_data.size()))
	else:
		print('no dev set specified, will split train set into train/dev folds')

	print('subsampling train set by {0}'.format(opt.percent))
	train_idx, train_num_ex = train_data.subsample(opt.percent, random=True)
	print('for the record, first 10 batches: {0}'.format(train_idx[:10]))

	val_idx = None
	val_num_ex = 0
	if val_data is None:
		val_data = train_data
		print('splitting train set into train/dev folds by {0}'.format(opt.div_percent))
		train_idx, val_idx, train_num_ex, val_num_ex = train_data.split(train_idx, opt.div_percent)
	else:
		val_idx, val_num_ex = val_data.subsample(1.0, random=False)	# use all val data as dev set

	print('final train set has {0} batches {1} examples'.format(len(train_idx), train_num_ex))
	print('for the record, first 10 batches: {0}'.format(train_idx[:10]))
	print('final val set has {0} batches {1} examples'.format(len(val_idx), val_num_ex))
	print('for the record, first 10 batches: {0}'.format(val_idx[:10]))

	start = 0
	for i in range(start, opt.epochs):
		train_perf, extra_train_perf, loss, num_ex = train_epoch(opt, shared, m, optim, ema, train_data, i, train_idx)
		train_perfs.append(train_perf)
		extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_train_perf])
		print('Train {0:.4f} All {1}'.format(train_perf, extra_perf_str))

		# evaluate
		#	and save if it's the best model
		val_perf, extra_val_perf, val_loss, num_ex = validate(opt, shared, m, val_data, val_idx)
		val_perfs.append(val_perf)
		extra_perfs.append(extra_val_perf)
		extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_val_perf])
		print('Val {0:.4f} All {1}'.format(val_perf, extra_perf_str))

		perf_table_str = ''
		cnt = 0
		print('Epoch  | Train | Val ...')
		for train_perf, extra_perf in zip(train_perfs, extra_perfs):
			extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_perf])
			perf_table_str += '{0}\t{1:.4f}\t{2}\n'.format(cnt+1, train_perf, extra_perf_str)
			cnt += 1
		print(perf_table_str)

		if opt.save_all_epochs == 1:
			print('saving model to {0}.{1}'.format(opt.save_file, i))
			param_dict = m.get_param_dict()
			save_param_dict(param_dict, '{0}.{1}.hdf5'.format(opt.save_file, i))
			if opt.ema == 1:
				ema_param_dict = ema.get_param_dict()
				save_param_dict(ema_param_dict, '{0}.{1}.ema.hdf5'.format(opt.save_file, i))

		if val_perf > best_val_perf:
			best_val_perf = val_perf
			print('saving model to {0}'.format(opt.save_file))
			param_dict = m.get_param_dict()
			save_param_dict(param_dict, '{0}.hdf5'.format(opt.save_file))
			save_opt(opt, '{0}.opt'.format(opt.save_file))
			# save ema
			if opt.ema == 1:
				ema_param_dict = ema.get_param_dict()
				save_param_dict(ema_param_dict, '{0}.ema.hdf5'.format(opt.save_file))

		else:
			print('skip saving model for perf <= {0:.4f}'.format(best_val_perf))



def validate(opt, shared, m, val_data, val_idx):
	m.train(False)

	val_loss = 0.0
	num_ex = 0

	loss = None
	if opt.loss == 'boundary':
		loss = BoundaryLoss(opt, shared)
	else:
		assert(False)

	print('validating on the {0} batches...'.format(len(val_idx)))

	val_data.begin_pass()
	m.begin_pass()
	for i in range(len(val_idx)):

		(data_name, source, target, char_source, char_target, 
			batch_ex_idx, batch_l, source_l, source_sent_l, target_l, span, res_map) = val_data[val_idx[i]]

		wv_idx1 = Variable(source, requires_grad=False)
		wv_idx2 = Variable(target, requires_grad=False)
		cv_idx1 = Variable(char_source, requires_grad=False)
		cv_idx2 = Variable(char_target, requires_grad=False)
		y_gold = Variable(span, requires_grad=False)

		# update network parameters
		m.update_context(batch_ex_idx, batch_l, source_l, source_sent_l, target_l, res_map)

		# forward pass
		pred = m.forward(wv_idx1, wv_idx2, cv_idx1, cv_idx2)

		# loss
		batch_loss = loss(pred, y_gold)

		# stats
		val_loss += float(batch_loss.data)
		num_ex += batch_l

	perf, extra_perf = loss.get_epoch_metric()
	m.end_pass()
	val_data.end_pass()

	return (perf, extra_perf, val_loss / num_ex, num_ex)



def main(args):
	opt = parser.parse_args(args)
	shared = Holder()

	# 
	opt.train_data = opt.dir + opt.train_data
	opt.val_data = opt.dir + opt.val_data
	opt.train_res = '' if opt.train_res == ''  else ','.join([opt.dir + path for path in opt.train_res.split(',')])
	opt.val_res = '' if opt.val_res == ''  else ','.join([opt.dir + path for path in opt.val_res.split(',')])
	opt.word_vecs = opt.dir + opt.word_vecs
	opt.char_idx = opt.dir + opt.char_idx
	opt.dict = opt.dir + opt.dict
	opt.char_dict = opt.dir + opt.char_dict

	torch.manual_seed(opt.seed)
	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(opt.seed)

	print(opt)

	# build model
	m = Pipeline(opt, shared)
	optim = Optimizer(opt, shared)
	ema = EMA(opt, shared)

	m.init_weight()
	model_parameters = filter(lambda p: p.requires_grad, m.parameters())
	num_params = sum([np.prod(p.size()) for p in model_parameters])
	print('total number of trainable parameters: {0}'.format(num_params))
	if opt.gpuid != -1:
		m = m.cuda()

	# loading data
	train_data = Data(opt, opt.train_data, None if opt.train_res == '' else opt.train_res.split(','))

	val_data = None
	if opt.val_data != opt.dir:
		val_data = Data(opt, opt.val_data, None if opt.val_res == '' else opt.val_res.split(','))

	train(opt, shared, m, optim, ema, train_data, val_data)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
