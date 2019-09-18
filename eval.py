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
from embeddings import *
from data import *
from boundary_loss import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="data/squad-v1.1/")
parser.add_argument('--data', help="Path to data hdf5 file.", default="squad-val.hdf5")
parser.add_argument('--load_file', help="Path from where model to be loaded.", default="")
parser.add_argument('--word_vecs', help="The path to word embeddings", default = "glove.hdf5")
parser.add_argument('--char_idx', help="The path to word2char index file", default = "char.idx.hdf5")
parser.add_argument('--dict', help="The path to word dictionary", default = "squad.word.dict")
parser.add_argument('--char_dict', help="The path to char dictionary", default = "char.dict.txt")
parser.add_argument('--res', help="Path to resource files, seperated by comma.", default="")
#
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
parser.add_argument('--hidden_size', help="The general hidden size of the pipeline", type=int, default=100)
parser.add_argument('--token_l', help="The maximal token length", type=int, default=16)
parser.add_argument('--char_dropout', help="The dropout probability on char encoder", type=float, default=0.0)
parser.add_argument('--elmo_dropout', help="The dropout probability on ELMO", type=float, default=0.0)
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.0)
#parser.add_argument('--dynamic_elmo', help="Whether to use elmo model to parse text dynamically, or use cached ELMo", type=int, default=0)
parser.add_argument('--fix_elmo', help="Whether to make ELMo model NOT learnable", type=int, default=1)
#
parser.add_argument('--enc', help="The type of encoder, encoder/encoder_with_elmo", default='encoder')
parser.add_argument('--att', help="The type of biattention, biattention", default='biatt')
parser.add_argument('--reenc', help="The type of reencoder, reencoder/match", default='reencoder')
parser.add_argument('--self_att', help="The type of self attention, self_att", default='self_att')
parser.add_argument('--cls', help="The type of classifier, boundary", default='boundary')
parser.add_argument('--loss', help="The type of loss, boundary", default='boundary')
# TODO, param_init of uniform dist or normal dist???
parser.add_argument('--fix_word_vecs', help="Whether to make word embeddings NOT learnable", type=int, default=1)
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--enc_rnn_layer', help="The number of layers of rnn encoder", type=int, default=1)
parser.add_argument('--reenc_rnn_layer', help="The number of layers of rnn reencoder", type=int, default=1)
parser.add_argument('--cls_rnn_layer', help="The number of layers of classifier rnn", type=int, default=1)
parser.add_argument('--num_cls_pass', help="The number passes in multipass classifier", type=int, default=1)
parser.add_argument('--birnn', help="Whether to use bidirectional rnn", type=int, default=1)
parser.add_argument('--rnn_type', help="The type of rnn to use (lstm or gru)", default='lstm')
parser.add_argument('--hw_layer', help="The number of highway layers to use", type=int, default=2)
parser.add_argument('--span_l', help="The maximal span length allowed for prediction", type=int, default=17)
# constraint
parser.add_argument('--rho_w', help="The weight of within layer struct attention penalty", type=float, default=1.0)
parser.add_argument('--constr_on', help="Directions of attentions to apply constraints on", default='1')
parser.add_argument('--within_constr', help="The list of att constraint layers to use, no if empty", default="")
parser.add_argument('--fix_rho', help="Whether to fix rho", type=int, default=1)
# printing
parser.add_argument('--verbose', help="Whether to print out every prediction", type=int, default=0)
parser.add_argument('--print', help="Prefix to where verbose printing will be piped", default='print')

def evaluate(opt, shared, m, data):
	m.train(False)

	val_loss = 0.0
	num_ex = 0
	verbose = opt.verbose==1
	loss = None
	if opt.loss == 'boundary':
		loss = BoundaryLoss(opt, shared)
	else:
		assert(False)

	loss.verbose = verbose

	m.begin_pass()
	for i in range(data.size()):
		(data_name, source, target, char_source, char_target, 
			batch_ex_idx, batch_l, source_l, source_sent_l, target_l, span, res_map) = data[i]

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

	return (perf, extra_perf, val_loss / num_ex, num_ex)



def main(args):
	opt = parser.parse_args(args)
	shared = Holder()

	# 
	opt.data = opt.dir + opt.data
	opt.res = '' if opt.res == ''  else ','.join([opt.dir + path for path in opt.res.split(',')])
	opt.word_vecs = opt.dir + opt.word_vecs
	opt.char_idx = opt.dir + opt.char_idx
	opt.dict = opt.dir + opt.dict
	opt.char_dict = opt.dir + opt.char_dict

	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(1)

	# build model
	m = Pipeline(opt, shared)

	# initialization
	print('loading pretrained model from {0}...'.format(opt.load_file))
	param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
	m.set_param_dict(param_dict)

	if opt.gpuid != -1:
		m = m.cuda()

	# loading data
	res_files = None if opt.res == '' else opt.res.split(',')
	data = Data(opt, opt.data, res_files)

	#
	perf, extra_perf, avg_loss, num_ex = evaluate(opt, shared, m, data)
	extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_perf])
	print('Val {0:.4f} Extra {1} Loss: {2:.4f}'.format(
		perf, extra_perf_str, avg_loss))

	#print('saving model to {0}'.format('tmp'))
	#param_dict = m.get_param_dict()
	#save_param_dict(param_dict, '{0}.hdf5'.format('tmp'))


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
