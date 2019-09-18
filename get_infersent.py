import torch
from torch import nn
from torch import cuda
from models import InferSent
import h5py
import sys
import argparse
import numpy as np

##### PUT this file under InferSent repo!!!!
##### Otherwise, won't work!!!!

cuda.set_device(0)


def split_par(par):
	sents = par.strip().split('|||')
	sents = [s for s in sents if s.strip() != '']
	sents = [s.strip().split(' ') for s in sents]
	return sents

def load_sent(path):
	par = []
	with open(path, 'r+') as f:
		for l in f:
			if l.strip() == '':
				continue
			par.append(split_par(l.rstrip()))
	return par

def load_token(path):
	tokens = []
	with open(path, 'r+') as f:
		for l in f:
			if l.rstrip() == '':
				continue
			tokens.append(l.strip().split(' '))
	return tokens

# input context is a list of pars, par is a list of sents, sent is a list of tokens
def get_unique_sent(sents):
	sent_map = {}
	retrace_indices = []
	for sent in sents:
		sent_str = ' '.join(sent)
		if sent_str not in sent_map:
			sent_map[sent_str] = (len(sent_map), sent)
		idx = sent_map[sent_str][0]
		retrace_indices.append(idx)

	# sort by idx
	unique_sent = [(sent, idx) for key, (idx, sent) in sent_map.items()]
	unique_sent = sorted(unique_sent, key=lambda x: x[1])
	unique_sent = [p[0] for p in unique_sent]

	# sanity check
	assert(len(retrace_indices) == len(sents))
	return unique_sent, retrace_indices


def load_unique_sent(context):
	sent = []
	with open(path, 'r+') as f:
		for l in f:
			sent.append(l.strip())
	return {sent:idx for idx, sent in enumerate(sent)}


def load_infersent_model(model_path, w2v_path):
	params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
		'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
	infersent = InferSent(params_model)
	infersent.load_state_dict(torch.load(model_path))
	infersent.set_w2v_path(w2v_path)
	return infersent


def process(model, context, query, output):
	context_sents = [sent for par in context for sent in par]
	all_sents = context_sents + query
	unique_sents, _ = get_unique_sent(all_sents)	# returned sents are already one string per sent (no longer tokens)

	print('{0} unique sentences to process'.format(len(unique_sents)))

	unique_sents = [' '.join(sent) for sent in unique_sents]
	sent_map = {sent: i for i, sent in enumerate(unique_sents)}
	assert(len(sent_map) == len(unique_sents))

	# make str
	model.build_vocab(unique_sents, tokenize=False)
	unique_sents = model.encode(unique_sents, tokenize=False)
	#unique_sents = [np.zeros(4096) for _ in unique_sents]

	print_every = 500
	f = h5py.File(output, 'w')
	for ex_idx, (par, q) in enumerate(zip(context, query)):
		sent_idx = [sent_map[' '.join(sent)] for sent in par]
		emb = [torch.from_numpy(unique_sents[i]).unsqueeze(0) for i in sent_idx]
		emb = torch.cat(emb, 0)

		assert(emb.shape == (len(par), 4096))
		f['{0}.context'.format(ex_idx)] = emb.float().numpy()
		f['{0}.query'.format(ex_idx)] = unique_sents[sent_map[' '.join(q)]].astype(np.float32)

		if (ex_idx+1) % print_every == 0:
			print(ex_idx+1)
	f.close()



def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--src', help="Path to the tokenize context with sentence separator", default="../qa/data/squad-v1.1/train.contextsent.txt")
	parser.add_argument('--tgt', help="Path to the tokenize query", default="../qa/data/squad-v1.1/train.query.txt")
	parser.add_argument('--model', help="Path to the pretrained infersent model", default="encoder/infersent2.pkl")
	parser.add_argument('--w2v', help="Path to the w2v file", default="crawl-300d-2M.vec")
	parser.add_argument('--output', help="Prefix of output files", default="../qa/data/train")
	opt = parser.parse_args(arguments)
	
	model = load_infersent_model(opt.model, opt.w2v)
	context = load_sent(opt.src)
	query = load_token(opt.tgt)

	process(model, context, query, opt.output + '.infersent.hdf5')

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))