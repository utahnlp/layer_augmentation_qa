import sys
import argparse
import h5py
import torch
import numpy as np


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
def get_unique_context(context):
	context_map = {}
	retrace_indices = []
	for par in context:
		par_str = ' ' .join([' '.join(s) for s in par])
		if par_str not in context_map:
			context_map[par_str] = (len(context_map), par)
		idx = context_map[par_str][0]
		retrace_indices.append(idx)

	# sort by idx
	unique_context = [(par, idx) for key, (idx, par) in context_map.items()]
	unique_context = sorted(unique_context, key=lambda x: x[1])
	unique_context = [p[0] for p in unique_context]

	# sanity check
	assert(len(retrace_indices) == len(context))
	return unique_context, retrace_indices


def get_unique_sent(sents):
	sent_map = {}
	retrace_indices = []
	for sent in sents:
		sent_str = ' ' .join(sent)
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


def split_par(par):
	sents = par.strip().split('|||')
	sents = [s for s in sents if s.strip() != '']
	sents = [s.strip().split(' ') for s in sents]
	return sents


def process(opt, context, query, output):
	assert(len(context) == len(query))

	context, _ = get_unique_context(context)
	context_sents = [sent for par in context for sent in par]
	context_sents, _ = get_unique_sent(context_sents)
	print('{0} unique context sent found'.format(len(context_sents)))

	query, _ = get_unique_sent(query)
	print('{0} unique query found'.format(len(query)))

	cnt = 0
	with open(output, 'w+') as f:
		for sent in context_sents:
			f.write(' '.join(sent) + '\n')
			cnt += 1

		for sent in query:
			f.write(' '.join(sent) + '\n')
			cnt += 1

	print('{0} sentences found'.format(cnt))


def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--src', help="Path to the tokenized context file with sentence separator", default="data/dev.contextsent.txt")
	parser.add_argument('--tgt', help="Path to the tokenized query file", default="data/dev.query.txt")
	parser.add_argument('--output', help="Prefix of output files", default="data/dev")
	opt = parser.parse_args(arguments)
	
	context = load_sent(opt.src)
	query = load_token(opt.tgt)
	process(opt, context, query, opt.output+'.uniquesent.txt')

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

