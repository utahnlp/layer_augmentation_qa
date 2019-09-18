import os
import sys
import argparse
import numpy as np
import h5py
import itertools
from collections import defaultdict
import json


class Indexer:
    def __init__(self, symbols = ["<blank>"]):
        self.PAD = symbols[0]
        self.num_oov = 1
        self.d = {self.PAD: 0}
        self.cnt = {self.PAD: 0}
        for i in range(self.num_oov): #hash oov words to one of 100 random embeddings
            oov_word = '<oov'+ str(i) + '>'
            self.d[oov_word] = len(self.d)
            self.cnt[oov_word] = 0
            
    def convert(self, w):        
        return self.d[w] if w in self.d else self.d['<oov' + str(np.random.randint(self.num_oov)) + '>']

    def convert_sequence(self, ls):
        return [self.convert(l) for l in ls]

    def write(self, outfile):
        print(len(self.d), len(self.cnt))
        assert(len(self.d) == len(self.cnt))
        with open(outfile, 'w+') as f:
            items = [(v, k) for k, v in self.d.items()]
            items.sort()
            for v, k in items:
                f.write('{0} {1} {2}\n'.format(k, v, self.cnt[k]))

    # register tokens only appear in wv
    #   NOTE, only do counting on training set
    def register_words(self, wv, seq, count):
        for w in seq:
            if w in wv and w not in self.d:
                self.d[w] = len(self.d)
                self.cnt[w] = 0
            if w in self.cnt:
                self.cnt[w] = self.cnt[w] + 1 if count else self.cnt[w]

    #   NOTE, only do counting on training set
    def register_all_words(self, seq, count):
        for w in seq:
            if w not in self.d:
                self.d[w] = len(self.d)
                self.cnt[w] = 0
            if w in self.cnt:
                self.cnt[w] = self.cnt[w] + 1 if count else self.cnt[w]


def pad(ls, length, symbol, pad_back = True):
    if len(ls) >= length:
        return ls[:length]
    if pad_back:
        return ls + [symbol] * (length -len(ls))
    else:
        return [symbol] * (length -len(ls)) + ls  


def get_glove_words(f):
    glove_words = set()
    word_vec_size = None
    for line in open(f, "r"):
        d = line.split()
        # get info from the first line
        if word_vec_size is None:
            word_vec_size = len(d) - 1
        # there might be multi-word token, so not just split()[0]
        word = d[:len(d)-word_vec_size]
        word = ' '.join(word)
        glove_words.add(word)
    return glove_words


# split a paragraphs with ||| sentence separator
#   into a list of sents, where each sent is a list of tokens
def split_par(par):
    sents = par.strip().split('|||')
    sents = [s for s in sents if s.strip() != '']
    sents = [s.strip().split(' ') for s in sents]
    return sents


def make_vocab(args, glove_vocab, all_word_indexer, word_indexer, srcfile, targetfile, seqlength, count):
    num_ex = 0
    for _, (src_orig, targ_orig) in enumerate(zip(open(srcfile,'r'), open(targetfile,'r'))):
        if args.lowercase == 1:
            src_orig = src_orig.lower()
            targ_orig = targ_orig.lower()

        targ = targ_orig.strip().split()
        #src = src_orig.strip().split()
        src = split_par(src_orig)
        src = [t for s in src for t in s]

        assert(len(targ) <= seqlength and len(src) <= seqlength)

        num_ex += 1
        all_word_indexer.register_all_words(targ, count)
        word_indexer.register_words(glove_vocab, targ, count)

        all_word_indexer.register_all_words(src, count)
        word_indexer.register_words(glove_vocab, src, count)
    return num_ex


def convert(args, all_word_indexer, word_indexer, srcfile, targetfile, spanfile, batchsize, seqlength, outfile, num_ex, min_sent_l=10000, max_sent_l=0, seed=0):
    np.random.seed(seed)

    max_sent_num = args.max_sent_num

    # record indices to all tokens
    all_targets = np.zeros((num_ex, seqlength), dtype=int)
    all_sources = np.zeros((num_ex, seqlength), dtype=int)

    # record indices to only those appear in word_indexer
    targets = np.zeros((num_ex, seqlength), dtype=int)
    sources = np.zeros((num_ex, seqlength), dtype=int)
    source_lengths = np.zeros((num_ex,), dtype=int) # the number of tokens in context
    target_lengths = np.zeros((num_ex,), dtype=int) # target sentence length (1 sentence)
    source_sent_lengths = np.zeros((num_ex, max_sent_num), dtype=int)   # the list of sentence lengths in context
    spans = np.zeros((num_ex, 2), dtype=int)
    batch_keys = np.array([None for _ in range(num_ex)])
    ex_idx = np.zeros(num_ex, dtype=int)

    dropped = 0
    sent_id = 0
    for _, (src_orig, targ_orig, span_orig) in enumerate(zip(open(srcfile,'r'), open(targetfile,'r'), open(spanfile,'r'))):
        if args.lowercase == 1:
            src_orig = src_orig.lower()
            targ_orig = targ_orig.lower()

        # remove sentence delimiter, if there is any
        #src_sents = src_orig.strip().split('|||')
        #src_sent_toks = [s.strip().split(' ') for s in src_sents]
        src_sent_toks = split_par(src_orig)
        src_sent_lengths = [len(s) for s in src_sent_toks]
        src_toks = [t for s in src_sent_toks for t in s]
        targ_toks = targ_orig.strip().split()
        span = span_orig.strip().split()
        assert(len(span) == 2)
        span = [int(span[0]), int(span[1])] # end idx is inclusive

        
        min_sent_l = min(len(targ_toks), len(src_toks), min_sent_l)
        max_sent_l = max(len(targ_toks), len(src_toks), max_sent_l)
        # DO NOT drop anything, causing inconsistent indices

        # pad to meet seqlength
        targ = pad(targ_toks, seqlength, word_indexer.PAD)
        targ = word_indexer.convert_sequence(targ)
        targ = np.array(targ, dtype=int)
        src = pad(src_toks, seqlength, word_indexer.PAD)
        src = word_indexer.convert_sequence(src)
        src = np.array(src, dtype=int)
        span = np.array(span, dtype=int)

        all_targ = pad(targ_toks, seqlength, all_word_indexer.PAD)
        all_targ = all_word_indexer.convert_sequence(all_targ)
        all_targ = np.array(all_targ, dtype=int)
        all_src = pad(src_toks, seqlength, all_word_indexer.PAD)
        all_src = all_word_indexer.convert_sequence(all_src)
        all_src = np.array(all_src, dtype=int)
        
        targets[sent_id] = np.array(targ,dtype=int)
        target_lengths[sent_id] = (targets[sent_id] != 0).sum()
        sources[sent_id] = np.array(src, dtype=int)
        source_lengths[sent_id] = (sources[sent_id] != 0).sum() 
        source_sent_lengths[sent_id, :len(src_sent_lengths)] = src_sent_lengths
        spans[sent_id] = np.array(span, dtype=int)
        all_targets[sent_id] = np.array(all_targ, dtype=int)
        all_sources[sent_id] = np.array(all_src, dtype=int)
        #batch_keys[sent_id] = (source_lengths[sent_id], target_lengths[sent_id])

        # use the list of sent lengths as batch key
        #   the consequences are most likely examples with the same context will be batched together
        #   and the question lengths may vary
        if args.batch_sent == 1:
            batch_keys[sent_id] = src_sent_lengths
        else:
            batch_keys[sent_id] = [sum(src_sent_lengths)]

        # sanity check
        assert((targets[sent_id] != 0).sum() == (all_targets[sent_id] != 0).sum())
        assert((sources[sent_id] != 0).sum() == (all_sources[sent_id] != 0).sum())
        assert(spans[sent_id][0] < source_lengths[sent_id] and spans[sent_id][1] < source_lengths[sent_id])

        sent_id += 1
        if sent_id % 10000 == 0:
            print("{}/{} sentences processed".format(sent_id, num_ex))

    assert(sent_id == num_ex)
    print("{}/{} sentences processed".format(sent_id, num_ex))
    # shuffle
    rand_idx = np.random.permutation(num_ex)
    targets = targets[rand_idx]
    sources = sources[rand_idx]
    spans = spans[rand_idx]
    source_lengths = source_lengths[rand_idx]
    target_lengths = target_lengths[rand_idx]
    source_sent_lengths = source_sent_lengths[rand_idx]
    batch_keys = batch_keys[rand_idx]
    ex_idx = rand_idx
    all_targets = all_targets[rand_idx]
    all_sources = all_sources[rand_idx]
    
    # break up batches based on source/target lengths
    sorted_keys = sorted([(i, p) for i, p in enumerate(batch_keys)], key=lambda x: x[1])
    sorted_idx = [i for i, _ in sorted_keys]
    # rearrange examples
    sources = sources[sorted_idx]
    targets = targets[sorted_idx]
    spans = spans[sorted_idx]
    target_l = target_lengths[sorted_idx]
    source_l = source_lengths[sorted_idx]
    source_sent_l = source_sent_lengths[sorted_idx]
    ex_idx = rand_idx[sorted_idx]
    all_targets = all_targets[sorted_idx]
    all_sources = all_sources[sorted_idx]

    cur_src_l = []
    batch_location = [] #idx where src sent length changes
    for j,i in enumerate(sorted_idx):
        if batch_keys[i] != cur_src_l:
        #if batch_keys[i][0] != cur_src_l or batch_keys[i][1] != cur_tgt_l:
            cur_src_l = batch_keys[i]
            batch_location.append(j)

    # get batch strides
    cur_idx = 0
    batch_idx = [0]
    batch_l = []
    #source_sent_l_new = []
    source_l_new = []
    for i in range(len(batch_location)-1):
        end_location = batch_location[i+1]
        while cur_idx < end_location:
            cur_idx = min(cur_idx + batchsize, end_location)
            batch_idx.append(cur_idx)

    # rearrange examples according to batch strides
    for i in range(len(batch_idx)):
        end = batch_idx[i+1] if i < len(batch_idx)-1 else len(sources)

        batch_l.append(end - batch_idx[i])
        source_l_new.append(source_l[batch_idx[i]])
        #source_sent_l_new.append(source_sent_l[batch_idx[i]])

        # sanity check
        for k in range(batch_idx[i], end):
            assert(source_l[k] == source_l_new[-1])
            assert(sources[k, source_l[k]:].sum() == 0)


    # Write output
    f = h5py.File(outfile, "w")        
    f["source"] = sources
    f["target"] = targets
    f["target_l"] = target_l    # (num_ex,)
    f["source_l"] = source_l_new    # (batch_l,)
    f['source_sent_l'] = source_sent_l
    f["span"] = spans
    f["batch_l"] = batch_l
    f["batch_idx"] = batch_idx
    f["source_size"] = np.array([len(word_indexer.d)])
    f["target_size"] = np.array([len(word_indexer.d)])
    f['ex_idx'] = ex_idx
    f['all_source'] = all_sources
    f['all_target'] = all_targets
    print("Saved {} sentences (dropped {} due to length/unk filter)".format(
        len(f["source"]), dropped))
    print('Number of batches: {0}'.format(len(batch_idx)))
    f.close()                
    return min_sent_l, max_sent_l

def process(args):
    all_word_indexer = Indexer()    # all tokens will be recorded
    word_indexer = Indexer()        # only glove tokens will be recorded
    glove_vocab = get_glove_words(args.glove)

    print("First pass through data to get vocab...")
    num_ex_train = make_vocab(args, glove_vocab, all_word_indexer, word_indexer, args.srcfile, args.targfile, args.seqlength,
        count=True)
    print("Number of sentences in training: {0}, number of tokens: {1}/{2}".format(num_ex_train, len(word_indexer.d), len(all_word_indexer.d)))
    num_ex_valid = make_vocab(args, glove_vocab, all_word_indexer, word_indexer, args.srcvalfile, args.targvalfile, args.seqlength,
        count=False)
    print("Number of sentences in valid: {0}, number of tokens: {1}/{2}".format(num_ex_valid, len(word_indexer.d), len(all_word_indexer.d)))   

    print('Number of all tokens found: {0}'.format(len(all_word_indexer.d)))
    all_word_indexer.write(args.outputfile + '.allword.dict')
    
    print('Number of tokens collected: {0}'.format(len(word_indexer.d)))
    word_indexer.write(args.outputfile + ".word.dict")

    min_sent_l = 1000000
    max_sent_l = 0
    min_sent_l, max_sent_l = convert(args, all_word_indexer, word_indexer, args.srcvalfile, args.targvalfile, args.spanvalfile, args.batchsize, args.seqlength, args.outputfile + "-val.hdf5", num_ex_valid,
                         min_sent_l, max_sent_l, args.seed)
    min_sent_l, max_sent_l = convert(args, all_word_indexer, word_indexer, args.srcfile, args.targfile, args.spanfile, args.batchsize, args.seqlength, args.outputfile + "-train.hdf5", num_ex_train, min_sent_l, max_sent_l, args.seed)
    print("Min sent length (before dropping): {}".format(min_sent_l))
    print("Max sent length (before dropping): {}".format(max_sent_l))    
    
def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', help="Path to the data dir",
                        default = "data/squad-v1.1/")
    parser.add_argument('--srcfile', help="Path to sent1 training data.",
                        default = "train.context.txt")
    parser.add_argument('--targfile', help="Path to sent2 training data.",
                        default = "train.query.txt")
    parser.add_argument('--spanfile', help="Path to span data.",
                        default = "train.span.txt")    
    parser.add_argument('--srcvalfile', help="Path to sent1 validation data.",
                        default = "dev.context.txt")
    parser.add_argument('--targvalfile', help="Path to sent2 validation data.",
                        default = "dev.query.txt")
    parser.add_argument('--spanvalfile', help="Path to span validation data.",
                        default = "dev.span.txt")
    parser.add_argument('--batch_sent', help="Whether to batchup according to context sentence lengths",
                        type=int, default = 0)
    
    parser.add_argument('--batchsize', help="Size of each minibatch.", type=int, default=15)
    parser.add_argument('--seqlength', help="Maximum sequence length.", type=int, default=900)
    parser.add_argument('--max_sent_num', help="Maximum sentence number.", type=int, default=50)
    parser.add_argument('--outputfile', help="Prefix of the output file names.", type=str, default = "squad")
    parser.add_argument('--lowercase', help="Whether to use lowercase for vocabulary.", type=int, default = 1)
    parser.add_argument('--seed', help="seed of shuffling sentences.", type = int, default = 1)
    parser.add_argument('--glove', type = str, default = '')    
    args = parser.parse_args(arguments)

    #
    args.srcfile = args.dir + args.srcfile
    args.targfile = args.dir + args.targfile
    args.spanfile = args.dir + args.spanfile
    args.srcvalfile = args.dir + args.srcvalfile
    args.targvalfile = args.dir + args.targvalfile
    args.spanvalfile = args.dir + args.spanvalfile
    args.outputfile = args.dir + args.outputfile

    process(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
