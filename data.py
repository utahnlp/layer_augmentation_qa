import io
import h5py
import torch
from torch import nn
from torch import cuda
import numpy as np
import ujson
from util import *

class Data():
	def __init__(self, opt, data_file, res_files=None):
		self.opt = opt
		self.data_name = data_file

		print('loading data from {0}'.format(data_file))
		f = h5py.File(data_file, 'r')
		self.source = f['source'][:]	# indices to glove tokens
		self.target = f['target'][:]
		self.all_source = f['all_source'][:]	# indices to all tokens
		self.all_target = f['all_target'][:]
		self.source_l = f['source_l'][:].astype(np.int32)	# (batch_l,)
		self.target_l = f['target_l'][:].astype(np.int32)	# (num_ex,)
		self.source_sent_l = f['source_sent_l'][:].astype(np.int32)	# (batch_l,)
		self.span = f['span'][:]
		self.batch_l = f['batch_l'][:].astype(np.int32)
		self.batch_idx = f['batch_idx'][:].astype(np.int32)
		self.source_size = f['source_size'][:].astype(np.int32)
		self.target_size = f['target_size'][:].astype(np.int32)
		self.ex_idx = f['ex_idx'][:].astype(np.int32)
		self.length = self.batch_l.shape[0]

		self.all_source = torch.from_numpy(self.all_source)
		self.all_target = torch.from_numpy(self.all_target)
		self.source = torch.from_numpy(self.source)
		self.target = torch.from_numpy(self.target)
		self.span = torch.from_numpy(self.span)

		# postpone the transfer to gpu to the batch running stage
		#if self.opt.gpuid != -1:
		#	self.source = self.source.cuda()
		#	self.target = self.target.cuda()
		#	self.span = self.span.cuda()

		# load char_idx file
		print('loading char idx from {0}'.format(opt.char_idx))
		f = h5py.File(opt.char_idx, 'r')
		self.char_idx = f['char_idx'][:]
		self.char_idx = torch.from_numpy(self.char_idx)
		assert(self.char_idx.shape[1] == opt.token_l)
		assert(self.char_idx.max()+1 == opt.num_char)
		print('{0} chars found'.format(self.char_idx.max()+1))

		self.batches = []
		for i in range(self.length):
			start = self.batch_idx[i]
			end = start + self.batch_l[i]

			# get example token indices
			target_l_i = self.target_l[start:end]	# (batch_l,)
			max_target_l = target_l_i.max()
			all_source_i = self.all_source[start:end, 0:self.source_l[i]]
			all_target_i = self.all_target[start:end, 0:max_target_l]
			source_i = self.source[start:end, 0:self.source_l[i]]
			target_i = self.target[start:end, 0:max_target_l]
			span_i = self.span[start:end]

			source_sent_l_i = []
			for ls in self.source_sent_l[start:end]:
				num_sent = (ls != 0).sum()
				source_sent_l_i.append([int(l) for l in ls[:num_sent]])

			# sanity check
			assert(self.source[start:end, self.source_l[i]:].sum() == 0)
			for k in range(start, end):
				assert(self.target[k, self.target_l[k]:].sum() == 0)

			# src, tgt, all_src, all_tgt, batch_l, src_l, tgt_l, span, raw info
			self.batches.append((source_i, target_i, all_source_i, all_target_i, 
				int(self.batch_l[i]), int(self.source_l[i]), source_sent_l_i, target_l_i, span_i))

		# count examples
		self.num_ex = 0
		for i in range(self.length):
			self.num_ex += self.batch_l[i]


		# load resource files
		self.res_names = []
		if res_files is not None:
			for f in res_files:
				if f.endswith('txt'):
					res_names = self.__load_txt(f)

				elif f.endswith('elmo.hdf5'):
					res_names = self.__load_elmo(f)

				elif f.endswith('json'):
					res_names = self.__load_json_res(f)

				else:
					assert(False)
				self.res_names.extend(res_names)


	def subsample(self, ratio, random):
		target_num_ex = int(float(self.num_ex) * ratio)
		sub_idx = [int(idx) for idx in torch.randperm(self.size())] if random else [i for i in range(self.size())]
		cur_num_ex = 0
		i = 0
		while cur_num_ex < target_num_ex:
			cur_num_ex += self.batch_l[sub_idx[i]]
			i += 1
		return sub_idx[:i], cur_num_ex


	def split(self, sub_idx, ratio):
		num_ex = sum([self.batch_l[i] for i in sub_idx])
		target_num_ex = int(float(num_ex) * ratio)

		cur_num_ex = 0
		cur_pos = 0
		for i in range(len(sub_idx)):
			cur_pos = i
			cur_num_ex += self.batch_l[sub_idx[i]]
			if cur_num_ex >= target_num_ex:
				break

		return sub_idx[:cur_pos+1], sub_idx[cur_pos+1:], cur_num_ex, num_ex - cur_num_ex


	def __load_txt(self, path):
		lines = []
		print('loading resource from {0}'.format(path))
		# read file in unicode mode!!!
		with io.open(path, 'r+', encoding="utf-8") as f:
			for l in f:
				lines.append(l.rstrip())
		# the second last extension is the res name
		res_name = path.split('.')[-2]
		res_data = lines[:]

		# some customized parsing
		parsed = []
		if res_name == 'token_span':
			for l in res_data:
				parsed.append([(int(p.split(':')[0]), int(p.split(':')[1])) for p in l.split()])
		elif res_name == 'raw_answer':
			for l in res_data:
				parsed.append(l.rstrip().split('|||'))	# a list of strings that are all ground truth answers
		elif res_name == 'context' or res_name == 'query':
			for l in res_data:
				parsed.append(l.rstrip().split(' '))
		else:
			parsed = res_data

		setattr(self, res_name, parsed)
		return [res_name]


	def __load_elmo(self, path):
		print('loading resources from {0}'.format(path))
		f = h5py.File(path, 'r')
		self.elmo_file = f

		# the attributes will not be assigned to self, instead, they are customized in __get_res
		return ['elmo_context', 'elmo_query']


	def __load_json_res(self, path):
		print('loading resource from {0}'.format(path))
		
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		# get key name of the file
		assert(len(j_obj) == 2)
		res_type = next(iter(j_obj))

		res_name = None
		if j_obj[res_type] == 'map':
			res_name = self.__load_json_map(path)
		elif j_obj[res_type] == 'list':
			res_name = self.__load_json_list(path)
		else:
			assert(False)

		return [res_name]

	
	def __load_json_map(self, path):
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		assert(len(j_obj) == 2)

		res_name = None
		for k, v in j_obj.items():
			if k != 'type':
				res_name = k

		# optimize indices
		res = {}
		for k, v in j_obj[res_name].items():
			lut = {}
			for i, j in v.items():
				if i == res_name:
					lut[res_name] = [int(l) for l in j]
				else:
					lut[int(i)] = ([l for l in j[0]], [l for l in j[1]])

			res[int(k)] = lut
		
		setattr(self, res_name, res)
		return res_name


	def __load_json_list(self, path):
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		assert(len(j_obj) == 2)
		
		res_name = None
		for k, v in j_obj.items():
			if k != 'type':
				res_name = k

		# optimize indices
		res = {}
		for k, v in j_obj[res_name].items():
			p = v['p']
			h = v['h']

			# for token indices, shift by 1 to incorporate the nul-token at the beginning
			res[int(k)] = ([l for l in p], [l for l in h])
		
		setattr(self, res_name, res)
		return res_name


	def size(self):
		return self.length


	def __get_res_elmo(self, res_name, idx):
		# if it's already in cache
		if idx in self.elmo_cache:
			return self.elmo_cache[idx][res_name]

		if res_name == 'elmo_context':
			embs = torch.from_numpy(self.elmo_file['{0}.context_batch'.format(idx)][:])
			if self.opt.gpuid != -1:
				embs = embs.cuda()
			return embs
		elif res_name == 'elmo_query':
			embs = torch.from_numpy(self.elmo_file['{0}.query_batch'.format(idx)][:])
			if self.opt.gpuid != -1:
				embs = embs.cuda()
			return embs
		else:
			raise Exception('unrecognized res {0}'.format(res_name))


	def __getitem__(self, idx):
		(source, target, all_source, all_target, 
			batch_l, source_l, source_sent_l, target_l, span) = self.batches[idx]
		token_l = self.opt.token_l

		# get char indices
		# 	the back forth data transfer should be eliminated
		max_target_l = target_l.max()
		char_source = self.char_idx[all_source.contiguous().view(-1)].view(batch_l, source_l, token_l)
		char_target = self.char_idx[all_target.contiguous().view(-1)].view(batch_l, max_target_l, token_l)

		# transfer to gpu if needed
		if self.opt.gpuid != -1:
			char_source = char_source.cuda()
			char_target = char_target.cuda()
			source = source.cuda()
			target = target.cuda()
			span = span.cuda()

		# get batch ex indices
		batch_ex_idx = [self.ex_idx[i] for i in range(self.batch_idx[idx], self.batch_idx[idx] + self.batch_l[idx])]

		res_map = self.__get_res(idx)

		return (self.data_name, source, target, char_source, char_target, 
			batch_ex_idx, batch_l, source_l, source_sent_l, target_l, span, res_map)


	def __get_res(self, idx):
		# if there is no resource presents, return None
		if len(self.res_names) == 0:
			return None

		batch_ex_idx = [self.ex_idx[i] for i in range(self.batch_idx[idx], self.batch_idx[idx] + self.batch_l[idx])]

		all_res = {}
		for res_n in self.res_names:
			# some customization for elmo is needed here for lazy loading
			if 'elmo' in res_n:
				batch_res = self.__get_res_elmo(res_n, idx)
				all_res[res_n] = batch_res
			else:
				res = getattr(self, res_n)

				batch_res = [res[ex_id] for ex_id in batch_ex_idx]
				all_res[res_n] = batch_res

		return all_res


	# NOTE, only call this function during training or eval
	#	this function will trigger preloading a number of batches every time cur_idx is not in the cache
	#def preload(self, batch_order, cur_idx):
	#	if self.opt.dynamic_elmo == 1:
	#		return
	#	if cur_idx not in self.elmo_cache:
	#		# first clean up the cache
	#		self.elmo_cache = {}
	#		k = cur_idx
	#		while k < self.opt.elmo_cache_size and k <= len(cur_idx):
	#			context = self.__get_res_elmo('elmo_context', batch_order[k])
	#			query = self.__get_res_elmo('elmo_query', batch_order[k])
	#			self.elmo_cache[batch_order[k]] = {'elmo_context': context, 'elmo_query':query}
	#			k += 1


	# something at the beginning of each pass of training/eval
	#	e.g. setup preloading
	def begin_pass(self):
		self.elmo_cache = {}


	def end_pass(self):
		self.elmo_cache = {}


if __name__ == '__main__':
	sample_data = './data/squad-val.hdf5'
	from holder import *
	opt = Holder()
	opt.gpuid = -1

	d = Data(opt, sample_data, res_files=None)
	name, src, tgt, batch_ex_idx, batch_l, src_l, tgt_l, span, res = d[113]
	print('data size: {0}'.format(d.size()))
	print('name: {0}'.format(name))
	print('source: {0}'.format(src))
	print('target: {0}'.format(tgt))
	print('batch_ex_idx: {0}'.format(batch_ex_idx))
	print('batch_l: {0}'.format(batch_l))
	print('src_l: {0}'.format(src_l))
	print('tgt_l: {0}'.format(tgt_l))
	print('span: {0}'.format(span))

	print(d.source_size)
	print(d.target_size)