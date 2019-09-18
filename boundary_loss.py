import sys

import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *


# Boundary Loss
class BoundaryLoss(torch.nn.Module):
	def __init__(self, opt, shared):
		super(BoundaryLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		# do not creat loss node globally
		self.idx1_correct = 0
		self.idx2_correct = 0
		self.total_em_bow = 0.0
		self.total_f1_bow = 0.0
		self.total_em1_sent = 0.0
		self.total_em2_sent = 0.0
		self.num_ex = 0
		self.verbose = False
		self.start_off_cnt = {}
		self.end_off_cnt = {}


	def forward(self, pred, gold):
		log_p1, log_p2 = pred
		# loss
		crit = torch.nn.NLLLoss(reduction='sum')	# for pytorch < 0.4.1, use size_average=False
		if self.opt.gpuid != -1:
			crit = crit.cuda()
		loss1 = crit(log_p1, gold[:,0])	# loss on start idx
		loss2 = crit(log_p2, gold[:,1])	# loss on end idx
		loss = (loss1 + loss2)

		# stats
		bounded = pick_best_span(log_p1.cpu().data, log_p2.cpu().data, self.opt.span_l)
		idx1, idx2 = bounded[:,0], bounded[:,1]
		self.idx1_correct += count_correct_idx(idx1, gold[:,0].cpu().data)
		self.idx2_correct += count_correct_idx(idx2, gold[:,1].cpu().data)
		self.num_ex += self.shared.batch_l

		# f1
		pred_ans = get_answer_txt(bounded[:,0], bounded[:,1], self.shared.res_map['token_span'], self.shared.res_map['raw_context'])
		gold_ans = self.shared.res_map['raw_answer']
		em_bow = get_em_bow(pred_ans, gold_ans)
		f1_bow = get_f1_bow(pred_ans, gold_ans)
		self.total_em_bow += sum(em_bow)
		self.total_f1_bow += sum(f1_bow)

		# sent em
		em1_sent = get_em_sent(idx1, gold[:, 0], self.shared.context_sent_l)
		em2_sent = get_em_sent(idx2, gold[:, 1], self.shared.context_sent_l)
		self.total_em1_sent += sum(em1_sent)
		self.total_em2_sent += sum(em2_sent)
		pred_sents = get_sent(idx1, self.shared.context_sent_l, self.shared.res_map['token_span'], self.shared.res_map['raw_context'])
		gold_sents = get_sent(gold[:, 0], self.shared.context_sent_l, self.shared.res_map['token_span'], self.shared.res_map['raw_context'])

		# verbose
		if self.verbose:
			raw_query = self.shared.res_map['raw_query']
			k = 0
			print('*************************** pred gold')
			for q, p, g, em, f1, em1_s, em2_s in zip(raw_query, pred_ans, gold_ans, em_bow, f1_bow, em1_sent, em2_sent):
				print(u'{0} {1} {2:.4f} {3:.4f} {4:.4f} {5:.4f}'.format(p, g, em, f1, em1_s, em2_s).encode('utf-8'))
				print(q.encode('utf-8'))
				print(pred_sents[k].encode('utf-8'))
				print(gold_sents[k].encode('utf-8'))
				k += 1

			# count the offset
			for i in range(self.shared.batch_l):
				gold = gold.cpu()
				pred_start, pred_end = bounded[i][0], bounded[i][1]
				gold_start, gold_end = gold[i][0], gold[i][1]
				start_off = int(pred_start - gold_start)
				end_off = int(pred_end - gold_end)

				if start_off not in self.start_off_cnt:
					self.start_off_cnt[start_off] = 0
				self.start_off_cnt[start_off] = self.start_off_cnt[start_off] + 1

				if end_off not in self.end_off_cnt:
					self.end_off_cnt[end_off] = 0
				self.end_off_cnt[end_off] = self.end_off_cnt[end_off] + 1

		return loss


	# return a string of stats
	def print_cur_stats(self):
		stats = 'Span {0:.3f}/{1:.3f} '.format(
			float(self.idx1_correct) / self.num_ex, float(self.idx2_correct) / self.num_ex)
		stats += 'EM {0:.3f} F1 {1:.3f} Sent{2:.3f}/{3:.3f} '.format(
			self.total_em_bow / self.num_ex,
			self.total_f1_bow / self.num_ex,
			self.total_em1_sent / self.num_ex,
			self.total_em2_sent / self.num_ex)
		return stats


	# get training metric (scalar metric, extra metric)
	#	the scalar metric will be used to pick the best model
	#	the extra metric a list of scalars for extra info
	def get_epoch_metric(self):
		acc1 = float(self.idx1_correct) / self.num_ex
		acc2 = float(self.idx2_correct) / self.num_ex
		em_bow = float(self.total_em_bow) / self.num_ex
		f1_bow = self.total_f1_bow / self.num_ex
		em1_sent = self.total_em1_sent / self.num_ex
		em2_sent = self.total_em2_sent / self.num_ex

		if self.verbose:
			out_path = self.opt.print + '.start_off_cnt.txt'
			print('printing to ' + out_path)
			with open(out_path, 'w') as f:
				print(self.start_off_cnt)
				f.write('\n'.join(['{0}:{1}'.format(off, cnt) for off, cnt in sorted([(off, cnt) for off, cnt in self.start_off_cnt.items()])]))

			out_path = self.opt.print + '.end_off_cnt.txt'
			print('printing to ' + out_path)
			with open(out_path, 'w') as f:
				print(self.end_off_cnt)
				f.write('\n'.join(['{0}:{1}'.format(off, cnt) for off, cnt in sorted([(off, cnt) for off, cnt in self.end_off_cnt.items()])]))

		return f1_bow, ((acc1 + acc2) / 2.0, em_bow, f1_bow, em1_sent, em2_sent)


	def begin_pass(self):
		# clear stats
		self.idx1_correct = 0
		self.idx2_correct = 0
		self.num_ex = 0
		self.total_em_bow = 0.0
		self.total_f1_bow = 0.0
		self.total_em1_sent = 0.0
		self.total_em2_sent = 0.0

	def end_pass(self):
		pass

