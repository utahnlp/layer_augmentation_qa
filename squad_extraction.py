import ujson
import sys
import argparse
import re
import spacy

spacy_nlp = spacy.load('en_core_web_sm')

# extra split referred to allenai docqa
extra_split_chars = ("-", "£", "€", "¥", "¢", "₹", "\u2212", "\u2014", "\u2013", "/", "~", "(", ")", "+", "^", "=", "\[", "\]", "'",
	'"', "'", "\ud01C", "\u2019", "\u201D", "\u2018", "\u00B0")
extra_split_tokens = ("``", "(?<=[^_])_(?=[^_])",  # dashes w/o a preceeding or following dash, so __wow___ -> ___ wow ___
	"''", "[" + "".join(extra_split_chars) + "]")
extra_split_chars_re = re.compile("(" + "|".join(extra_split_tokens) + ")")

def extra_split(tokens):
	return [x for t in tokens for x in extra_split_chars_re.split(t) if x != ""]

def rephrase_quote(tokens):
	return [t.replace("''", '"').replace("``", '"') for t in tokens]

# split token along with pos tag (basically duplicate)
def extra_split_with_pos(tokens, pos):
	rs_tokens = []
	rs_pos = []
	for t, p in zip(tokens, pos):
		split = [x for x in extra_split_chars_re.split(t) if x != ""]
		rs_tokens.extend(split)
		rs_pos.extend([p] * len(split))
	assert(len(rs_tokens) == len(rs_pos))
	return rs_tokens, rs_pos


# tokenizer that
#	splits into sentences (optnal)
#	tokenize
def tokenize_spacy(text, split_sent, tag_type):
	tokenized = spacy_nlp(text)
	if split_sent:
		tokenized_sents = []
		pos_sents = []
		for sent in tokenized.sents:
			tokenized_sents.append([tok.text for tok in sent if not tok.is_space])
			if tag_type == 'universal':
				pos_sents.append([tok.pos_ for tok in sent if not tok.is_space])
			elif tag_type == 'ptb':
				pos_sents.append([tok.tag_ for tok in sent if not tok.is_space])
			else:
				assert(False)
		return tokenized_sents, pos_sents

	toks = [tok.text for tok in tokenized if not tok.is_space]
	pos = None
	if tag_type == 'universal':
		pos = [tok.pos_ for tok in tokenized if not tok.is_space]
	elif tag_type == 'ptb':
		pos = [tok.tag_ for tok in tokenized if not tok.is_space]
	else:
		assert(False)

	return toks, pos


def get_gold(answer_spans):
	cnt = {}
	for span in answer_spans:
		if span in cnt:
			cnt[span] = cnt[span] + 1
		else:
			cnt[span] = 1
	sorted_keys = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
	maj_span = sorted_keys[0][0]
	return (maj_span, answer_spans.index(maj_span))



def write_to(ls, out_file):
	print('writing to {0}'.format(out_file))
	with open(out_file, 'w+') as f:
		for l in ls:
			f.write((l + '\n'))


def remap_char_idx(context, context_toks):
	context_tok_seq = ' '.join(context_toks)
	m = [-1 for _ in range(len(context))]
	i = 0
	j = 0
	while (i < len(context) and j < len(context_tok_seq)):
		# skip white spaces
		while context[i].strip() == '':
			i += 1
		while context_tok_seq[j].strip() == '':
			j += 1

		if context[i] == context_tok_seq[j]:
			m[i] = j
			i += 1
			j += 1
		elif context[i] == "'" and context[i+1] == "'" and context_tok_seq[j] == '"':
			m[i] = j
			i += 2
			j += 1
		#elif context[i] == '"' and context_tok_seq[j] == '\'':
		#	m[i] = j
		#	i += 1
		#	if context_tok_seq[j+1] == '\'':
		#		j += 2
		else:
			print(context.encode('utf8'))
			print(context_tok_seq.encode('utf8'))
			print(context[:i+1].encode('utf8'))
			print(context_tok_seq[:j+1].encode('utf8'))
			assert(False)

	return m

def remap_token_span(m, context, context_toks):
	tok_str = ' '.join(context_toks)
	remap = [-1 for _ in range(len(tok_str))]
	assert(len(m) == len(context))
	for i in range(len(m)):
		if m[i] != -1:
			remap[m[i]] = i

	assert(remap[0] != -1 and remap[-1] != -1)

	token_spans = []
	in_span = True
	start = 0
	for i in range(len(remap)):
		if (remap[i] == -1 and in_span):
			token_spans.append((remap[start], remap[i-1]))
			in_span = False
		elif remap[i] != -1 and not in_span:
			start = i
			in_span=True
	token_spans.append((remap[start], remap[-1]))

	# sanity check
	if len(token_spans) != len(context_toks):
		print(token_spans)
		print(context_toks)
		print(len(token_spans), len(context_toks))
		for i in range(len(context_toks)):
			if token_spans[i][1] - token_spans[i][0] + 1 != len(context_toks[i]):
				print(context_toks[i])
				print(token_spans[i])
				print(i)
	assert(len(token_spans) == len(context_toks))
	# make sure there is no -1 on boundary
	assert(sum([-1 in m[s[0]:s[1]+1] for s in token_spans]) == 0)

	return token_spans


def map_answer_idx(context, context_toks, m, char_idx1, char_idx2):
	context_tok_seq = ' '.join(context_toks)
	#m = remap_char_idx(context, context_toks)
	new_char_idx1 = m[char_idx1]
	new_char_idx2 = m[char_idx2]

	# count number of spaces
	tok_idx1 = context_tok_seq[new_char_idx1::-1].count(' ')
	tok_idx2 = context_tok_seq[new_char_idx2::-1].count(' ')

	# sanity check
	assert(tok_idx1 < len(context_toks))
	assert(tok_idx2 < len(context_toks))

	# NOTE, ending index is inclusive
	return (tok_idx1, tok_idx2)


def check_span_overlap(span1, span2):
	return (span1[0] <= span2[0] and span2[0] <= span1[1]) or (span1[0] <= span2[1] and span2[1] <= span1[1]) or \
		(span2[0] <= span1[0] and span1[0] <= span2[1]) or (span2[0] <= span1[1] and span1[1] <= span2[1])


def filter_by(keys, tokens, pos, token_span, ans_char_span, ans_tok_span):
	rs_tokens = []
	rs_pos = []
	rs_token_span = []
	rs_ans_span = [ans_tok_span[0], ans_tok_span[1]]
	for i, (tok, tag, tok_span) in enumerate(zip(tokens, pos, token_span)):
	
		#if tag in keys and sum([check_span_overlap(tok_span, a) for a in ans_char_span]) == 0:
		if tag in keys:
			if i < ans_tok_span[0]:
				rs_ans_span[0] -= 1
			if i <= ans_tok_span[1]:
				rs_ans_span[1] -= 1

			rs_ans_span[0] = 0 if rs_ans_span[0] == -1 else rs_ans_span[0]
			rs_ans_span[1] = 0 if rs_ans_span[1] == -1 else rs_ans_span[1]

			# few cases where the start idx becomes greater than end idx
			if rs_ans_span[0] > rs_ans_span[1]:
				rs_ans_span[0] = rs_ans_span[1]	# in this case, the answer is probably not good

		else:
			rs_tokens.append(tok)
			rs_pos.append(tag)
			rs_token_span.append((tok_span))

	assert(rs_ans_span[0] < len(rs_tokens))
	assert(rs_ans_span[1] < len(rs_tokens))

	return rs_tokens, rs_pos, rs_token_span, rs_ans_span


def extract(opt, json_file):
	all_raw_context = []
	all_context = []
	all_context_sents = []
	all_context_pos = []
	all_query = []
	all_query_pos = []
	all_span = []
	all_token_spans = []	# only for context
	all_raw_ans = []
	context_max_sent_num = 0
	max_sent_l = 0

	with open(json_file, 'r') as f:
		f_str = f.read()
	j_obj = ujson.loads(f_str)

	data = j_obj['data']

	for article in data:
		title = article['title']
		pars = article['paragraphs']
		for p in pars:
			context = p['context']
			qas = p['qas']
			# tokenize
			context = context.replace('\n', ' ')	# there are few cases have multiple paras, take them as single para
			context_sent_toks, context_sent_pos = tokenize_spacy(context, split_sent=True, tag_type=opt.tag_type)

			packed = [extra_split_with_pos(sent, pos_sent) for sent, pos_sent in zip(context_sent_toks, context_sent_pos)]

			context_sent_toks = [p[0] for p in packed]
			context_sent_toks = [rephrase_quote(s) for s in context_sent_toks]
			context_toks = [t for s in context_sent_toks for t in s]

			context_sent_pos = [p[1] for p in packed]
			context_pos = [pos for s in context_sent_pos for pos in s]

			assert(len(context_toks) == len(context_pos))

			# get the token spans (token to original char span)
			#	span end idx is inclusive
			char_remap = remap_char_idx(context, context_toks)
			token_spans = remap_token_span(char_remap, context, context_toks)

			max_sent_l = max(max_sent_l, len(context_toks))

			for qa in qas:
				query = qa['question']
				ans = qa['answers']
				# tokenize
				query_toks, query_pos = tokenize_spacy(query, split_sent=False, tag_type=opt.tag_type)
				query_toks, query_pos = extra_split_with_pos(query_toks, query_pos)
				query_toks = rephrase_quote(query_toks)

				assert(len(query_toks) == len(query_pos))

				max_sent_l = max(max_sent_l, len(query_toks))

				answer_orig_spans = []
				for a in ans:
					a_txt = a['text']
					idx1 = a['answer_start']
					idx2 = idx1 + len(a_txt) - 1	# end idx is inclusive

					answer_orig_spans.append((idx1, idx2))

				orig_maj_span = get_gold(answer_orig_spans)[0]
				# map orig char idx to tokenized word idx
				tok_idx1, tok_idx2 = map_answer_idx(context, context_toks, char_remap, orig_maj_span[0], orig_maj_span[1])

				# 
				orig_answer = context[orig_maj_span[0]:orig_maj_span[1]+1]
				all_orig_answers = [context[orig_span[0]:orig_span[1]+1] for orig_span in answer_orig_spans]
				matched_answer = context_toks[tok_idx1:tok_idx2+1]
				recovered_answer = context[token_spans[tok_idx1][0]:token_spans[tok_idx2][1]+1]
				print(orig_maj_span, (tok_idx1, tok_idx2), orig_answer, matched_answer, recovered_answer)

				# sanity check
				#	make sure recovered token is a superset of ground truth
				#	(some gold answers are partial token)
				if orig_answer not in recovered_answer:
					print(context)
					print(orig_answer)
					print(token_spans)
					print(tok_idx1, tok_idx2)
					assert(False)

				# concat sent tokens with sentence delimiter
				context_toks_separated = []
				for s in context_sent_toks:
					context_toks_separated.extend(s + ['|||'])

				# TODO, add option to filter pos in query as well
				if opt.filter != '':
					filters = opt.filter.split(',')
					filtered_context_toks, filtered_context_pos, filtered_span, filtered_ans_tok_idx = filter_by(
						filters, context_toks, context_pos, token_spans, [orig_maj_span], (tok_idx1, tok_idx2))

					filtered_ans = filtered_context_toks[filtered_ans_tok_idx[0]:filtered_ans_tok_idx[1]+1]
					if matched_answer != filtered_ans:
						print('chopped answer: {0}, {1}'.format(matched_answer, filtered_ans))
						

					# TODO, context_toks_separated is not filtered unfortunately...
					all_raw_context.append(context.rstrip())
					all_context.append(' '.join(filtered_context_toks))
					all_context_sents.append(' '.join(context_toks_separated))
					all_context_pos.append(' '.join(filtered_context_pos))
					all_query.append(' '.join(query_toks))
					all_query_pos.append(' '.join(query_pos))
					all_span.append(filtered_ans_tok_idx)
					all_token_spans.append(filtered_span)
					all_raw_ans.append('|||'.join(all_orig_answers))

				else:
					# add to final list
					all_raw_context.append(context.rstrip())
					all_context.append(' '.join(context_toks))
					all_context_sents.append(' '.join(context_toks_separated))
					all_context_pos.append(' '.join(context_pos))
					all_query.append(' '.join(query_toks))
					all_query_pos.append(' '.join(query_pos))
					all_span.append((tok_idx1, tok_idx2))
					all_token_spans.append(token_spans)
					all_raw_ans.append('|||'.join(all_orig_answers))

		print('max sent len: {0}'.format(max_sent_l))

	return (all_raw_context, all_context_sents, all_context, all_query, all_span, all_raw_ans, all_token_spans, all_context_pos, all_query_pos)


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', help="Path to the data dir", default="data/squad-v1.1/")
parser.add_argument('--data', help="Path to SQUAD json file", default="dev-v1.1.json")
parser.add_argument('--output', help="Prefix to the path of output", default="dev")
parser.add_argument('--filter', help="List of pos tags to filter out", default="")
parser.add_argument('--tag_type', help="The type of pos tag, universal/ptb", default="universal")


def main(args):
	opt = parser.parse_args(args)

	# append path
	opt.data = opt.dir + opt.data
	opt.output = opt.dir + opt.output

	raw_context, context_sents, context, query, span, raw_ans, token_spans, context_pos, query_pos = extract(opt, opt.data)
	print('{0} examples processed.'.format(len(context)))

	assert(len(raw_context) == len(context_sents))
	assert(len(query) == len(context_sents))
	assert(len(span) == len(context_sents))
	assert(len(raw_ans) == len(context_sents))
	assert(len(token_spans) == len(context_sents))
	assert(len(context) == len(context_sents))

	write_to(raw_context, opt.output + '.raw_context.txt')
	write_to(context, opt.output + '.context.txt')
	write_to(context_sents, opt.output + '.context_sent.txt')
	write_to(context_pos, opt.output + '.context_pos.txt')

	write_to(query, opt.output + '.raw_query.txt')
	write_to(query, opt.output + '.query.txt')
	write_to(query_pos, opt.output + '.query_pos.txt')

	write_to(raw_ans, opt.output + '.raw_answer.txt')

	span = ['{0} {1}'.format(p[0], p[1]) for p in span]
	write_to(span, opt.output + '.span.txt')

	token_span_ls = []
	for tok in token_spans:
		token_span_ls.append(' '.join(['{0}:{1}'.format(s,e) for (s,e) in tok]))
	write_to(token_span_ls, opt.output + '.token_span.txt')


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))


