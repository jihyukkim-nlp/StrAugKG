# -*- coding: utf-8 -*-
"""
Python File Template
Built on the source code of seq2seq-keyphrase-pytorch: https://github.com/memray/seq2seq-keyphrase-pytorch
"""
import codecs
import inspect
import itertools
import json
import re
import traceback
from collections import Counter
from collections import defaultdict
from typing import DefaultDict
import numpy as np
import sys
from tqdm import tqdm

import torch
import torch.utils.data

from threading import Lock

from pykp import io_graph_util

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
SEP_WORD = '<sep>'
DIGIT = '<digit>'
PEOS_WORD = '<peos>'

class KeyphraseDatasetGraphIntegrated(torch.utils.data.Dataset):
	def __init__(self, examples, word2idx, idx2word, vocab_size, device, type='one2many', delimiter_type=0, load_train=True, remove_src_eos=False, use_title=False):
		
		# keys of matter. `src_oov_map` is for mapping pointed word to dict, `oov_dict` is for determining the dim of predicted logit: dim=vocab_size+max_oov_dict_in_batch
		assert type in ['one2one', 'one2many']
		if type == 'one2one':
			keys = ['src', 'trg', 'trg_copy', 'src_oov', 'oov_dict', 'oov_list', 'src_str',]
		elif type == 'one2many':
			keys = ['src', 'src_oov', 'oov_dict', 'oov_list', 'src_str', 'trg_str', 'trg', 'trg_copy', ]

		keys += ['ret', 'ret_oov', 'ret_str',]

		if use_title:
			keys += ['title', 'title_oov', 'title_str']

		filtered_examples = []

		for e in tqdm(examples):
			filtered_example = {}
			for k in keys:
				filtered_example[k] = e[k]
				
			if 'oov_list' in filtered_example:
				filtered_example['oov_number'] = len(filtered_example['oov_list'])

			filtered_examples.append(filtered_example)
		
		self.examples = filtered_examples
		self.word2idx = word2idx
		self.id2xword = idx2word
		self.vocab_size = vocab_size
		self.pad_idx = word2idx[PAD_WORD]
		self.type = type
		if delimiter_type == 0:
			self.delimiter = self.word2idx[SEP_WORD]
		else:
			self.delimiter = self.word2idx[EOS_WORD]
		self.load_train = load_train
		self.remove_src_eos = remove_src_eos
		self.use_title = use_title
		self.device = device

	def __getitem__(self, index):
		return self.examples[index]

	def __len__(self):
		return len(self.examples)

	def _pad(self, input_list):
		input_list_lens = [len(l) for l in input_list]
		max_seq_len = max(input_list_lens)
		padded_batch = self.pad_idx * np.ones((len(input_list), max_seq_len))

		for j in range(len(input_list)):
			current_len = input_list_lens[j]
			padded_batch[j][:current_len] = input_list[j]

		padded_batch = torch.LongTensor(padded_batch)

		input_mask = torch.ne(padded_batch, self.pad_idx)
		input_mask = input_mask.type(torch.FloatTensor)

		return padded_batch, input_list_lens, input_mask
	

	def collate_fn_one2one(self, batches):
		'''
		Puts each data field into a tensor with outer dimension batch size"
		'''
		assert self.type == 'one2one', 'The type of dataset should be one2one.'

		nodes_batch, nodes_oov_batch, nodes_str_batch = [], [], []
		src_adj_matrix_batch, ret_adj_matrix_batch = [], []
		src_nid_set_batch, ret_nid_set_batch = [], []
		if self.use_title:
			title_adj_matrix_batch = []
			title_nid_set_batch = []

		for i_batch, b in enumerate(batches):
			# ---------------------------------------------------------------------------------------
			# Get data
			src, src_oov, src_str = b['src'], b['src_oov'], b['src_str']
			if self.use_title:
				title, title_oov, title_str = b['title'], b['title_oov'], b['title_str']
			ret_2dlist, ret_oov_2dlist, ret_str_2dlist = b['ret'], b['ret_oov'], b['ret_str']
			

			
			# ---------------------------------------------------------------------------------------
			# Construct Graph
			src = [self.word2idx[BOS_WORD]] + src.copy() + [self.word2idx[EOS_WORD]]
			src_str = [BOS_WORD] + src_str.copy() + [EOS_WORD]
			src_oov = [self.word2idx[BOS_WORD]] + src_oov.copy() + [self.word2idx[EOS_WORD]]
			str2nid_dict, nodes, nodes_oov, nodes_str, src_adj_dict, src_nid_set = io_graph_util.construct_graph(
				word_seq=src, word_oov_seq=src_oov, word_str_seq=src_str, 
				pad=self.word2idx[PAD_WORD], bos=self.word2idx[BOS_WORD], eos=self.word2idx[EOS_WORD],
			)



			# ---------------------------------------------------------------------------------------
			# Extend Graph
			
			# 1. Title
			if self.use_title:
				# try:
				title = [self.word2idx[BOS_WORD]] + title.copy() + [self.word2idx[EOS_WORD]]
				title_str = [BOS_WORD] + title_str.copy() + [EOS_WORD]
				title_oov = [self.word2idx[BOS_WORD]] + title_oov.copy() + [self.word2idx[EOS_WORD]]
				_n_src_nodes = len(str2nid_dict)
				str2nid_dict, nodes, nodes_oov, nodes_str, title_adj_dict, title_nid_set = \
					io_graph_util.extend_graph(word_seq=title, word_oov_seq=title_oov, word_str_seq=title_str, 
					nodes=nodes, nodes_oov=nodes_oov, nodes_str=nodes_str, str2nid_dict=str2nid_dict,
					field_nid_set=set(), adj_dict=defaultdict(float),
				)
				# Source nodes subsume title nodes
				assert _n_src_nodes == len(str2nid_dict) == len(nodes) == len(nodes_oov) == len(nodes_str)
			
			# 2. Retrieved Keyphrases
			ret_nid_set = set()
			ret_adj_dict = defaultdict(float)
			for ret, ret_oov, ret_str in zip(ret_2dlist, ret_oov_2dlist, ret_str_2dlist):
				ret = [self.word2idx[BOS_WORD]] + ret.copy() + [self.word2idx[EOS_WORD]]
				ret_str = [BOS_WORD] + ret_str.copy() + [EOS_WORD]
				ret_oov = [self.word2idx[BOS_WORD]] + ret_oov.copy() + [self.word2idx[EOS_WORD]]
				str2nid_dict, nodes, nodes_oov, nodes_str, ret_adj_dict, ret_nid_set =\
					io_graph_util.extend_graph(word_seq=ret, word_oov_seq=ret_oov, word_str_seq=ret_str, 
					nodes=nodes, nodes_oov=nodes_oov, nodes_str=nodes_str, str2nid_dict=str2nid_dict,
					field_nid_set=ret_nid_set, adj_dict=ret_adj_dict,
				)

			# Validity Check
			assert len(str2nid_dict) == len(nodes) == len(nodes_oov) == len(nodes_str)



			# ---------------------------------------------------------------------------------------
			# Convert Adjacency Dictionary to Adjacency Matrix
			n_nodes = len(str2nid_dict)
			src_adj_matrix = io_graph_util.from_dict_to_matrix(adj_dict=src_adj_dict, n_nodes=n_nodes)
			ret_adj_matrix = io_graph_util.from_dict_to_matrix(adj_dict=ret_adj_dict, n_nodes=n_nodes)
			if self.use_title:
				title_adj_matrix = io_graph_util.from_dict_to_matrix(adj_dict=title_adj_dict, n_nodes=n_nodes)



			# ---------------------------------------------------------------------------------------
			# Append result to batch list
			nodes_batch.append(nodes), nodes_oov_batch.append(nodes_oov), nodes_str_batch.append(nodes_str)
			src_adj_matrix_batch.append(src_adj_matrix), ret_adj_matrix_batch.append(ret_adj_matrix)
			src_nid_set_batch.append(src_nid_set), ret_nid_set_batch.append(ret_nid_set)
			if self.use_title:
				title_adj_matrix_batch.append(title_adj_matrix)
				title_nid_set_batch.append(title_nid_set)
			
			
		# ---------------------------------------------------------------------------------------
		# Batchify Graph Data
		
		# 1. Padding Nodes
		nodes, _, nodes_mask = self._pad(input_list=nodes_batch)
		nodes_oov, _, _ = self._pad(input_list=nodes_oov_batch)
		nodes_str_list = nodes_str_batch

		# 2. Padding Adjacency Matrices
		src_adj = io_graph_util.adjacency_matrix_padding(matrix_list=src_adj_matrix_batch)
		ret_adj = io_graph_util.adjacency_matrix_padding(matrix_list=ret_adj_matrix_batch)
		if self.use_title:
			title_adj = io_graph_util.adjacency_matrix_padding(matrix_list=title_adj_matrix_batch)
		else:
			title_adj = None
		
		# 3. Masks for Each Field
		src_mask = io_graph_util.field_mask(nid_set_list=src_nid_set_batch, n_nodes=nodes.shape[1])
		ret_mask = io_graph_util.field_mask(nid_set_list=ret_nid_set_batch, n_nodes=nodes.shape[1])
		if self.use_title:
			title_mask = io_graph_util.field_mask(nid_set_list=title_nid_set_batch, n_nodes=nodes.shape[1])
		else:
			title_mask = None

		# target_input: input to decoder, ends with <eos> and oovs are replaced with <unk>
		trg = [b['trg'] + [self.word2idx[EOS_WORD]] for b in batches]

		# target for copy model, ends with <eos>, oovs are replaced with temporary idx, e.g. 50000, 50001 etc.)
		trg_oov = [b['trg_copy'] + [self.word2idx[EOS_WORD]] for b in batches]

		oov_lists = [b['oov_list'] for b in batches]

		# pad the src and target sequences with <pad> token and convert to LongTensor
		trg, trg_lens, trg_mask = self._pad(trg)
		trg_oov, _, _ = self._pad(trg_oov)

		

		# nodes						: LongTensor  [batch, n_nodes]
		# nodes_oov					: LongTensor  [batch, n_nodes]
		# nodes_mask				: FloatTensor [batch, n_nodes]
		# src_adj					: FloatTensor [batch, n_nodes, n_nodes]
		# ret_adj					: FloatTensor [batch, n_nodes, n_nodes]
		# title_adj 				: FloatTensor [batch, n_nodes, n_nodes], (optinal, None)
		# src_mask					: FloatTensor [batch, n_nodes]
		# ret_mask					: FloatTensor [batch, n_nodes]
		# title_mask				: FloatTensor [batch, n_nodes]
		# trg						: LongTensor  [batch, max_trg_len]
		# trg_oov					: LongTensor  [batch, max_trg_len]
		# trg_mask					: FloatTensor [batch, max_trg_len]
		# trg_lens					: List [ Int ]
		# oov_lists					: List [ Str ]

		return nodes, nodes_oov, nodes_mask, src_adj, ret_adj, title_adj, src_mask, ret_mask, title_mask, trg, trg_oov, trg_mask, trg_lens, oov_lists, nodes_str_list

	def collate_fn_one2many(self, batches):
		'''
		Puts each data field into a tensor with outer dimension batch size"
		'''
		assert self.type == 'one2many', 'The type of dataset should be one2one.'

		nodes_batch, nodes_oov_batch, nodes_str_batch = [], [], []
		src_adj_matrix_batch, ret_adj_matrix_batch = [], []
		src_nid_set_batch, ret_nid_set_batch = [], []
		if self.use_title:
			title_adj_matrix_batch = []
			title_nid_set_batch = []

		for i_batch, b in enumerate(batches):
			# ---------------------------------------------------------------------------------------
			# Get data
			src, src_oov, src_str = b['src'], b['src_oov'], b['src_str']
			if self.use_title:
				title, title_oov, title_str = b['title'], b['title_oov'], b['title_str']
			ret_2dlist, ret_oov_2dlist, ret_str_2dlist = b['ret'], b['ret_oov'], b['ret_str']
			
			# ---------------------------------------------------------------------------------------
			# Construct Graph
			src = [self.word2idx[BOS_WORD]] + src.copy()  + [self.word2idx[EOS_WORD]]
			src_str = [BOS_WORD] + src_str.copy()  + [EOS_WORD]
			src_oov = [self.word2idx[BOS_WORD]] + src_oov.copy()  + [self.word2idx[EOS_WORD]]
			str2nid_dict, nodes, nodes_oov, nodes_str, src_adj_dict, src_nid_set = io_graph_util.construct_graph(
				word_seq=src, word_oov_seq=src_oov, word_str_seq=src_str, 
				pad=self.word2idx[PAD_WORD], bos=self.word2idx[BOS_WORD], eos=self.word2idx[EOS_WORD],
			)
			
			# ---------------------------------------------------------------------------------------
			# Extend Graph
			
			# 1. Title
			if self.use_title:
				title = [self.word2idx[BOS_WORD]] + title.copy() + [self.word2idx[EOS_WORD]]
				title_str = [BOS_WORD] + title_str.copy() + [EOS_WORD]
				title_oov = [self.word2idx[BOS_WORD]] + title_oov.copy() + [self.word2idx[EOS_WORD]]
				_n_src_nodes = len(str2nid_dict)
				str2nid_dict, nodes, nodes_oov, nodes_str, title_adj_dict, title_nid_set =\
					io_graph_util.extend_graph(word_seq=title, word_oov_seq=title_oov, word_str_seq=title_str, 
					nodes=nodes, nodes_oov=nodes_oov, nodes_str=nodes_str, str2nid_dict=str2nid_dict,
					field_nid_set=set(), adj_dict=defaultdict(float),
				)
				# Source nodes subsume title nodes
				assert _n_src_nodes == len(str2nid_dict) == len(nodes) == len(nodes_oov) == len(nodes_str)
			
			# 2. Retrieved Keyphrases
			ret_nid_set = set()
			ret_adj_dict = defaultdict(float)
			for ret, ret_oov, ret_str in zip(ret_2dlist, ret_oov_2dlist, ret_str_2dlist):
				ret = [self.word2idx[BOS_WORD]] + ret.copy() + [self.word2idx[EOS_WORD]]
				ret_str = [BOS_WORD] + ret_str.copy() + [EOS_WORD]
				ret_oov = [self.word2idx[BOS_WORD]] + ret_oov.copy() + [self.word2idx[EOS_WORD]]
				str2nid_dict, nodes, nodes_oov, nodes_str, ret_adj_dict, ret_nid_set =\
					io_graph_util.extend_graph(word_seq=ret, word_oov_seq=ret_oov, word_str_seq=ret_str, 
					nodes=nodes, nodes_oov=nodes_oov, nodes_str=nodes_str, str2nid_dict=str2nid_dict,
					field_nid_set=ret_nid_set, adj_dict=ret_adj_dict,
				)

			# Validity Check
			assert len(str2nid_dict) == len(nodes) == len(nodes_oov) == len(nodes_str)


			# ---------------------------------------------------------------------------------------
			# Convert Adjacency Dictionary to Adjacency Matrix
			n_nodes = len(str2nid_dict)
			src_adj_matrix = io_graph_util.from_dict_to_matrix(adj_dict=src_adj_dict, n_nodes=n_nodes)
			ret_adj_matrix = io_graph_util.from_dict_to_matrix(adj_dict=ret_adj_dict, n_nodes=n_nodes)
			if self.use_title:
				title_adj_matrix = io_graph_util.from_dict_to_matrix(adj_dict=title_adj_dict, n_nodes=n_nodes)
			
			
			# ---------------------------------------------------------------------------------------
			# Append result to batch list
			nodes_batch.append(nodes), nodes_oov_batch.append(nodes_oov), nodes_str_batch.append(nodes_str)
			src_adj_matrix_batch.append(src_adj_matrix), ret_adj_matrix_batch.append(ret_adj_matrix)
			src_nid_set_batch.append(src_nid_set), ret_nid_set_batch.append(ret_nid_set)
			if self.use_title:
				title_adj_matrix_batch.append(title_adj_matrix)
				title_nid_set_batch.append(title_nid_set)
			
			
		# ---------------------------------------------------------------------------------------
		# Batchify Graph Data
		
		# 1. Padding Nodes
		nodes, _, nodes_mask = self._pad(input_list=nodes_batch)
		nodes_oov, _, _ = self._pad(input_list=nodes_oov_batch)
		nodes_str_list = nodes_str_batch

		# 2. Padding Adjacency Matrices
		src_adj = io_graph_util.adjacency_matrix_padding(matrix_list=src_adj_matrix_batch)
		ret_adj = io_graph_util.adjacency_matrix_padding(matrix_list=ret_adj_matrix_batch)
		if self.use_title:
			title_adj = io_graph_util.adjacency_matrix_padding(matrix_list=title_adj_matrix_batch)
		else:
			title_adj = None
		
		# 3. Masks for Each Field
		src_mask = io_graph_util.field_mask(nid_set_list=src_nid_set_batch, n_nodes=nodes.shape[1])
		ret_mask = io_graph_util.field_mask(nid_set_list=ret_nid_set_batch, n_nodes=nodes.shape[1])
		if self.use_title:
			title_mask = io_graph_util.field_mask(nid_set_list=title_nid_set_batch, n_nodes=nodes.shape[1])
		else:
			title_mask = None

		# trg: a list of concatenated targets, the targets in a concatenated target are separated by a delimiter, oov replaced by UNK
		# trg_oov: a list of concatenated targets, the targets in a concatenated target are separated by a delimiter, oovs are replaced with temporary idx, e.g. 50000, 50001 etc.)
		trg_str_2dlist = [b['trg_str'] for b in batches]
		if self.load_train:
			trg = []
			trg_oov = []
			for b in batches:
				trg_concat = []
				trg_oov_concat = []
				trg_size = len(b['trg'])
				assert len(b['trg']) == len(b['trg_copy'])
				for trg_idx, (trg_phase, trg_phase_oov) in enumerate(zip(b['trg'], b['trg_copy'])):
					if trg_idx == trg_size - 1:  # if this is the last keyphrase, end with <eos>
						trg_concat += trg_phase + [self.word2idx[EOS_WORD]]
						trg_oov_concat += trg_phase_oov + [self.word2idx[EOS_WORD]]
					else:
						trg_concat += trg_phase + [self.delimiter]  # trg_concat = [target_1] + [delimiter] + [target_2] + [delimiter] + ...
						trg_oov_concat += trg_phase_oov + [self.delimiter]
				trg.append(trg_concat)
				trg_oov.append(trg_oov_concat)
			
			# pad the src and target sequences with <pad> token and convert to LongTensor
			trg, trg_lens, trg_mask = self._pad(trg)
			trg_oov, _, _ = self._pad(trg_oov)
		else:
			trg, trg_lens, trg_mask, trg_oov = None, None, None, None

		oov_lists = [b['oov_list'] for b in batches]


		# nodes						: LongTensor  [batch, n_nodes]
		# nodes_oov					: LongTensor  [batch, n_nodes]
		# nodes_mask				: FloatTensor [batch, n_nodes]
		# src_adj					: FloatTensor [batch, n_nodes, n_nodes]
		# ret_adj					: FloatTensor [batch, n_nodes, n_nodes]
		# title_adj 				: FloatTensor [batch, n_nodes, n_nodes], (optinal, None)
		# src_mask					: FloatTensor [batch, n_nodes]
		# ret_mask					: FloatTensor [batch, n_nodes]
		# title_mask				: FloatTensor [batch, n_nodes]
		# trg						: LongTensor  [batch, max_trg_len]
		# trg_oov					: LongTensor  [batch, max_trg_len]
		# trg_mask					: FloatTensor [batch, max_trg_len]
		# trg_lens					: List [ Int ]
		# oov_lists					: List [ Str ]

		return nodes, nodes_oov, nodes_mask, src_adj, ret_adj, title_adj, src_mask, ret_mask, title_mask, trg, trg_oov, trg_mask, trg_lens, oov_lists, nodes_str_list
			


def build_interactive_predict_dataset(tokenized_src, tokenized_ret, word2idx, idx2word, opt, title_list=None):
	# build a dummy trg list, and then combine it with src, and pass it to the build_dataset method
	num_lines = len(tokenized_src)
	tokenized_trg = [['.']] * num_lines  # create a dummy tokenized_trg
	tokenized_src_ret_trg_pairs = list(zip(tokenized_src, tokenized_ret, tokenized_trg))
	return build_dataset(tokenized_src_ret_trg_pairs, word2idx, idx2word, opt, mode='one2many', include_original=True, title_list=title_list)

def build_dataset(src_rets_trgs_pairs, word2idx, idx2word, opt, mode='one2one', include_original=False, title_list=None):
	'''
	Standard process for copy model
	:param mode: one2one or one2many
	:param include_original: keep the original texts of source and target
	:return:
	'''
	return_examples = []
	oov_target = 0
	max_oov_len = 0
	max_oov_sent = ''
	if title_list != None:
		assert len(title_list) == len(src_rets_trgs_pairs)

	for idx, (source, retrieved, targets) in enumerate(tqdm(src_rets_trgs_pairs)):

		# if w is not seen in training data vocab (word2idx, size could be larger than opt.vocab_size), replace with <unk>
		# if w's id is larger than opt.vocab_size, replace with <unk>
		src = [word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD] for w in source]

		if title_list is not None:
			title_word_list = title_list[idx]
			#title_all = [word2idx[w] if w in word2idx else word2idx[UNK_WORD] for w in title_word_list]
			title = [word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD] for w in title_word_list]

		# create a local vocab for the current source text. If there're V words in the vocab of this string, len(itos)=V+2 (including <unk> and <pad>), len(stoi)=V+1 (including <pad>)
		src_oov, oov_dict, oov_list = extend_vocab_OOV(source, word2idx, opt.vocab_size, opt.max_unk_words)
		examples = []  # for one-to-many
		
		ret = [[word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD] for w in ret_word_list] for ret_word_list in retrieved]
		ret_oov = [
			[
				word2idx[w] 
				if w in word2idx and word2idx[w] < opt.vocab_size \
					else word2idx[UNK_WORD] if w not in oov_dict else oov_dict[w]
				for w in ret_word_list
			] 
			for ret_word_list in retrieved
		]

		for target in targets:
			example = {}
			example['ret'] = ret
			example['ret_oov'] = ret_oov

			if include_original:
				example['src_str'] = source
				example['trg_str'] = target
				example['ret_str'] = retrieved
				if title_list is not None:
					example['title_str'] = title_word_list
			
			example['src'] = src

			if title_list is not None:
				example['title'] = title

			trg = [word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD] for w in target]
			example['trg'] = trg

			example['src_oov'] = src_oov
			example['oov_dict'] = oov_dict
			example['oov_list'] = oov_list
			if len(oov_list) > max_oov_len:
				max_oov_len = len(oov_list)
				max_oov_sent = source

			# oov words are replaced with new index
			trg_copy = []
			for w in target:
				if w in word2idx and word2idx[w] < opt.vocab_size:
					trg_copy.append(word2idx[w])
				elif w in oov_dict:
					trg_copy.append(oov_dict[w])
				else:
					trg_copy.append(word2idx[UNK_WORD])
			example['trg_copy'] = trg_copy

			if title_list is not None:
				title_oov = []
				for w in title_word_list:
					if w in word2idx and word2idx[w] < opt.vocab_size:
						title_oov.append(word2idx[w])
					elif w in oov_dict:
						title_oov.append(oov_dict[w])
					else:
						title_oov.append(word2idx[UNK_WORD])
				example['title_oov'] = title_oov

			if any([w >= opt.vocab_size for w in trg_copy]):
				oov_target += 1

			if idx % 100000 == 0:
				print('-------------------- %s: %d ---------------------------' % (inspect.getframeinfo(inspect.currentframe()).function, idx))
				print('source    \n\t\t[len=%d]: %s' % (len(source), source))
				print('target    \n\t\t[len=%d]: %s' % (len(target), target))
				print('src       \n\t\t[len=%d]: %s' % (len(example['src']), example['src']))
				print('trg       \n\t\t[len=%d]: %s' % (len(example['trg']), example['trg']))

				print('src_oov   \n\t\t[len=%d]: %s' % (len(src_oov), src_oov))

				print('oov_dict         \n\t\t[len=%d]: %s' % (len(oov_dict), oov_dict))
				print('oov_list         \n\t\t[len=%d]: %s' % (len(oov_list), oov_list))
				if len(oov_dict) > 0:
					print('Find OOV in source')

				print('trg_copy         \n\t\t[len=%d]: %s' % (len(trg_copy), trg_copy))

				if any([w >= opt.vocab_size for w in trg_copy]):
					print('Find OOV in target')

			if mode == 'one2one':
				return_examples.append(example)
			else:
				examples.append(example)

		if mode == 'one2many' and len(examples) > 0:
			o2m_example = {}
			keys = examples[0].keys()
			for key in keys:
				if key.startswith('src') or key.startswith('oov') or key.startswith('title') or key.startswith('ret'):
					o2m_example[key] = examples[0][key]
				else:
					o2m_example[key] = [e[key] for e in examples]
			if include_original:
				assert len(o2m_example['src']) == len(o2m_example['src_oov']) == len(o2m_example['src_str'])
				assert len(o2m_example['oov_dict']) == len(o2m_example['oov_list'])
				assert len(o2m_example['trg']) == len(o2m_example['trg_copy']) == len(o2m_example['trg_str'])
			else:
				assert len(o2m_example['src']) == len(o2m_example['src_oov'])
				assert len(o2m_example['oov_dict']) == len(o2m_example['oov_list'])
				assert len(o2m_example['trg']) == len(o2m_example['trg_copy'])
			if title_list is not None:
				assert len(o2m_example['title']) == len(o2m_example['title_oov']) == len(o2m_example['title_str'])
			return_examples.append(o2m_example)

	print('Find #(oov_target)/#(all) = %d/%d' % (oov_target, len(return_examples)))
	print('Find max_oov_len = %d' % (max_oov_len))
	print('max_oov sentence: %s' % str(max_oov_sent))

	return return_examples


def extend_vocab_OOV(source_words, word2idx, vocab_size, max_unk_words):
	"""
	Map source words to their ids, including OOV words. Also return a list of OOVs in the article.
	WARNING: if the number of oovs in the source text is more than max_unk_words, ignore and replace them as <unk>
	Args:
		source_words: list of words (strings)
		word2idx: vocab word2idx
		vocab_size: the maximum acceptable index of word in vocab
	Returns:
		ids: A list of word ids (integers); OOVs are represented by their temporary article OOV number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
		oovs: A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers.
	"""
	src_oov = []
	oov_dict = {}
	for w in source_words:
		if w in word2idx and word2idx[w] < vocab_size:  # a OOV can be either outside the vocab or id>=vocab_size
			src_oov.append(word2idx[w])
		else:
			if len(oov_dict) < max_unk_words:
				# e.g. 50000 for the first article OOV, 50001 for the second...
				word_id = oov_dict.get(w, len(oov_dict) + vocab_size)
				oov_dict[w] = word_id
				src_oov.append(word_id)
			else:
				# exceeds the maximum number of acceptable oov words, replace it with <unk>
				word_id = word2idx[UNK_WORD]
				src_oov.append(word_id)

	oov_list = [w for w, w_id in sorted(oov_dict.items(), key=lambda x:x[1])]
	return src_oov, oov_dict, oov_list
