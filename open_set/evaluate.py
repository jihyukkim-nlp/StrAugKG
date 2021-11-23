#from nltk.stem.porter import *
import torch
from pykp.masked_loss import masked_cross_entropy
from utils.statistics import LossStatistics, RewardStatistics
import time
from utils.time_log import time_since
import pykp
import logging
import numpy as np
from collections import defaultdict
import os
import sys
from utils.string_helper import *
from pykp.reward import sample_list_to_str_2dlist, compute_batch_reward

def evaluate_loss(data_loader, model, opt):
	model.eval()
	evaluation_loss_sum = 0.0
	total_trg_tokens = 0
	n_batch = 0
	loss_compute_time_total = 0.0
	forward_time_total = 0.0

	with torch.no_grad():
		for batch_i, batch in enumerate(data_loader):

			nodes, nodes_oov, nodes_mask, src_adj, ret_adj, title_adj, src_mask, ret_mask, title_mask, trg, trg_oov, trg_mask, trg_lens, oov_lists, nodes_str_list = batch

			max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

			batch_size = len(nodes)
			n_batch += batch_size

			# move data to GPU if available
			nodes = nodes.to(opt.device)
			nodes_oov = nodes_oov.to(opt.device)
			nodes_mask = nodes_mask.to(opt.device)
			src_adj = src_adj.to(opt.device)
			ret_adj = ret_adj.to(opt.device)
			src_mask = src_mask.to(opt.device)
			ret_mask = ret_mask.to(opt.device)
			# 
			trg = trg.to(opt.device)
			trg_mask = trg_mask.to(opt.device)
			trg_oov = trg_oov.to(opt.device)

			if opt.use_title:
				title_mask = title_mask.to(opt.device)
				title_adj = title_adj.to(opt.device)

			start_time = time.time()
			# if not opt.one2many:
			decoder_dist, h_t, attention_dist, encoder_final_state = \
				model(
					nodes=nodes, nodes_oov=nodes_oov, nodes_mask=nodes_mask, 
					src_adj=src_adj, ret_adj=ret_adj, 
					src_mask=src_mask, ret_mask=ret_mask,
					trg=trg, trg_oov=trg_oov, max_num_oov=max_num_oov, 
					title_mask=title_mask, title_adj=title_adj, 
				)
			forward_time = time_since(start_time)
			forward_time_total += forward_time

			start_time = time.time()
			if opt.copy_attention:  # Compute the loss using target with oov words
				loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,)
			else:  # Compute the loss using target without oov words
				loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,)
			loss_compute_time = time_since(start_time)
			loss_compute_time_total += loss_compute_time

			evaluation_loss_sum += loss.item()
			total_trg_tokens += sum(trg_lens)

	eval_loss_stat = LossStatistics(evaluation_loss_sum, total_trg_tokens, n_batch, forward_time=forward_time_total, loss_compute_time=loss_compute_time_total)
	return eval_loss_stat

def evaluate_reward(data_loader, generator, opt):
	"""Return the avg. reward in the validation dataset"""
	generator.model.eval()
	final_reward_sum = 0.0
	n_batch = 0
	sample_time_total = 0.0
	topk = opt.topk
	reward_type = opt.reward_type
	match_type = opt.match_type
	eos_idx = opt.word2idx[pykp.io.EOS_WORD]
	delimiter_word = opt.delimiter_word
	one2many = opt.one2many
	one2many_mode = opt.one2many_mode
	if one2many and one2many_mode > 1:
		num_predictions = opt.num_predictions
	else:
		num_predictions = 1

	with torch.no_grad():
		for batch_i, batch in enumerate(data_loader):
			# load one2many dataset
			src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _, title, title_oov, title_lens, title_mask = batch
			num_trgs = [len(trg_str_list) for trg_str_list in
						trg_str_2dlist]  # a list of num of targets in each batch, with len=batch_size

			batch_size = src.size(0)
			n_batch += batch_size

			# move data to GPU if available
			src = src.to(opt.device)
			src_mask = src_mask.to(opt.device)
			src_oov = src_oov.to(opt.device)
			if opt.use_title:
				title = title.to(opt.device)
				title_mask = title_mask.to(opt.device)
				# title_oov = title_oov.to(opt.device)

			start_time = time.time()
			# sample a sequence
			# sample_list is a list of dict, {"prediction": [], "scores": [], "attention": [], "done": True}, preidiction is a list of 0 dim tensors
			sample_list, log_selected_token_dist, output_mask, pred_idx_mask, _, _, _ = generator.sample(
				src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=True, one2many=one2many,
				one2many_mode=one2many_mode, num_predictions=num_predictions, perturb_std=0, title=title, title_lens=title_lens, title_mask=title_mask)
			pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx,
														delimiter_word, opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk,
														src_str_list)
			sample_time = time_since(start_time)
			sample_time_total += sample_time

			final_reward = compute_batch_reward(pred_str_2dlist, trg_str_2dlist, batch_size, reward_type, topk, match_type, regularization_factor=0.0)  # np.array, [batch_size]

			final_reward_sum += final_reward.sum(0)

	eval_reward_stat = RewardStatistics(final_reward_sum, pg_loss=0, n_batch=n_batch, sample_time=sample_time_total)

	return eval_reward_stat


def preprocess_beam_search_result(beam_search_result, idx2word, vocab_size, oov_lists, eos_idx, unk_idx, replace_unk, src_str_list):
	batch_size = beam_search_result['batch_size']
	predictions = beam_search_result['predictions']
	scores = beam_search_result['scores']
	attention = beam_search_result['attention']
	assert len(predictions) == batch_size
	pred_list = []  # a list of dict, with len = batch_size
	for pred_n_best, score_n_best, attn_n_best, oov, src_word_list in zip(predictions, scores, attention, oov_lists, src_str_list):
		# attn_n_best: list of tensor with size [trg_len, src_len], len=n_best
		pred_dict = {}
		sentences_n_best = []
		for pred, attn in zip(pred_n_best, attn_n_best):
			sentence = prediction_to_sentence(pred, idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk, src_word_list, attn)
			sentences_n_best.append(sentence)
		pred_dict['sentences'] = sentences_n_best  # a list of list of word, with len [n_best, out_seq_len], does not include tbe final <EOS>
		pred_dict['scores'] = score_n_best  # a list of zero dim tensor, with len [n_best]
		pred_dict['attention'] = attn_n_best  # a list of FloatTensor[output sequence length, src_len], with len = [n_best]
		pred_list.append(pred_dict)
	return pred_list


def evaluate_beam_search(generator, one2many_data_loader, opt, delimiter_word='<sep>'):
	#score_dict_all = defaultdict(list)  # {'precision@5':[],'recall@5':[],'f1_score@5':[],'num_matches@5':[],'precision@10':[],'recall@10':[],'f1score@10':[],'num_matches@10':[]}
	# file for storing the predicted keyphrases
	if opt.pred_file_prefix == "":
		pred_output_file = open(os.path.join(opt.pred_path, "predictions.txt"), "w", encoding="utf-8")
	else:
		pred_output_file = open(os.path.join(opt.pred_path, "%s_predictions.txt" % opt.pred_file_prefix), "w", encoding="utf-8")
	# debug
	interval = 1000

	with torch.no_grad():
		start_time = time.time()
		for batch_i, batch in enumerate(one2many_data_loader):
			if (batch_i + 1) % interval == 0:
				print("Batch %d: Time for running beam search on %d batches : %.1f" % (batch_i+1, interval, time_since(start_time)))
				sys.stdout.flush()
				start_time = time.time()
			nodes, nodes_oov, nodes_mask, src_adj, ret_adj, title_adj, src_mask, ret_mask, title_mask, trg, trg_oov, trg_mask, trg_lens, oov_lists, nodes_str_list = batch

			# move data to GPU if available
			nodes = nodes.to(opt.device)
			nodes_oov = nodes_oov.to(opt.device)
			nodes_mask = nodes_mask.to(opt.device)
			src_adj = src_adj.to(opt.device)
			ret_adj = ret_adj.to(opt.device)
			src_mask = src_mask.to(opt.device)
			ret_mask = ret_mask.to(opt.device)
			if opt.use_title:
				title_mask = title_mask.to(opt.device)
				title_adj = title_adj.to(opt.device)

			beam_search_result = generator.beam_search(
				nodes=nodes, nodes_oov=nodes_oov, nodes_mask=nodes_mask, 
				src_adj=src_adj, ret_adj=ret_adj, 
				src_mask=src_mask, ret_mask=ret_mask,
				title_mask=title_mask, title_adj=title_adj, 

				oov_lists=oov_lists, word2idx=opt.word2idx, max_eos_per_output_seq=opt.max_eos_per_output_seq,
			)
			pred_list = preprocess_beam_search_result(
				beam_search_result, opt.idx2word, opt.vocab_size, oov_lists, 
				opt.word2idx[pykp.io.EOS_WORD], opt.word2idx[pykp.io.UNK_WORD], 
				opt.replace_unk, 
				nodes_str_list,
			)
			# list of {"sentences": [], "scores": [], "attention": []}


			# Process every src in the batch
			for src_str, pred, oov in zip(nodes_str_list, pred_list, oov_lists):
				# src_str: a list of words; trg_str: a list of keyphrases, each keyphrase is a list of words
				# pred_seq_list: a list of sequence objects, sorted by scores
				# oov: a list of oov words
				pred_str_list = pred['sentences']  # predicted sentences from a single src, a list of list of word, with len=[beam_size, out_seq_len], does not include the final <EOS>
				pred_score_list = pred['scores']
				pred_attn_list = pred['attention']  # a list of FloatTensor[output sequence length, src_len], with len = [n_best]

				if opt.one2many:
					all_keyphrase_list = []  # a list of word list contains all the keyphrases in the top max_n sequences decoded by beam search
					for word_list in pred_str_list:
						all_keyphrase_list += split_word_list_by_delimiter(word_list, delimiter_word, opt.separate_present_absent, pykp.io.PEOS_WORD)
					pred_str_list = all_keyphrase_list

				# output the predicted keyphrases to a file
				pred_print_out = ''
				pred_str_list = [_ for _ in pred_str_list if '<eos>' not in _]
				for word_list_i, word_list in enumerate(pred_str_list):
					if word_list_i < len(pred_str_list) - 1:
						pred_print_out += '%s;' % ' '.join(word_list)
					else:
						pred_print_out += '%s' % ' '.join(word_list)
				pred_print_out += '\n'
				pred_output_file.write(pred_print_out)

	pred_output_file.close()
	print("done!")



if __name__ == '__main__':
	pass