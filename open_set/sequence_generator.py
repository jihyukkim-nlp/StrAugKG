"""
Adapted from
OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py
and seq2seq-keyphrase-pytorch: https://github.com/memray/seq2seq-keyphrase-pytorch
"""

import sys
import torch
import pykp
import logging
from beam import Beam
from beam import GNMTGlobalScorer

EPS = 1e-8

class SequenceGenerator(object):
	"""Class to generate sequences from an image-to-text model."""

	def __init__(self,
				 model,
				 eos_idx,
				 bos_idx,
				 pad_idx,
				 beam_size,
				 max_sequence_length,
				 copy_attn=False,
				 include_attn_dist=True,
				 cuda=True,
				 n_best=None,
				 block_ngram_repeat=0,
				 ignore_when_blocking=[],
				 ):
		"""Initializes the generator.

		Args:
		  model: recurrent model, with inputs: (input, dec_hidden) and outputs len(vocab) values
		  eos_idx: the idx of the <eos> token
		  beam_size: Beam size to use when generating sequences.
		  max_sequence_length: The maximum sequence length before stopping the search.
		  include_attn_dist: include the attention distribution in the sequence obj or not.
		  length_normalization_factor: If != 0, a number x such that sequences are
			scored by logprob/length^x, rather than logprob. This changes the
			relative scores of sequences depending on their lengths. For example, if
			x > 0 then longer sequences will be favored.
			alpha in: https://arxiv.org/abs/1609.08144
		  length_normalization_const: 5 in https://arxiv.org/abs/1609.08144
		"""
		self.model = model
		self.eos_idx = eos_idx
		self.bos_idx = bos_idx
		self.pad_idx = pad_idx
		self.beam_size = beam_size
		self.max_sequence_length = max_sequence_length
		self.include_attn_dist = include_attn_dist
		self.copy_attn = copy_attn
		self.global_scorer = GNMTGlobalScorer()
		self.cuda = cuda
		self.block_ngram_repeat = block_ngram_repeat
		self.ignore_when_blocking = ignore_when_blocking
		if n_best is None:
			self.n_best = self.beam_size
		else:
			self.n_best = n_best
		
	def beam_search(self, 
		nodes, nodes_oov, nodes_mask, 
		src_adj, ret_adj, 
		src_mask, ret_mask,
		title_mask, title_adj, 
		oov_lists, word2idx, max_eos_per_output_seq=1,
	):
		"""
		:param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
		:param src_lens: a list containing the length of src sequences for each batch, with len=batch
		:param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
		:param src_mask: a FloatTensor, [batch, src_seq_len]
		:param oov_lists: list of oov words (idx2word) for each batch, len=batch
		:param word2idx: a dictionary
		"""
		self.model.eval()
		batch_size = nodes.size(0)
		beam_size = self.beam_size
		n_nodes = nodes.size(1)
		n_nodes_list = nodes_mask.long().sum(-1).cpu().data.numpy().tolist()
		

		# Encoding
		memory_bank, encoder_final_state = self.model.encoder(nodes=nodes, nodes_mask=nodes_mask, 
			src_adj=src_adj, ret_adj=ret_adj, title_adj=title_adj,
			src_mask=src_mask, ret_mask=ret_mask, title_mask=title_mask,
		)
		assert memory_bank.size() == torch.Size([batch_size, n_nodes, self.model.emb_dim])
		assert encoder_final_state.size() == torch.Size([batch_size, self.model.emb_dim])
		
		max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

		# Init decoder state
		decoder_init_state = self.model.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]

		# expand memory_bank, src_mask
		memory_bank = memory_bank.repeat(beam_size, 1, 1)  # [batch * beam_size, max_src_len, memory_bank_size]
		nodes_mask = nodes_mask.repeat(beam_size, 1)  # [batch * beam_size, src_seq_len]
		nodes = nodes.repeat(self.beam_size, 1)  # [batch * beam_size, src_seq_len]
		nodes_oov = nodes_oov.repeat(self.beam_size, 1)  # [batch * beam_size, src_seq_len]
		decoder_state = decoder_init_state.repeat(1, self.beam_size, 1)  # [dec_layers, batch_size * beam_size, decoder_size]

		# exclusion_list = ["<t>", "</t>", "."]
		exclusion_tokens = set([word2idx[t] for t in self.ignore_when_blocking])

		beam_list = [Beam(beam_size, n_best=self.n_best, cuda=self.cuda, global_scorer=self.global_scorer, pad=self.pad_idx, eos=self.eos_idx, bos=self.bos_idx, max_eos_per_output_seq=max_eos_per_output_seq, block_ngram_repeat=self.block_ngram_repeat, exclusion_tokens=exclusion_tokens) for _ in range(batch_size)]

		# Help functions for working with beams and batches
		def var(a):
			return a.detach().clone()

		y_decoded = None
		# Convert the contextualized embeddings into the feature space of decoder input embeddings
		memory_bank_proj = self.model.decoder.proj_src_enc_to_dec(memory_bank)
		vocab_emb_proj = self.model.decoder.proj_dec_out_to_dec(self.model.decoder.vocab_dist_linear_2.weight) # [vocab_size, decoder.emb_size]

		'''
		Run beam search.
		'''
		for t in range(1, self.max_sequence_length + 1):
			if all((b.done() for b in beam_list)):
				break

			# Construct batch x beam_size nxt words.
			# Get all the pending current beam words and arrange for forward.
			# b.get_current_tokens(): [beam_size]
			# torch.stack([ [beam of batch 1], [beam of batch 2], ... ]) -> [batch, beam]
			# after transpose -> [beam, batch]
			# After flatten, it becomes
			# [batch_1_beam_1, batch_2_beam_1,..., batch_N_beam_1, batch_1_beam_2, ..., batch_N_beam_2, ...]
			# this match the dimension of hidden state
			decoder_input = var(torch.stack([b.get_current_tokens() for b in beam_list]).t().contiguous().view(-1))
			# decoder_input: [batch_size * beam_size]

			# Turn any copied words to UNKS
			if self.copy_attn:
				decoder_input = decoder_input.masked_fill(
					decoder_input.gt(self.model.vocab_size - 1), self.model.unk_idx)

			# Convert the generated eos token to bos token, only useful in one2many_mode=2 or one2many_mode=3
			decoder_input = decoder_input.masked_fill(decoder_input == self.eos_idx, self.bos_idx)

			# run one step of decoding
			# [flattened_batch, vocab_size], [dec_layers, flattened_batch, decoder_size], [flattened_batch, memory_bank_size], [flattened_batch, src_len], [flattened_batch, src_len]
			decoder_dist, decoder_state, context, attn_dist, p_gen = \
				self.model.decoder(y=decoder_input, h=decoder_state, memory_bank=memory_bank, src_mask=nodes_mask.float(), max_num_oovs=max_num_oov, src_oov=nodes_oov, y_decoded=y_decoded,)
			log_decoder_dist = torch.log(decoder_dist + EPS)

			# Compute a vector of batch x beam word scores
			log_decoder_dist = log_decoder_dist.view(beam_size, batch_size, -1)  # [beam_size, batch_size, vocab_size]
			attn_dist = attn_dist.view(beam_size, batch_size, -1)  # [beam_size, batch_size, src_seq_len]

			# Advance each beam
			for batch_idx, beam in enumerate(beam_list):
				beam.advance(log_decoder_dist[:, batch_idx], attn_dist[:, batch_idx, :n_nodes_list[batch_idx]])
				self.beam_decoder_state_update(batch_idx, beam.get_current_origin(), decoder_state)
			
			decoder_input = var(torch.stack([b.get_current_tokens() for b in beam_list])
					  .t().contiguous().view(-1))
			y_decoded = self.model.obtain_y_decoded(y_decoded_idx=decoder_input, src_oov=nodes_oov, attn_dist=attn_dist.view(beam_size*batch_size, -1), p_gen=p_gen, memory_bank_proj=memory_bank_proj, vocab_emb_proj=vocab_emb_proj)
			

		# Extract sentences from beam.
		result_dict = self._from_beam(beam_list)
		result_dict['batch_size'] = batch_size
		return result_dict

	def _from_beam(self, beam_list):
		ret = {"predictions": [], "scores": [], "attention": []}
		for b in beam_list:
			n_best = self.n_best
			scores, ks = b.sort_finished(minimum=n_best)
			hyps, attn = [], []
			# Collect all the decoded sentences in to hyps (list of list of idx) and attn (list of tensor)
			for i, (times, k) in enumerate(ks[:n_best]):
				# Get the corresponding decoded sentence, and also the attn dist [seq_len, memory_bank_size].
				hyp, att = b.get_hyp(times, k)
				hyps.append(hyp)
				attn.append(att)
			ret["predictions"].append(hyps)  # 3d list of idx (zero dim tensor), with len [batch_size, n_best, output_seq_len]
			ret['scores'].append(scores)  # a 2d list of zero dim tensor, with len [batch_size, n_best]
			ret["attention"].append(attn)  # a 2d list of FloatTensor[output sequence length, src_len] , with len [batch_size, n_best]
		return ret

	def beam_decoder_state_update(self, batch_idx, beam_indices, decoder_state):
		"""
		:param batch_idx: int
		:param beam_indices: a long tensor of previous beam indices, size: [beam_size]
		:param decoder_state: [dec_layers, flattened_batch_size, decoder_size]
		:return:
		"""
		decoder_layers, flattened_batch_size, decoder_size = list(decoder_state.size())
		assert flattened_batch_size % self.beam_size == 0
		original_batch_size = flattened_batch_size//self.beam_size
		# select the hidden states of a particular batch, [dec_layers, batch_size * beam_size, decoder_size] -> [dec_layers, beam_size, decoder_size]
		decoder_state_transformed = decoder_state.view(decoder_layers, self.beam_size, original_batch_size, decoder_size)[:, :, batch_idx]
		# select the hidden states of the beams specified by the beam_indices -> [dec_layers, beam_size, decoder_size]
		decoder_state_transformed.data.copy_(decoder_state_transformed.data.index_select(1, beam_indices))

	def sample(self, src, src_lens, src_oov, src_mask, oov_lists, max_sample_length, greedy=False, one2many=False, one2many_mode=1, num_predictions=1, perturb_std=0, entropy_regularize=False, title=None, title_lens=None, title_mask=None):
		# src, src_lens, src_oov, src_mask, oov_lists, word2idx
		"""
		:param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
		:param src_lens: a list containing the length of src sequences for each batch, with len=batch, with oov words replaced by unk idx
		:param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
		:param src_mask: a FloatTensor, [batch, src_seq_len]
		:param oov_lists: list of oov words (idx2word) for each batch, len=batch
		:param max_sample_length: The max length of sequence that can be sampled by the model
		:param greedy: whether to sample the word with max prob at each decoding step
		:return:
		"""
		batch_size, max_src_len = list(src.size())
		max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

		# Encoding
		memory_bank, encoder_final_state = self.model.encoder(src, src_lens, src_mask, title, title_lens, title_mask)
		assert memory_bank.size() == torch.Size([batch_size, max_src_len, self.model.num_directions * self.model.encoder_size])
		assert encoder_final_state.size() == torch.Size([batch_size, self.model.num_directions * self.model.encoder_size])
		if greedy and entropy_regularize:
			raise ValueError("When using greedy, should not use entropy regularization.")

		# Init decoder state
		h_t_init = self.model.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]

		location_of_eos_for_each_batch = torch.zeros(batch_size, dtype=torch.long)

		if self.model.separate_present_absent:
			location_of_peos_for_each_batch = torch.zeros(batch_size, dtype=torch.long)
		else:
			location_of_peos_for_each_batch = None

		# init y_t to be BOS token
		y_t_init = src.new_ones(batch_size) * self.bos_idx  # [batch_size]
		sample_list = [{"prediction": [], "attention": [], "done": False} for _ in range(batch_size)]
		log_selected_token_dist = []

		unfinished_mask = src.new_ones((batch_size, 1), dtype=torch.bool)  # all seqs in a batch are unfinished at the beginning
		unfinished_mask_all = [unfinished_mask]
		pred_counters = src.new_zeros(batch_size, dtype=torch.bool)  # [batch_size]
		re_init_indicators = y_t_init == self.eos_idx
		eos_idx_mask_all = [re_init_indicators.unsqueeze(1)]

		if entropy_regularize:
			entropy = torch.zeros(batch_size).to(src.device)
		else:
			entropy = None

		for t in range(max_sample_length):
			if t > 0:
				re_init_indicators = (y_t_next == self.eos_idx)  # [batch_size]
				pred_counters += re_init_indicators
				eos_idx_mask_all.append(re_init_indicators.unsqueeze(1))
				unfinished_mask = pred_counters < num_predictions
				unfinished_mask = unfinished_mask.unsqueeze(1)
				unfinished_mask_all.append(unfinished_mask)

			if t == 0:
				h_t = h_t_init
				y_t = y_t_init
			elif one2many and one2many_mode == 2 and re_init_indicators.sum().item() > 0:
				h_t = []
				y_t = []
				for batch_idx, (indicator, pred_count) in enumerate(
					zip(re_init_indicators, pred_counters)):
					if indicator.item() == 1 and pred_count.item() < num_predictions:
						# some examples complete one keyphrase
						h_t.append(h_t_init[:, batch_idx, :].unsqueeze(1))
						y_t.append(y_t_init[batch_idx].unsqueeze(0))
					else:  # indicator.item() == 0 or indicator.item() == 1 and pred_count.item() == num_predictions:
						h_t.append(h_t_next[:, batch_idx, :].unsqueeze(1))
						y_t.append(y_t_next[batch_idx].unsqueeze(0))
				h_t = torch.cat(h_t, dim=1)  # [dec_layers, batch_size, decoder_size]
				y_t = torch.cat(y_t, dim=0)  # [batch_size]
			elif one2many and one2many_mode == 3 and re_init_indicators.sum().item() > 0:
				h_t = h_t_next
				y_t = []
				for batch_idx, (indicator, pred_count) in enumerate(
					zip(re_init_indicators, pred_counters)):
					if indicator.item() == 1 and pred_count.item() < num_predictions:
						# some examples complete one keyphrase
						# reset input to <BOS>
						y_t.append(y_t_init[batch_idx].unsqueeze(0))
						# add a noisy vector to hidden state
						if perturb_std > 0:
							'''
							if perturb_decay_along_phrases:
								perturb_std_at_t = perturb_std / pred_count.item()
							else:
								perturb_std_at_t = perturb_std
							'''
							perturb_std_at_t = perturb_std / pred_count.item()
							h_t = h_t + torch.normal(mean=0.0, std=torch.ones_like(h_t) * perturb_std_at_t)  # [dec_layers, batch_size, decoder_size]
					else:  # indicator.item() == 0 or indicator.item() == 1 and pred_count.item() == num_predictions:
						y_t.append(y_t_next[batch_idx].unsqueeze(0))
				y_t = torch.cat(y_t, dim=0)  # [batch_size]
			else:
				h_t = h_t_next
				y_t = y_t_next

			# Turn any copied words to UNKS
			if self.copy_attn:
				y_t = y_t.masked_fill(
					y_t.gt(self.model.vocab_size - 1), self.model.unk_idx)

			# [batch, vocab_size], [dec_layers, batch, decoder_size], [batch, memory_bank_size], [batch, src_len], [batch, src_len]
			decoder_dist, h_t_next, context, attn_dist, _ = self.model.decoder(y_t, h_t, memory_bank, src_mask, max_num_oov, src_oov)

			log_decoder_dist = torch.log(decoder_dist + EPS)  # [batch, vocab_size]

			if entropy_regularize:
				entropy -= torch.bmm(decoder_dist.unsqueeze(1), log_decoder_dist.unsqueeze(2)).view(batch_size)  # [batch]

			if greedy:  # greedy decoding, only use in self-critical
				selected_token_dist, prediction = torch.max(decoder_dist, 1)
				selected_token_dist = selected_token_dist.unsqueeze(1)  # [batch, 1]
				prediction = prediction.unsqueeze(1)  # [batch, 1]
				log_selected_token_dist.append(torch.log(selected_token_dist + EPS))
			else:  # sampling according to the probability distribution from the decoder
				prediction = torch.multinomial(decoder_dist, 1)  # [batch, 1]
				# select the probability of sampled tokens, and then take log, size: [batch, 1], append to a list
				log_selected_token_dist.append(log_decoder_dist.gather(1, prediction))

			for batch_idx, sample in enumerate(sample_list):
				if not sample['done']:
					sample['prediction'].append(prediction[batch_idx][0])  # 0 dim tensor
					sample['attention'].append(attn_dist[batch_idx])  # [src_len] tensor
					if int(prediction[batch_idx][0].item()) == self.model.eos_idx and pred_counters[batch_idx].item() == num_predictions-1:
						sample['done'] = True
						location_of_eos_for_each_batch[batch_idx] = t
				else:
					pass

			prediction = prediction * unfinished_mask.type_as(prediction)

			y_t_next = prediction[:, 0]  # [batch]

			if all((s['done'] for s in sample_list)):
				break

		for sample in sample_list:
			sample['attention'] = torch.stack(sample['attention'], dim=0)  # [trg_len, src_len]

		log_selected_token_dist = torch.cat(log_selected_token_dist, dim=1)  # [batch, t]
		assert log_selected_token_dist.size() == torch.Size([batch_size, t+1])

		unfinished_mask_all = torch.cat(unfinished_mask_all, dim=1).type_as(log_selected_token_dist)
		assert unfinished_mask_all.size() == log_selected_token_dist.size()

		eos_idx_mask_all = torch.cat(eos_idx_mask_all, dim=1).to(src.device)
		assert eos_idx_mask_all.size() == log_selected_token_dist.size()

		return sample_list, log_selected_token_dist, unfinished_mask_all, eos_idx_mask_all, entropy, location_of_eos_for_each_batch, location_of_peos_for_each_batch
