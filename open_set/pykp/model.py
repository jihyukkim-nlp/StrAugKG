import logging
import torch
import torch.nn as nn
import numpy as np
import random
import pykp
from pykp.mask import GetMask, masked_softmax, TimeDistributedDense
from pykp.rnn_encoder import *
from pykp.gcn_encoder import GraphEncoderBasic, GraphEncoderIntegrated
from pykp.rnn_decoder import StructureAwareRNNDecoder
from pykp.target_encoder import TargetEncoder
from pykp.attention import Attention

class Seq2SeqModel(nn.Module):
	"""Container module with an encoder, deocder, embeddings."""

	def __init__(self, opt):
		"""Initialize model."""
		super().__init__()
		self.vocab_size = opt.vocab_size
		self.emb_dim = opt.word_vec_size
		self.num_directions = 2 if opt.bidirectional else 1
		self.encoder_size = opt.encoder_size
		self.decoder_size = opt.decoder_size
		self.batch_size = opt.batch_size
		self.bidirectional = opt.bidirectional
		self.enc_layers = opt.enc_layers
		self.dec_layers = opt.dec_layers
		self.dropout = opt.dropout

		self.bridge = opt.bridge
		self.one2many_mode = opt.one2many_mode
		self.one2many = opt.one2many

		self.copy_attn = opt.copy_attention

		self.pad_idx_src = opt.word2idx[pykp.io.PAD_WORD]
		self.pad_idx_trg = opt.word2idx[pykp.io.PAD_WORD]
		self.bos_idx = opt.word2idx[pykp.io.BOS_WORD]
		self.eos_idx = opt.word2idx[pykp.io.EOS_WORD]
		self.unk_idx = opt.word2idx[pykp.io.UNK_WORD]
		self.sep_idx = opt.word2idx[pykp.io.SEP_WORD]

		self.share_embeddings = opt.share_embeddings

		self.attn_mode = opt.attn_mode

		self.device = opt.device

		self.separate_present_absent = opt.separate_present_absent

		self.use_title = opt.use_title

		if self.separate_present_absent:
			self.peos_idx = opt.word2idx[pykp.io.PEOS_WORD]


		self.use_title = opt.use_title
		self.encoder = GraphEncoderIntegrated(
			vocab_size=self.vocab_size,
			embed_size=self.emb_dim,
			num_layers=self.enc_layers,
			pad_token=self.pad_idx_src,
			dropout=opt.gcn_dropout,
			
			use_title=self.use_title,
		)

		self.decoder = StructureAwareRNNDecoder(
			vocab_size=self.vocab_size,
			embed_size=self.emb_dim,
			hidden_size=self.decoder_size,
			num_layers=self.dec_layers,
			memory_bank_size=self.num_directions * self.encoder_size,
			copy_attn=self.copy_attn,
			pad_idx=self.pad_idx_trg,
			attn_mode=self.attn_mode,
			dropout=self.dropout,
		)

		if self.bridge == 'dense':
			self.bridge_layer = nn.Linear(self.encoder_size * self.num_directions, self.decoder_size)
		elif opt.bridge == 'dense_nonlinear':
			self.bridge_layer = nn.tanh(nn.Linear(self.encoder_size * self.num_directions, self.decoder_size))
		else:
			self.bridge_layer = None

		if self.bridge == 'copy':
			assert self.encoder_size * self.num_directions == self.decoder_size, 'encoder hidden size and decoder hidden size are not match, please use a bridge layer'

		if self.share_embeddings:
			self.encoder.embedding.weight = self.decoder.embedding.weight
		
		self.init_weights()

	def init_weights(self):
		"""Initialize weights."""
		initrange = 0.1
		self.encoder.embedding.weight.data.uniform_(-initrange, initrange)
		if not self.share_embeddings:
			self.decoder.embedding.weight.data.uniform_(-initrange, initrange)
		
	def obtain_y_decoded(self, y_decoded_idx, src_oov, attn_dist, p_gen, memory_bank_proj, vocab_emb_proj):
		# y_decoded_idx: [batch]
		y_unk = y_decoded_idx.clone()
		y_unk[y_unk>=self.vocab_size] = self.unk_idx

		# Locate the decoded token in the given documents
		y_decoded_copy_mask = (src_oov==y_decoded_idx.unsqueeze(-1)).float()
		# Re-normalize attention weights, making the weights for the decoder input locations sum to one
		y_decoded_copy_attn_masked = attn_dist*y_decoded_copy_mask
		y_decoded_copy_attn_masked_normalized = y_decoded_copy_attn_masked/(y_decoded_copy_attn_masked.sum(-1, keepdim=True)+1e-9)
		# Get weighted sum of contextualized embeddings using the re-normalized attention weights
		y_decoded_copy = torch.matmul(memory_bank_proj.transpose(1,2), y_decoded_copy_attn_masked_normalized.unsqueeze(-1)).squeeze(-1)
		# :data y_decoded_copy (FloatTensor, ``(batch_size, embed_size)``)

		# Get uncontextualized embeddings for the decoded token
		y_decoded_vocab = vocab_emb_proj[y_unk] # [batch, embed_size]
		
		# Integrate both embeddings
		y_decoded = torch.zeros_like(y_decoded_copy)
		# 1. Copy-exclusive keywords: OOV keywords (target vocab)
		copy_exclusive_mask = y_decoded_idx>=self.vocab_size
		y_decoded[copy_exclusive_mask , :] = y_decoded_copy [copy_exclusive_mask , :]
		# 2. Gen-exclusive keywords: Unseen keywords or UNK keyword
		gen_exclusive_mask = y_decoded_copy_mask.sum(-1)==0
		gen_exclusive_mask = gen_exclusive_mask | (y_decoded_idx==self.pad_idx_src)
		gen_exclusive_mask = gen_exclusive_mask | (y_decoded_idx==self.unk_idx)
		gen_exclusive_mask = gen_exclusive_mask | (y_decoded_idx==self.eos_idx)
		y_decoded[gen_exclusive_mask, :] = y_decoded_vocab[gen_exclusive_mask, :]
		# 3. Integration: SEEN & Included in target vocabulary
		both_included_mask = ((~copy_exclusive_mask) & (~gen_exclusive_mask))
		p_gen_ = p_gen[both_included_mask, :] # [batch, 1]
		y_decoded[both_included_mask, :] = p_gen_ * y_decoded_vocab[both_included_mask, :] + ( 1 - p_gen_ ) * y_decoded_copy[both_included_mask, :]
		return y_decoded

		
	def forward(self, 
		nodes, nodes_oov, nodes_mask, 
		src_adj, ret_adj, 
		src_mask, ret_mask,
		trg, trg_oov, max_num_oov,
		title_mask=None, title_adj=None, 
	):
		batch_size, n_nodes = tuple(nodes.size())
		assert tuple(nodes.size()) == tuple(nodes_oov.size()) == tuple(nodes_mask.size())
		assert tuple(nodes_mask.size()) == tuple(src_mask.size()) == tuple(ret_mask.size())
		assert tuple(src_adj.size()) == tuple(ret_adj.size())
		if self.use_title:
			assert (title_mask is not None)
			assert tuple(nodes_mask.size()) == tuple(title_mask.size())
			assert (title_adj is not None)
			assert tuple(title_adj.size()) == tuple(src_adj.size())

		# Encoding
		memory_bank, encoder_final_state = self.encoder(nodes=nodes, nodes_mask=nodes_mask, 
			src_adj=src_adj, ret_adj=ret_adj, title_adj=title_adj,
			src_mask=src_mask, ret_mask=ret_mask, title_mask=title_mask,
		)
		assert memory_bank.size() == torch.Size([batch_size, n_nodes, self.emb_dim])
		assert encoder_final_state.size() == torch.Size([batch_size, self.emb_dim])

		# Decoding
		h_t_init = self.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]
		max_target_length = trg.size(1)

		decoder_dist_all = []
		attention_dist_all = []

		# init y_t to be BOS token
		y_t_init = trg.new_ones(batch_size) * self.bos_idx  # [batch_size]
		y_decoded = None
		# Convert the contextualized embeddings into the feature space of decoder input embeddings
		memory_bank_proj = self.decoder.proj_src_enc_to_dec(memory_bank) # [batch, max_seq_lens, decoder.embed_size]
		vocab_emb_proj = self.decoder.proj_dec_out_to_dec(self.decoder.vocab_dist_linear_2.weight) # [vocab_size, decoder.emb_size]

		for t in range(max_target_length):

			if t == 0:
				h_t = h_t_init
				y_t = y_t_init
			else:
				h_t = h_t_next
				y_t = y_t_next
			
			decoder_dist, h_t_next, context, attn_dist, p_gen = \
				self.decoder(y=y_t, h=h_t, memory_bank=memory_bank, src_mask=nodes_mask.float(), max_num_oovs=max_num_oov, src_oov=nodes_oov, y_decoded=y_decoded,)

			decoder_dist_all.append(decoder_dist.unsqueeze(1))  # [batch, 1, vocab_size]
			attention_dist_all.append(attn_dist.unsqueeze(1))  # [batch, 1, src_seq_len]
			y_t_next = trg[:, t]  # [batch]

			y_decoded = self.obtain_y_decoded(y_decoded_idx=trg_oov[:, t], src_oov=nodes_oov, attn_dist=attn_dist, p_gen=p_gen, memory_bank_proj=memory_bank_proj, vocab_emb_proj=vocab_emb_proj)

		decoder_dist_all = torch.cat(decoder_dist_all, dim=1)  # [batch_size, trg_len, vocab_size]
		attention_dist_all = torch.cat(attention_dist_all, dim=1)  # [batch_size, trg_len, src_len]
		if self.copy_attn:
			assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size + max_num_oov))
		else:
			assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size))
		assert attention_dist_all.size() == torch.Size((batch_size, max_target_length, n_nodes))

		return decoder_dist_all, h_t_next, attention_dist_all, encoder_final_state

	def tensor_2dlist_to_tensor(self, tensor_2d_list, batch_size, hidden_size, seq_lens):
		"""
		:param tensor_2d_list: a 2d list of tensor with size=[hidden_size], len(tensor_2d_list)=batch_size, len(tensor_2d_list[i])=seq_len[i]
		:param batch_size:
		:param hidden_size:
		:param seq_lens: a list that store the seq len of each batch, with len=batch_size
		:return: [batch_size, hidden_size, max_seq_len]
		"""
		# assert tensor_2d_list[0][0].size() == torch.Size([hidden_size])
		max_seq_len = max(seq_lens)
		for i in range(batch_size):
			for j in range(max_seq_len - seq_lens[i]):
				tensor_2d_list[i].append( torch.ones(hidden_size).to(self.device) * self.pad_idx_trg )  # [hidden_size]
			tensor_2d_list[i] = torch.stack(tensor_2d_list[i], dim=1)  # [hidden_size, max_seq_len]
		tensor_3d = torch.stack(tensor_2d_list, dim=0)  # [batch_size, hidden_size, max_seq_len]
		return tensor_3d

	def init_decoder_state(self, encoder_final_state):
		"""
		:param encoder_final_state: [batch_size, self.num_directions * self.encoder_size]
		:return: [1, batch_size, decoder_size]
		"""
		batch_size = encoder_final_state.size(0)
		if self.bridge == 'none':
			decoder_init_state = None
		elif self.bridge == 'copy':
			decoder_init_state = encoder_final_state
		else:
			decoder_init_state = self.bridge_layer(encoder_final_state)
		decoder_init_state = decoder_init_state.unsqueeze(0).expand((self.dec_layers, batch_size, self.decoder_size))
		# [dec_layers, batch_size, decoder_size]
		return decoder_init_state

	def init_context(self, memory_bank):
		# Init by max pooling, may support other initialization later
		context, _ = memory_bank.max(dim=1)
		return context
