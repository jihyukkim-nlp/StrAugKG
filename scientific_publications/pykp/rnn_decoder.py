import logging
import torch
import torch.nn as nn
from pykp.attention import Attention
import numpy as np
from pykp.masked_softmax import MaskedSoftmax
import math
import logging
from pykp.target_encoder import TargetEncoder

class StructureAwareRNNDecoder(nn.Module):
	def __init__(self, 
	vocab_size, embed_size, hidden_size, num_layers, memory_bank_size, copy_attn,
	pad_idx, attn_mode, dropout=0.0,):
		super().__init__()

		self.proj_src_enc_to_dec = nn.Linear(memory_bank_size, embed_size)
		self.proj_dec_out_to_dec = nn.Linear(hidden_size, embed_size)

		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.vocab_size = vocab_size
		self.memory_bank_size = memory_bank_size
		self.dropout = nn.Dropout(dropout)
		self.copy_attn = copy_attn
		self.pad_token = pad_idx
		self.embedding = nn.Embedding(
			self.vocab_size,
			self.embed_size,
			self.pad_token
		)
		self.input_size = embed_size

		self.rnn = nn.GRU(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers,
						  bidirectional=False, batch_first=False, dropout=dropout)
		self.attention_layer = Attention(
			decoder_size=hidden_size,
			memory_bank_size=memory_bank_size,
			attn_mode=attn_mode
		)
		if copy_attn:
			p_gen_input_size = embed_size + hidden_size + memory_bank_size
			self.p_gen_linear = nn.Linear(p_gen_input_size, 1)

		self.sigmoid = nn.Sigmoid()

		self.vocab_dist_linear_1 = nn.Linear(hidden_size + memory_bank_size, hidden_size)
		self.vocab_dist_linear_2 = nn.Linear(hidden_size, vocab_size)
		self.softmax = MaskedSoftmax(dim=1)

	def forward(self, y, h, memory_bank, src_mask, max_num_oovs, src_oov, y_decoded=None,):
		"""
		:param y: [batch_size]
		:param h: [num_layers, batch_size, decoder_size]
		:param memory_bank: [batch_size, max_src_seq_len, memory_bank_size]
		:param src_mask: [batch_size, max_src_seq_len]
		:param max_num_oovs: int
		:param src_oov: [batch_size, max_src_seq_len]
		:param target_encoder_state: [1, batch_size, target_encoder_size]
		:param y_decoded: [batch_size, decoder_size]
		:return:
		"""
		batch_size, max_src_seq_len = list(src_oov.size())
		assert y.size() == torch.Size([batch_size])
		assert h.size() == torch.Size([self.num_layers, batch_size, self.hidden_size])

		# 
		# init input embedding
		if (y_decoded is not None):
			# Use structure-aware representations obtained from encoder
			y_emb = y_decoded.unsqueeze(0) # [1, batch_size, embed_size]
		else:
			# Use word embeddings from predefined vocabulary
			y_emb = self.embedding(y).unsqueeze(0) # [1, batch_size, embed_size]
		# pass the concatenation of the input embedding and context vector to the RNN
		# insert one dimension to the context tensor

		rnn_input = y_emb

		if self.num_layers > 1:
			h = h.contiguous()

		_, h_next = self.rnn(rnn_input, h)

		assert h_next.size() == torch.Size([self.num_layers, batch_size, self.hidden_size])

		last_layer_h_next = h_next[-1,:,:]  # [batch, decoder_size]

		# apply attention, get input-aware context vector, attention distribution and update the coverage vector
		context, attn_dist = self.attention_layer(last_layer_h_next, memory_bank, src_mask)
		# context: [batch_size, memory_bank_size], attn_dist: [batch_size, max_input_seq_len]
		assert context.size() == torch.Size([batch_size, self.memory_bank_size])
		assert attn_dist.size() == torch.Size([batch_size, max_src_seq_len])

		vocab_dist_input = torch.cat((context, last_layer_h_next), dim=1)  # [B, memory_bank_size + decoder_size]

		vocab_dist = self.softmax(self.vocab_dist_linear_2(self.dropout(self.vocab_dist_linear_1(vocab_dist_input))))

		p_gen = None
		if self.copy_attn:
			p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0)), dim=1)  # [B, memory_bank_size + decoder_size + embed_size]
			p_gen = self.sigmoid(self.p_gen_linear(p_gen_input))

			vocab_dist_ = p_gen * vocab_dist
			attn_dist_ = (1-p_gen) * attn_dist

			if max_num_oovs > 0:
				extra_zeros = vocab_dist_.new_zeros((batch_size, max_num_oovs))
				vocab_dist_ = torch.cat((vocab_dist_, extra_zeros), dim=1)
			final_dist = vocab_dist_.scatter_add(1, src_oov, attn_dist_)
			assert final_dist.size() == torch.Size([batch_size, self.vocab_size + max_num_oovs])
		else:
			final_dist = vocab_dist
			assert final_dist.size() == torch.Size([batch_size, self.vocab_size])

		return final_dist, h_next, context, attn_dist, p_gen
