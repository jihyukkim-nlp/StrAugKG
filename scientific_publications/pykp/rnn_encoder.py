import logging
import torch
import torch.nn as nn
import math
import logging
from pykp.masked_softmax import MaskedSoftmax


class RNNEncoder(nn.Module):
	"""
	Base class for rnn encoder
	"""
	def forward(self, src, src_lens, src_mask=None, title=None, title_lens=None, title_mask=None):
		raise NotImplementedError

class RNNEncoderBasic(RNNEncoder):
	def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional, pad_token, dropout=0.0):
		super(RNNEncoderBasic, self).__init__()
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.bidirectional = bidirectional
		self.n_direction = 2 if self.bidirectional else 1
		self.pad_token = pad_token
		self.embedding = nn.Embedding(
			self.vocab_size,
			self.embed_size,
			self.pad_token
		)
		self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
			bidirectional=bidirectional, batch_first=True, dropout=dropout)

	def forward(self, src, src_lens, src_mask=None, title=None, title_lens=None, title_mask=None):
		"""
		:param src: [batch, src_seq_len]
		:param src_lens: a list containing the length of src sequences for each batch, with len=batch
		Other parameters will not be used in the RNNENcoderBasic class, they are here because we want to have a unify interface
		:return:
		"""
		src_embed = self.embedding(src) # [batch, src_len, embed_size]
		packed_input_src = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens, batch_first=True)
		memory_bank, encoder_final_state = self.rnn(packed_input_src)
		# ([batch, seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
		memory_bank, _ = nn.utils.rnn.pad_packed_sequence(memory_bank, batch_first=True) # unpack (back to padded)

		# only extract the final state in the last layer
		if self.bidirectional:
			encoder_last_layer_final_state = torch.cat((encoder_final_state[-1,:,:], encoder_final_state[-2,:,:]), 1) # [batch, hidden_size*2]
		else:
			encoder_last_layer_final_state = encoder_final_state[-1, :, :] # [batch, hidden_size]

		return memory_bank.contiguous(), encoder_last_layer_final_state