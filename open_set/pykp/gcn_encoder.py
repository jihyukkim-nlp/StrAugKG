import logging
from pykp import mask
import torch
import torch.nn as nn
import math
import logging
from pykp.masked_softmax import MaskedSoftmax

from pykp.io_graph_util import _normalize_adj


class GraphConvolution(nn.Module):
	def __init__(self, n_features, dropout=0.5,):
		"""GraphConvolution

		Args:
			n_features (int): the number of features (dimension of vectors)
			dropout (float, optional): Defaults to 0.5.
		"""
		super().__init__()
		self.n_features = n_features
		#* Instanciate learnable parameters for (Left/Right/Self-proximity) (l/r/s) relation modeling & context aggregation (W) and combination (G)
		self.Wl = nn.Linear(self.n_features, self.n_features)
		self.Wr = nn.Linear(self.n_features, self.n_features)
		self.Ws = nn.Linear(self.n_features, self.n_features)
		self.Gl = nn.Linear(self.n_features, self.n_features)
		self.Gr = nn.Linear(self.n_features, self.n_features)
		self.Gs = nn.Linear(self.n_features, self.n_features)
		self.dropout = nn.Dropout(p=dropout)
		
	def forward(self, input, mask, forward_adj, backward_adj):
		"""A single layer Multi-Field Graph Convolution operation 

		Args:
			input (FloatTensor, (``batch, n_nodes, n_features``)): input feature vector
			mask (BoolTensor, (``batch, n_nodes``)): input feature vector 에 대한 Mask
			forward_adj (FloatTensor, (``batch, n_nodes, n_nodes``)): forward-distance adj matrix
			backward_adj (FloatTensor, (``batch, n_nodes, n_nodes``)): backward-distance adj matrix
				(usage: modeling intra-field relation)
		Returns:
			(FloatTensor, ``(batch, n_nodes, n_features)``): output feature vector
		"""
		adj1 = forward_adj #* FloatTensor, ``(batch, n_nodes, n_nodes)``
		adj2 = backward_adj #* FloatTensor, ``(batch, n_nodes, n_nodes)``
		mask = mask.float().unsqueeze(-1)
		#* Neighborhood Context Aggregation 
		x1 = (
			torch.bmm(adj1, self.Wl(input)) #* ``(batch, n_nodes, dim)``
			+ torch.bmm(adj2, self.Wr(input)) #* ``(batch, n_nodes, dim)``
			+ self.Ws(input) #* ``(batch, n_nodes, dim)``
		)
		x1 = x1 * mask
		#* :result x1 (FloatTensor, ``(batch, n_nodes, dim)``)

		#* Combination: Residual Gated Linear Unit (GLU)
		gate = torch.sigmoid((
			torch.bmm(adj1, self.Gl(input)) #* ``(batch, n_nodes, dim)``
			+ torch.bmm(adj2, self.Gr(input)) #* ``(batch, n_nodes, dim)``
			+ self.Gs(input) #* ``(batch, n_nodes, dim)``
		))
		gate = gate * mask
		#* :result gate (FloatTensor, ``(batch, n_nodes, dim)``)
		return x1*gate

class GraphEncoder(nn.Module):
	"""
	Base class for rnn encoder
	"""
	def forward(self, src, src_lens, src_mask=None, title=None, title_lens=None, title_mask=None):
		raise NotImplementedError

class GraphEncoderIntegrated(GraphEncoder):
	def __init__(self, vocab_size, embed_size, num_layers, pad_token, dropout=0.0, use_title=False):
		super().__init__()
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.num_layers = num_layers
		self.pad_token = pad_token
		self.use_title = use_title

		# Define Layers
		self.embedding = nn.Embedding(self.vocab_size, self.embed_size, self.pad_token,)
		self.src_gcn_layers = nn.ModuleList([GraphConvolution(n_features=embed_size, dropout=dropout) for _ in range(num_layers)])
		self.ret_gcn_layers = nn.ModuleList([GraphConvolution(n_features=embed_size, dropout=dropout) for _ in range(num_layers)])
		if use_title:
			self.title_gcn_layers = nn.ModuleList([GraphConvolution(n_features=embed_size, dropout=dropout) for _ in range(num_layers)])
		self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

	def forward(self, 
		nodes, nodes_mask, 
		src_adj, ret_adj, 
		src_mask, ret_mask,
		title_adj=None, title_mask=None,
	):
		# nodes					: LongTensor 	[batch, n_nodes]
		# nodes_mask			: FloatTensor 	[batch, n_nodes]
		# [src/ret/title]_adj	: FloatTensor 	[batch, n_nodes, n_nodes]

		# Normalize Adjacency Matrix
		src_forward_adj = _normalize_adj(A=src_adj, mask=nodes_mask)
		src_backward_adj = _normalize_adj(A=src_adj.transpose(1,2), mask=nodes_mask)
		ret_forward_adj = _normalize_adj(A=ret_adj, mask=nodes_mask)
		ret_backward_adj = _normalize_adj(A=ret_adj.transpose(1,2), mask=nodes_mask)
		if self.use_title:
			title_forward_adj = _normalize_adj(A=title_adj, mask=nodes_mask)
			title_backward_adj = _normalize_adj(A=title_adj.transpose(1,2), mask=nodes_mask)
			
		node_emb = self.embedding(nodes) # [batch, n_nodes, embed_size]
		nodes_feature_mask = nodes_mask.float().unsqueeze(-1) # [batch, n_nodes, 1]

		memory_bank = node_emb
		memory_bank = memory_bank * nodes_feature_mask # [batch, n_nodes, embed_size]

		for i_layer, (src_gcn_layer, ret_gcn_layer, dropout_layer) in enumerate(zip(self.src_gcn_layers, self.ret_gcn_layers, self.dropout_layers)):
			
			_add = src_gcn_layer(
				input=memory_bank, mask=nodes_mask, 
				forward_adj=src_forward_adj, backward_adj=src_backward_adj, 
			)
			_add_meta = ret_gcn_layer(
				input=memory_bank, mask=nodes_mask, 
				forward_adj=ret_forward_adj, backward_adj=ret_backward_adj, 
			)

			if self.use_title:
				_add_meta = _add_meta + self.title_gcn_layers[i_layer](
					input=memory_bank, mask=nodes_mask, 
					forward_adj=title_forward_adj, backward_adj=title_backward_adj, 
				)

			memory_bank = memory_bank + _add + _add_meta
			memory_bank = dropout_layer(memory_bank)
			memory_bank = memory_bank * nodes_feature_mask
		# memory_bank: [batch, n_nodes, embed_size]

		masked_avg_memory = memory_bank.sum(1)/(nodes_mask.float().sum(-1).unsqueeze(-1)+1e-9)
		encoder_last_layer_final_state = masked_avg_memory

		return memory_bank, encoder_last_layer_final_state


class GraphEncoderBasic(GraphEncoder):
	def __init__(self, vocab_size, unk_idx, embed_size, num_layers, pad_token, dropout=0.0):
		super().__init__()
		self.vocab_size = vocab_size
		self.unk_idx = unk_idx
		self.embed_size = embed_size
		self.num_layers = num_layers
		self.pad_token = pad_token
		self.embedding = nn.Embedding(self.vocab_size, self.embed_size, self.pad_token,)
		self.gcn_layers = nn.ModuleList([GraphConvolution(n_features=embed_size, dropout=dropout) for _ in range(num_layers)])
		self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

	def forward(self, nodes, nodes_mask, forward_adj, backward_adj,):
		# nodes			: LongTensor [batch, n_nodes]; The indices include ``oov_dict`` indices
		# nodes_mask	: BoolTensor [batch, n_nodes]
		# forward_adj	: FloatTensor [batch, n_nodes, n_nodes]
		# backward_adj	: FloatTensor [batch, n_nodes, n_nodes]

		input_nodes = nodes.clone()
		input_nodes[input_nodes>=self.vocab_size] = self.unk_idx
		node_emb = self.embedding(input_nodes) # [batch, n_nodes, embed_size]
		nodes_feature_mask = nodes_mask.float().unsqueeze(-1) # [batch, n_nodes, 1]

		memory_bank = node_emb
		memory_bank = memory_bank * nodes_feature_mask # [batch, n_nodes, embed_size]
		for gcn_layer, dropout_layer in zip(self.gcn_layers, self.dropout_layers):
			_add = gcn_layer(
				input=memory_bank, mask=nodes_mask, 
				forward_adj=forward_adj, backward_adj=backward_adj, 
			)
			memory_bank = dropout_layer(memory_bank + _add)
			memory_bank = memory_bank * nodes_feature_mask
		# memory_bank: [batch, n_nodes, embed_size]

		masked_avg_memory = memory_bank.sum(1)/(nodes_mask.float().sum(-1).unsqueeze(-1)+1e-9)
		encoder_last_layer_final_state = masked_avg_memory

		return memory_bank, encoder_last_layer_final_state
