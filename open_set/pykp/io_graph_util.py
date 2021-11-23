import numpy as np
from collections import defaultdict
import torch

from pykp import io

def adjacency_matrix_padding(matrix_list):
	max_n_nodes = max(len(x) for x in matrix_list)
	batch_size = len(matrix_list)
	matrix_pad = np.zeros((batch_size, max_n_nodes, max_n_nodes))
	for i, matrix in enumerate(matrix_list):
		l = len(matrix)
		matrix_pad[i, :l, :l] = matrix
	matrix_pad = torch.tensor(matrix_pad, dtype=torch.float)
	return matrix_pad

def field_mask(nid_set_list, n_nodes):
	mask = torch.zeros(len(nid_set_list), n_nodes, dtype=torch.float)
	for i_batch, nid_set in enumerate(nid_set_list):
		mask[i_batch, list(nid_set)] = 1
	return mask


def from_dict_to_matrix(adj_dict, n_nodes):
	""" data conversion

	Args:
		adj_dict (dict[tuple[int,int]:float]): (u,v) --> edge weight
		n_nodes (int): The number of nodes in a graph 

	Returns:
		(ndarray[np.float32], ``(n_nodes, n_nodes)``): adjacency matrix
	"""
	adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
	for (row, col), val in adj_dict.items():
		adj_matrix[row, col] = val
	return adj_matrix

def compute_distance(node_idx_ls, max_dist=-1, adj=None):
	"""Compute distance between indices in ``node_idx_ls``

	Args:
		node_idx_ls (List[int]): List of indices (aligned with adj)
		max_dist (int, optional): 이 거리 보다 멀리 떨어진 node 간의 거리는 무시 (computational efficiency 를 위해). Defaults to -1.
	"""
	if (adj is None):
		adj = defaultdict(float)
		return_flag=True
	else:
		return_flag=False
	l = len(node_idx_ls)
	for i in range(l-1):
		front_index = node_idx_ls[i]
		iterator = range(i+1, min(l, i+1+max_dist)) \
			if max_dist>-1 else range(i+1, l)
		for j in iterator:
			back_index = node_idx_ls[j]
			adj[(front_index, back_index)] += 1/(j-i)
	if return_flag:
		return dict(adj)

def construct_graph(word_seq, word_oov_seq, word_str_seq, pad=0, bos=1, eos=2):

	# Construct Nodes
	str2nid_dict = {io.PAD_WORD:0, io.BOS_WORD:1, io.EOS_WORD:2}
	nodes, nodes_oov, nodes_str = [pad, bos, eos], [pad, bos, eos], [io.PAD_WORD, io.BOS_WORD, io.EOS_WORD]
	
	node_idx_ls = []
	field_nid_set = set()
	
	for word, word_oov, word_str in zip(word_seq, word_oov_seq, word_str_seq):
		
		if word_str not in str2nid_dict:
			_node_id = len(str2nid_dict)
			str2nid_dict[word_str] = _node_id
			nodes.append(word)
			nodes_oov.append(word_oov)
			nodes_str.append(word_str)

		_node_id = str2nid_dict[word_str]
		field_nid_set.add(_node_id)
		node_idx_ls.append(_node_id)
	# Construct Edges
	adj_dict = compute_distance(node_idx_ls=node_idx_ls)
	return str2nid_dict, nodes, nodes_oov, nodes_str, adj_dict, field_nid_set

def extend_graph(word_seq, word_oov_seq, word_str_seq, 
	nodes, nodes_oov, nodes_str, str2nid_dict,
	field_nid_set=set(), adj_dict=defaultdict(float),
):
	# Extend Nodes
	node_idx_ls = []
	for word, word_oov, word_str in zip(word_seq, word_oov_seq, word_str_seq):
	
		if word_str not in str2nid_dict:
			_node_id = len(str2nid_dict)
			str2nid_dict[word_str] = _node_id
			nodes.append(word)
			nodes_oov.append(word_oov)
			nodes_str.append(word_str)
		
		_node_id = str2nid_dict[word_str]
		field_nid_set.add(_node_id)
		node_idx_ls.append(_node_id)

	# Construct Edges
	compute_distance(node_idx_ls=node_idx_ls, adj=adj_dict)
	
	return str2nid_dict, nodes, nodes_oov, nodes_str, adj_dict, field_nid_set


def _normalize_adj(A, mask):
	"""
	Args:
		A (FloatTensor, ``(batch, len, len)``): the original adjacency matrix (without self-loop)
		mask (BoolTensor,  ``(batch, len)``): mask for adjacency matrix to mask-out padding
	Returns:
		(FloatTensor, ``(batch, len, len)``): the symmetric Laplacian normalized adjacency matrix
	"""
	# print("(before eye added) A[0]: \n", A[0])
	#* :data: mask: (batch, len)``
	mask = mask.type(torch.float)
	A_mask = mask.unsqueeze(
		dim=-1 # (batch, len, 1)
	).repeat(*[1]*mask.dim(), mask.shape[-1]) # (batch, len, len)
	A_mask = A_mask * mask.unsqueeze(-2) # (batch, len, len)
	#* :result: A_mask: (batch, len, len)
	A = A + (torch.eye(A.size(1), device=A.device).unsqueeze(0).repeat(A.size(0), 1, 1) * A_mask )
	# print("(after  eye added) A[0]: \n", A[0])
	D = A.sum(dim=-1) # (batch, len)
	D_root_inverse = torch.pow(D + 1e-8, -0.5).masked_fill(D==0.0, 0) # (batch, len)
	D_root_inverse = D_root_inverse * mask
	# print("D_root_inverse[0]: \n", D_root_inverse);exit()
	A_normalized = A*D_root_inverse.unsqueeze(-1)*D_root_inverse.unsqueeze(-2) # (batch, len, len)
	return A_normalized # (batch, len, len)
