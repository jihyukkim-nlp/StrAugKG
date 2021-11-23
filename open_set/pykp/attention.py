import logging
import torch
import torch.nn as nn
import numpy as np
from pykp.masked_softmax import MaskedSoftmax

class Attention(nn.Module):
    def __init__(self, decoder_size, memory_bank_size, attn_mode):
        super(Attention, self).__init__()
        # attention
        if attn_mode == "concat":
            self.v = nn.Linear(decoder_size, 1, bias=False)
            self.decode_project = nn.Linear(decoder_size, decoder_size)
        self.memory_project = nn.Linear(memory_bank_size, decoder_size, bias=False)
        self.softmax = MaskedSoftmax(dim=1)
        self.tanh = nn.Tanh()
        self.attn_mode = attn_mode

    def score(self, memory_bank, decoder_state, coverage=None):
        """
        :param memory_bank: [batch_size, max_input_seq_len, self.num_directions * self.encoder_size]
        :param decoder_state: [batch_size, decoder_size]
        :param coverage: [batch_size, max_input_seq_len]
        :return: score: [batch_size, max_input_seq_len]
        """
        batch_size, max_input_seq_len, memory_bank_size = list(memory_bank.size())
        decoder_size = decoder_state.size(1)

        if self.attn_mode == "general":
            # project memory_bank
            memory_bank_ = memory_bank.view(-1,
                                            memory_bank_size)  # [batch_size*max_input_seq_len, memory_bank_size]
            encoder_feature = self.memory_project(memory_bank_)  # [batch_size*max_input_seq_len, decoder size]

            # expand decoder state
            decoder_state_expanded = decoder_state.unsqueeze(1).expand(batch_size, max_input_seq_len,
                                                                   decoder_size).contiguous()
            decoder_state_expanded = decoder_state_expanded.view(-1,
                                                             decoder_size)  # [batch_size*max_input_seq_len, decoder_size]
            # Perform bi-linear operation
            scores = torch.bmm(decoder_state_expanded.unsqueeze(1), encoder_feature.unsqueeze(2))  # [batch_size*max_input_seq_len, 1, 1]

        else:  # Bahdanau style attention
            # project memory_bank
            memory_bank_ = memory_bank.view(-1, memory_bank_size)  # [batch_size*max_input_seq_len, memory_bank_size]
            encoder_feature = self.memory_project(memory_bank_)  # [batch_size*max_input_seq_len, decoder size]
            # project decoder state
            dec_feature = self.decode_project(decoder_state)  # [batch_size, decoder_size]
            dec_feature_expanded = dec_feature.unsqueeze(1).expand(batch_size, max_input_seq_len,
                                                                   decoder_size).contiguous()
            dec_feature_expanded = dec_feature_expanded.view(-1,
                                                             decoder_size)  # [batch_size*max_input_seq_len, decoder_size]
            # sum up attention features
            att_features = encoder_feature + dec_feature_expanded  # [batch_size*max_input_seq_len, decoder_size]

            # compute attention score and normalize them
            e = self.tanh(att_features)  # [batch_size*max_input_seq_len, decoder_size]
            scores = self.v(e)  # [batch_size*max_input_seq_len, 1]

        scores = scores.view(-1, max_input_seq_len)  # [batch_size, max_input_seq_len]
        return scores

    def forward(self, decoder_state, memory_bank, src_mask=None, coverage=None):
        """
        :param decoder_state: [batch_size, decoder_size]
        :param memory_bank: [batch_size, max_input_seq_len, self.num_directions * self.encoder_size]
        :param src_mask: [batch_size, max_input_seq_len]
        :param coverage: [batch_size, max_input_seq_len]
        :return: context: [batch_size, self.num_directions * self.encoder_size], attn_dist: [batch_size, max_input_seq_len], coverage: [batch_size, max_input_seq_len]
        """
        # init dimension info
        batch_size, max_input_seq_len, memory_bank_size = list(memory_bank.size())
        #decoder_size = decoder_state.size(1)

        if src_mask is None:  # if it does not supply a source mask, create a dummy mask with all ones
            src_mask = memory_bank.new_ones(batch_size, max_input_seq_len)

        scores = self.score(memory_bank, decoder_state)  # [batch_size, max_input_seq_len]
        attn_dist = self.softmax(scores, mask=src_mask)  # src_mask: [batch_size, max_input_seq_len]

        # Compute weighted sum of memory bank features
        attn_dist = attn_dist.unsqueeze(1) # [batch_size, 1, max_input_seq_len]
        memory_bank = memory_bank.view(-1, max_input_seq_len, memory_bank_size)  # batch_size, max_input_seq_len, memory_bank_size]
        context = torch.bmm(attn_dist, memory_bank)  # [batch_size, 1, memory_bank_size]
        context = context.squeeze(1)  # [batch_size, memory_bank_size]
        attn_dist = attn_dist.squeeze(1)  # [batch_size, max_input_seq_len]

        assert attn_dist.size() == torch.Size([batch_size, max_input_seq_len])
        assert context.size() == torch.Size([batch_size, memory_bank_size])

        return context, attn_dist
