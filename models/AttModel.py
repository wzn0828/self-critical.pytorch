# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from models import model_utils
from .CaptionModel import CaptionModel
import random
import json
import numpy as np
from misc import incre_std_avg

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)


class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.encoded_feat_size = opt.encoded_feat_size
        self.input_encoding_size = opt.input_encoding_size
        # self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_attfeat_location = opt.drop_attfeat_location
        self.drop_prob_attfeat = opt.drop_prob_attfeat
        self.drop_prob_fcfeat = opt.drop_prob_fcfeat
        self.drop_prob_embed = opt.drop_prob_embed
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.tie_weights = opt.tie_weights
        self.LSTMN = opt.LSTMN
        self.adaptive_t0 = opt.adaptive_t0
        self.LN_out_embedding = opt.LN_out_embedding
        self.print_lang_weights = opt.print_lang_weights
        self.input_feature = opt.input_feature
        self.att_normalize_method = opt.att_normalize_method

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0  # Schedule sampling probability

        self.embed = nn.Sequential(*((nn.Embedding(self.vocab_size + 1, self.input_encoding_size),) +
                                     ((nn.ReLU(),) if opt.emb_relu else ()) +
                                     ((nn.BatchNorm1d(self.input_encoding_size),) if opt.emb_use_bn else ()) +
                                     (nn.Dropout(self.drop_prob_embed),)
                                     ))
        self.fc_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.fc_feat_size, momentum=0.02),) if opt.fc_use_bn else ()) +
                (nn.Linear(self.fc_feat_size, self.encoded_feat_size),
                 nn.ReLU(),) +
                ((nn.BatchNorm1d(self.encoded_feat_size),) if opt.fc_use_bn == 2 else ()) +
                (nn.Dropout(self.drop_prob_fcfeat),)))

        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.encoded_feat_size),
                 nn.ReLU(),) +
                ((nn.BatchNorm1d(self.encoded_feat_size),) if self.use_bn == 2 else ()) +
                ((nn.Dropout(self.drop_prob_attfeat),) if self.drop_attfeat_location == 'before_attention' else ())))

        # prepair for normalize
        if self.att_normalize_method == '2' and self.use_bn == 2:
            # self.att_embed[3].bias.requires_grad = False
            # self.att_embed[3].bias = self.att_embed[3].weight.data.new_zeros(self.rnn_size)
            self.att_embed[3].bias = None

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
            model_utils.kaiming_normal('relu', 0, self.logit)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in
                          range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(
                *(reduce(lambda x, y: x + y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))

        self.ctx2att = nn.Sequential(*((nn.Linear(self.encoded_feat_size, self.att_hid_size),) +
                                       ((nn.BatchNorm1d(self.att_hid_size),) if opt.BN_other else ())
                                       ))

        if self.tie_weights:
            if self.input_encoding_size != self.rnn_size:
                raise ValueError('When using the tied flag, input_encoding_size must be equal to rnn_size')
            del self.logit
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1, bias=False)
            self.logit.weight = self.embed[0].weight
            self.project_outputembedding = nn.Linear(self.rnn_size, self.rnn_size, bias=False)
            # initialization
            nn.init.normal_(self.project_outputembedding.weight, mean=0, std=(2.0**0.5)/self.rnn_size)

        if self.LN_out_embedding:
            self.ln_out_embedding = nn.LayerNorm(self.rnn_size)

        # initialization
        nn.init.normal_(self.embed[0].weight, mean=0, std=0.2)
        model_utils.kaiming_normal('relu', 0, filter(lambda x: 'linear' in str(type(x)), self.fc_embed)[0],
                                   filter(lambda x: 'linear' in str(type(x)), self.att_embed)[0])
        model_utils.xavier_normal('tanh', self.ctx2att[0])
        self.beta = ((self.att_hid_size + self.encoded_feat_size) / (float(self.att_hid_size) + self.rnn_size + self.encoded_feat_size)) ** 0.5
        self.ctx2att[0].weight.data = self.ctx2att[0].weight*self.beta

        if self.LSTMN and self.adaptive_t0:
            self.h1t0 = nn.Parameter(self.logit.weight.data.new_zeros((1, 1, self.rnn_size)))
            self.c1t0 = nn.Parameter(self.logit.weight.data.new_zeros((1, 1, self.rnn_size)))
            self.h2t0 = nn.Parameter(self.logit.weight.data.new_zeros((1, 1, self.rnn_size)))
            self.c2t0 = nn.Parameter(self.logit.weight.data.new_zeros((1, 1, self.rnn_size)))
            nn.init.normal_(self.h1t0, mean=0, std=0.2)
            nn.init.normal_(self.c1t0, mean=0, std=0.2)
            nn.init.normal_(self.h2t0, mean=0, std=0.1)
            nn.init.normal_(self.c2t0, mean=0, std=0.1)

        self.ctx2att0 = nn.Linear(self.encoded_feat_size, self.att_hid_size)
        self.ctx2att2 = nn.Linear(self.encoded_feat_size, self.att_hid_size)
        model_utils.xavier_normal('linear', self.ctx2att0, self.ctx2att2)

        self.save_att_statics = opt.save_att_statics
        if self.save_att_statics:
            self.att_feat_dict = {}

        # fc feature normalization
        self.fc_normalize_method = opt.fc_normalize_method
        if self.fc_normalize_method == '3':
            self.fc_BN = nn.BatchNorm1d(self.fc_feat_size, affine=False, momentum=0.02)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.LSTMN:
            if self.adaptive_t0:
                return (torch.cat([self.h1t0, self.c1t0, self.h2t0, self.c2t0, weight.new_zeros(1, 1, self.rnn_size)],
                                  0).expand(-1, bsz, -1),)
            else:
                return (weight.new_zeros(2 * self.num_layers + 1, bsz, self.rnn_size),)  # h01,c01,h02,c02,xt
        else:
            return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.num_layers, bsz, self.rnn_size))  # (h0, c0, sentinal)
            # weight.new_zeros(bsz, 1, self.rnn_size))       # (h0, c0, sentinal)

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        # fc normalization method 3: (attention-running_mean)/l2 + running_mean
        if self.fc_normalize_method == '3':
            l2_recip = (att_masks.data.long().sum(1).float())**0.5
            self.fc_BN(fc_feats)
            fc_feats = (fc_feats - self.fc_BN.running_mean) * l2_recip.unsqueeze(1) + self.fc_BN.running_mean
        fc_feats = self.fc_embed(fc_feats)

        # embed att feats
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        # p_att_feats = self.ctx2att(att_feats)
        p_att_feats = pack_wrapper(self.ctx2att, att_feats, att_masks)
        p0_att_feats = self.ctx2att0(att_feats)
        p2_att_feats = self.ctx2att2(att_feats)

        att_masks_expand = att_masks.unsqueeze(2).expand_as(att_feats)
        att_feats_masked = att_feats * att_masks_expand
        lens = att_masks.data.long().sum(1)
        lens = lens.unsqueeze(1).expand(lens.size(0), att_feats.size(-1))
        average_att_feat = att_feats_masked.sum(1) / lens.data.float()  # batch*rnn_size

        return fc_feats, att_feats, p_att_feats, att_masks, average_att_feat, p0_att_feats, p2_att_feats

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        '''
        :param fc_feats: size is [batch_size, 2048]
        :param att_feats: size is [batch_size, num_att(36), 2048]
        :param seq: labels, size is [batch_size, 18]
        :param att_masks:
        :return:
        '''
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)  # [num_layers, batchsize, rnn_size]

        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size + 1)  # [batch_size, 17, vocab_size+1]
        att_sentinals = []
        lang_sentinals = []
        att_hiddens = []
        lan_hiddens = []
        lang_weights = []

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, average_att_feat, p0_att_feats, p2_att_feats = self._prepare_feature(
            fc_feats,
            att_feats,
            att_masks)  # p_fc_feats [batch_size, rnn_size], p_att_feats [batch_size, 36, rnn_size], pp_att_feats [batch_size, 36, att_hid_size]
        if self.input_feature == 'att_mean':
            p_fc_feats = average_att_feat

        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    # prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    # it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs[:, i - 1].detach())  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()
                # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state, lang_weight = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p0_att_feats, p2_att_feats, p_att_masks,
                                                    average_att_feat, state)  # batch*(vocab_size+1)
            outputs[:, i] = output

            if not self.LSTMN:
                att_hiddens.append(state[0][0])
                lan_hiddens.append(state[0][1])
            else:
                att_hiddens.append(state[-1][0])
                lan_hiddens.append(state[-1][2])

            att_sentinals.append(state[-1][0])
            lang_sentinals.append(state[-1][1])
            # if i==6:
            lang_weights.append(lang_weight)

        # return outputs, p_fc_feats, p_att_feats, torch.stack(att_hiddens), torch.stack(lan_hiddens), torch.stack(
        #     sentinals)
        return outputs, p_fc_feats, p_att_feats, torch.stack(att_hiddens), torch.stack(lan_hiddens), torch.stack(
            att_sentinals), torch.stack(lang_sentinals), torch.stack(lang_weights, dim=2)

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, p0_att_feats, p2_att_feats, att_masks, average_att_feat, state):
        # layer normal output embedding matrix
        if self.LN_out_embedding:
            self.logit.weight.data = self.ln_out_embedding(self.logit.weight)

        # 'it' contains a word index
        xt = self.embed(it)  # [batch_size, input_encoding_size]

        output, state, lang_weights = self.core(xt, fc_feats, att_feats, p_att_feats, p0_att_feats, p2_att_feats, state, att_masks, self.att_embed[3].weight, self.att_embed[3].bias)  # batch*rn_size
        logprobs = F.log_softmax(self.logit(output), dim=1)  # batch*(vocab_size+1)

        return logprobs, state, lang_weights

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, average_att_feat, p0_att_feats, p2_att_feats = self._prepare_feature(
            fc_feats,
            att_feats,
            att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        self.states = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k + 1].expand(beam_size, p_fc_feats.size(1))
            tmp_att_feats = p_att_feats[k:k + 1].expand(*((beam_size,) + p_att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = pp_att_feats[k:k + 1].expand(*((beam_size,) + pp_att_feats.size()[1:])).contiguous()
            tmp_p0_att_feats = p0_att_feats[k:k + 1].expand(*((beam_size,) + p0_att_feats.size()[1:])).contiguous()
            tmp_p2_att_feats = p2_att_feats[k:k + 1].expand(*((beam_size,) + p2_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k + 1].expand(
                *((beam_size,) + p_att_masks.size()[1:])).contiguous() if att_masks is not None else None
            tmp_average_att_feat = average_att_feat[k:k + 1].expand(beam_size, average_att_feat.size(1))

            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state, lang_weights = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats,
                                                                        tmp_p_att_feats, tmp_p0_att_feats,
                                                                        tmp_p2_att_feats,
                                                                        tmp_att_masks, tmp_average_att_feat, state)

            self.done_beams[k], states = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                          tmp_p0_att_feats, tmp_p2_att_feats,
                                                          tmp_att_masks, tmp_average_att_feat, opt=opt)
            self.states[k] = [state[:, 0, :] for state in states[0][1:]]
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']

        # att_hiddens = fc_feats.new_empty((len(self.states[0]), len(self.states), self.states[0][0].size(1)))
        # lang_hiddens = fc_feats.new_empty((len(self.states[0]), len(self.states), self.states[0][0].size(1)))
        # for i, state in enumerate(self.states):
        #     for j, item in enumerate(state):
        #         att_hiddens[j, i, :] = item[0, :]
        #         lang_hiddens[j, i, :] = item[2, :]

        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):

        # # ---generate a variance statics dictionary--- #
        # if self.save_att_statics:
        #     batch_size = att_feats.size(0)
        #     num_feats = att_masks.sum(dim=1).int()
        #     for i in range(batch_size):
        #         num_feat = num_feats[i].item()
        #
        #         feat_std = att_feats[i][:num_feats[i], :].std(dim=1).mean().item()
        #         feat_mean = att_feats[i][:num_feats[i], :].mean(dim=1).mean().item()
        #
        #         self.core.generate_dict(num_feat, self.att_feat_dict, 0.0, 0.0, feat_mean, feat_std)
        # # ---generate a variance statics dictionary--- #

        att_hiddens = []
        lan_hiddens = []

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)  # (2*batch*rnn_size, 2*batch*rnn_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, average_att_feat, p0_att_feats, p2_att_feats = self._prepare_feature(
            fc_feats, att_feats,
            att_masks)  # batch*rnn_size, batch*36*rnn_size, batch*36*att_hid_size

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)  # batch*16
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)  # batch*16
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs, state, lang_weights = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p0_att_feats, p2_att_feats, p_att_masks, average_att_feat,
                                                      state)  # batch*(vocab_size+1), (2*batch*rnn_size, 2*batch*rnn_size)

            if self.print_lang_weights:
                print(lang_weights)

            if not self.LSTMN:
                att_hiddens.append(state[0][0])
                lan_hiddens.append(state[0][1])
            else:
                att_hiddens.append(state[-1][0])
                lan_hiddens.append(state[-1][2])

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seqLogprobs[:, t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs


class AdaAtt_lstm(nn.Module):
    def __init__(self, opt, use_maxout=True):
        super(AdaAtt_lstm, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        # self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_maxout = use_maxout

        # Build a LSTM
        self.w2h = nn.Linear(self.input_encoding_size, (4 + (use_maxout == True)) * self.rnn_size)
        self.v2h = nn.Linear(self.rnn_size, (4 + (use_maxout == True)) * self.rnn_size)

        self.i2h = nn.ModuleList(
            [nn.Linear(self.rnn_size, (4 + (use_maxout == True)) * self.rnn_size) for _ in range(self.num_layers - 1)])
        self.h2h = nn.ModuleList(
            [nn.Linear(self.rnn_size, (4 + (use_maxout == True)) * self.rnn_size) for _ in range(self.num_layers)])

        # Layers for getting the fake region
        if self.num_layers == 1:
            self.r_w2h = nn.Linear(self.input_encoding_size, self.rnn_size)
            self.r_v2h = nn.Linear(self.rnn_size, self.rnn_size)
        else:
            self.r_i2h = nn.Linear(self.rnn_size, self.rnn_size)
        self.r_h2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, xt, img_fc, state):

        hs = []
        cs = []
        for L in range(self.num_layers):
            # c,h from previous timesteps
            prev_h = state[0][L]
            prev_c = state[1][L]
            # the input to this layer
            if L == 0:
                x = xt
                i2h = self.w2h(x) + self.v2h(img_fc)
            else:
                x = hs[-1]
                x = F.dropout(x, self.drop_prob_lm, self.training)
                i2h = self.i2h[L - 1](x)

            all_input_sums = i2h + self.h2h[L](prev_h)

            sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
            sigmoid_chunk = F.sigmoid(sigmoid_chunk)
            # decode the gates
            in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
            forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
            out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
            # decode the write inputs
            if not self.use_maxout:
                in_transform = F.tanh(all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size))
            else:
                in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
                in_transform = torch.max( \
                    in_transform.narrow(1, 0, self.rnn_size),
                    in_transform.narrow(1, self.rnn_size, self.rnn_size))
            # perform the LSTM update
            next_c = forget_gate * prev_c + in_gate * in_transform
            # gated cells form the output
            tanh_nex_c = F.tanh(next_c)
            next_h = out_gate * tanh_nex_c
            if L == self.num_layers - 1:
                if L == 0:
                    i2h = self.r_w2h(x) + self.r_v2h(img_fc)
                else:
                    i2h = self.r_i2h(x)
                n5 = i2h + self.r_h2h(prev_h)
                fake_region = F.sigmoid(n5) * tanh_nex_c

            cs.append(next_c)
            hs.append(next_h)

        # set up the decoder
        top_h = hs[-1]
        top_h = F.dropout(top_h, self.drop_prob_lm, self.training)
        fake_region = F.dropout(fake_region, self.drop_prob_lm, self.training)

        state = (torch.cat([_.unsqueeze(0) for _ in hs], 0),
                 torch.cat([_.unsqueeze(0) for _ in cs], 0))
        return top_h, fake_region, state


class AdaAtt_attention(nn.Module):
    def __init__(self, opt):
        super(AdaAtt_attention, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        # self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_hid_size = opt.att_hid_size

        # fake region embed
        self.fr_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm))
        self.fr_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        # h out embed
        self.ho_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.Tanh(),
            nn.Dropout(self.drop_prob_lm))
        self.ho_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        self.alpha_net = nn.Linear(self.att_hid_size, 1, bias=False)
        self.att2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed, att_masks=None):
        # View into three dimensions
        att_size = conv_feat.numel() // conv_feat.size(0) // self.rnn_size
        conv_feat = conv_feat.view(-1, att_size, self.rnn_size)
        conv_feat_embed = conv_feat_embed.view(-1, att_size, self.att_hid_size)

        # view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
        fake_region = self.fr_linear(fake_region)
        fake_region_embed = self.fr_embed(fake_region)

        h_out_linear = self.ho_linear(h_out)
        h_out_embed = self.ho_embed(h_out_linear)

        txt_replicate = h_out_embed.unsqueeze(1).expand(h_out_embed.size(0), att_size + 1, h_out_embed.size(1))

        img_all = torch.cat([fake_region.view(-1, 1, self.input_encoding_size), conv_feat], 1)
        img_all_embed = torch.cat([fake_region_embed.view(-1, 1, self.input_encoding_size), conv_feat_embed], 1)

        hA = F.tanh(img_all_embed + txt_replicate)
        hA = F.dropout(hA, self.drop_prob_lm, self.training)

        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))
        PI = F.softmax(hAflat.view(-1, att_size + 1), dim=1)

        if att_masks is not None:
            att_masks = att_masks.view(-1, att_size)
            PI = PI * torch.cat([att_masks[:, :1], att_masks], 1)  # assume one one at the first time step.
            PI = PI / PI.sum(1, keepdim=True)

        visAtt = torch.bmm(PI.unsqueeze(1), img_all)
        visAttdim = visAtt.squeeze(1)

        atten_out = visAttdim + h_out_linear

        h = F.tanh(self.att2h(atten_out))
        h = F.dropout(h, self.drop_prob_lm, self.training)
        return h


class AdaAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(AdaAttCore, self).__init__()
        self.lstm = AdaAtt_lstm(opt, use_maxout)
        self.attention = AdaAtt_attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        h_out, p_out, state = self.lstm(xt, fc_feats, state)
        atten_out = self.attention(h_out, p_out, att_feats, p_att_feats, att_masks)
        return atten_out, state


class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        # lang_lstm_input = torch.cat([att, h_att], 1)    # batch_size * 2rnn_size
        # lang_lstm_input = F.dropout(lang_lstm_input, self.drop_prob_rnn, self.training)
        lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_rnn, self.training)], 1)  # ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state


class BottomUpCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(BottomUpCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        # self.attention = Attention(opt)

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, average_att_feat, att_masks=None):
        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        # att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        # lang_lstm_input = torch.cat([att, h_att], 1)    # batch_size * 2rnn_size
        # lang_lstm_input = F.dropout(lang_lstm_input, self.drop_prob_rnn, self.training)
        lang_lstm_input = torch.cat([average_att_feat, F.dropout(h_att, self.drop_prob_rnn, self.training)], 1)  # ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state


class TopDownOriginalCore(TopDownCore):
    def __init__(self, opt, use_maxout=False):
        super(TopDownOriginalCore, self).__init__(opt)
        del self.att_lstm, self.lang_lstm
        self.att_lstm = nn.LSTMCell(opt.rnn_size + opt.fc_feat_size + opt.input_encoding_size,
                                    opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.att_feat_size + opt.rnn_size, opt.rnn_size)  # h^1_t, \hat v

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)


class TopDownUpCatWeightedHiddenCore3(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownUpCatWeightedHiddenCore3, self).__init__()
        self.iteration = 0
        self.drop_prob_attfeat = opt.drop_prob_attfeat
        self.drop_attfeat_location = opt.drop_attfeat_location
        self.drop_prob_rnn1 = opt.drop_prob_rnn1
        self.drop_prob_rnn2 = opt.drop_prob_rnn2
        self.drop_prob_output = opt.drop_prob_output
        self.rnn_size = opt.rnn_size
        self.input_encoding_size = opt.input_encoding_size
        self.language_attention = opt.language_attention
        self.project_hidden = opt.project_hidden
        self.skip_connection = opt.skip_connection
        self.attention_gate = opt.attention_gate
        self.LSTMN = opt.LSTMN
        self.LSTMN_att = opt.LSTMN_att
        self.LSTMN_lang = opt.LSTMN_lang
        self.LSTMN_lang_att_score_method = opt.LSTMN_lang_att_score_method
        self.input_first_att = opt.input_first_att
        self.input_second_att = opt.input_second_att
        self.lstm_layer_norm = opt.lstm_layer_norm
        self.layer_norm = opt.layer_norm
        self.norm_input = opt.norm_input
        self.norm_output = opt.norm_output
        self.norm_hidden = opt.norm_hidden
        self.noh2pre = opt.noh2pre
        self.pre = opt.pre
        self.input_feature = opt.input_feature
        self.output_attention = opt.output_attention
        self.att_normalize_method = opt.att_normalize_method
        self.att_normalize_rate = opt.att_normalize_rate
        self.encoded_feat_size = opt.encoded_feat_size
        self.att_norm_reg_paras_path = opt.att_norm_reg_paras_path

        if self.noh2pre:
            inputsize = opt.input_encoding_size + opt.encoded_feat_size
        else:
            inputsize = opt.input_encoding_size + opt.rnn_size + opt.encoded_feat_size

        if self.lstm_layer_norm:
            self.att_lstm = model_utils.LayerNormLSTMCell(inputsize, opt.rnn_size, self.layer_norm, self.norm_input, self.norm_output, self.norm_hidden)  # we, fc, h^2_t-1
            self.lang_lstm = model_utils.LayerNormLSTMCell(opt.rnn_size + opt.encoded_feat_size, opt.rnn_size, self.layer_norm, self.norm_input, self.norm_output, self.norm_hidden)  # h^1_t, \hat v
        else:
            self.att_lstm = nn.LSTMCell(inputsize, opt.rnn_size)  # we, fc, h^2_t-1
            self.lang_lstm = nn.LSTMCell(opt.rnn_size + opt.encoded_feat_size, opt.rnn_size)  # h^1_t, \hat v

        if self.LSTMN:
            if self.input_first_att == 1:
                input_first_att_size = self.input_encoding_size
            elif self.input_first_att == 2:
                input_first_att_size = self.input_encoding_size + self.rnn_size
            elif self.input_first_att == 3:
                input_first_att_size = self.input_encoding_size + 2*self.rnn_size
            self.intra_att_att_lstm = IntraAttention(opt, input_first_att_size, opt.LSTMN_att_att_score_method)

            if self.input_second_att == 1:
                input_second_att_size = self.rnn_size
            if self.input_second_att == 2:
                input_second_att_size = 2*self.rnn_size
            self.intra_att_lang_lstm = IntraAttention(opt, input_second_att_size, opt.LSTMN_lang_att_score_method)

        ### --- visual attention --- ###
        self.attention = Attention(opt)

        if opt.weighted_hidden:
            self.sen_attention = SentinalAttention(opt)
        else:
            self.sen_attention = self.average_hiddens

        # -------generate sentinal--------#
        self.transfer_horizontal = opt.transfer_horizontal
        self.sentinal_embed1 = nn.Linear(opt.rnn_size, opt.rnn_size, bias=False)
        self.sentinal_embed2 = lambda x: x

        # output
        self.h2_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.ws_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.drop = nn.Dropout(self.drop_prob_rnn2)

        if opt.nonlinear == 'relu':
            self.tgh = nn.ReLU()
            model_utils.kaiming_normal('relu', 0, self.h2_affine, self.ws_affine)
        elif opt.nonlinear == 'prelu':
            self.tgh = nn.PReLU()
            model_utils.kaiming_normal('leaky_relu', 0.25, self.h2_affine, self.ws_affine)
        elif opt.nonlinear == 'lecun_tanh':
            self.tgh = lambda x: 1.7159 * F.tanh((2.0 / 3.0) * x)
            model_utils.xavier_normal('tanh', self.h2_affine, self.ws_affine)
        elif opt.nonlinear == 'maxout':
            del self.h2_affine, self.ws_affine
            self.h2_affine = nn.Linear(opt.rnn_size, 2 * opt.rnn_size)
            self.ws_affine = nn.Linear(opt.rnn_size, 2 * opt.rnn_size)
            self.tgh = lambda x: torch.max(x.narrow(1, 0, self.rnn_size),
                                           x.narrow(1, self.rnn_size, self.rnn_size))
            model_utils.xavier_normal('linear', self.h2_affine, self.ws_affine)
        elif opt.nonlinear == 'tanh':
            self.tgh = nn.Tanh()
            model_utils.xavier_normal('tanh', self.h2_affine, self.ws_affine)
        elif opt.nonlinear == 'x':
            self.tgh = lambda x: x
            model_utils.xavier_normal('linear', self.h2_affine, self.ws_affine)

        if opt.sentinel_nonlinear == 'relu':
            self.sentinel_nonlinear = nn.ReLU()
            model_utils.kaiming_normal('relu', 0, self.sentinal_embed1)
        elif opt.sentinel_nonlinear == 'prelu':
            self.sentinel_nonlinear = nn.PReLU()
            model_utils.kaiming_normal('leaky_relu', 0.25, self.sentinal_embed1)
        elif opt.sentinel_nonlinear == 'lecun_tanh':
            self.sentinel_nonlinear = lambda x: 1.7159 * F.tanh((2.0 / 3.0) * x)
            model_utils.xavier_normal('tanh', self.sentinal_embed1)
        elif opt.sentinel_nonlinear == 'maxout':
            del self.sentinal_embed1
            self.sentinal_embed1 = nn.Linear(opt.rnn_size, 2 * opt.rnn_size, bias=False)
            self.sentinel_nonlinear = lambda x: torch.max(x.narrow(1, 0, self.rnn_size),
                                           x.narrow(1, self.rnn_size, self.rnn_size))
            model_utils.xavier_normal('linear', self.sentinal_embed1)
        elif opt.sentinel_nonlinear == 'tanh':
            self.sentinel_nonlinear = nn.Tanh()
            model_utils.xavier_normal('tanh', self.sentinal_embed1)
        elif opt.sentinel_nonlinear == 'x':
            del self.sentinal_embed1
            self.sentinal_embed1 = lambda x: x
            self.sentinel_nonlinear = lambda x: x

        if self.skip_connection == 'CAT':
            self.h1_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
            self.beta = ((opt.rnn_size + opt.rnn_size)/(2*opt.rnn_size + float(opt.rnn_size)))**0.5
            model_utils.xavier_normal(opt.nonlinear, self.h1_affine)
            self.h1_affine.weight.data = self.h1_affine.weight * self.beta
            self.h2_affine.weight.data = self.h2_affine.weight * self.beta

        if not self.project_hidden:
            del self.h2_affine
            self.h2_affine = lambda x: x
            del self.drop
            self.drop = nn.Dropout(0)
            if self.skip_connection == 'CAT':
                del self.h1_affine
                self.h1_affine = lambda x: x

        if self.attention_gate:
            self.att_gate_1 = nn.Linear(opt.rnn_size + inputsize,
                                             inputsize)
            self.att_gate_2 = nn.Linear(3 * opt.rnn_size, 2 * opt.rnn_size)

            init.orthogonal_(self.att_gate_1.weight)
            init.orthogonal_(self.att_gate_2.weight)
            init.constant_(self.att_gate_1.bias, 1.0)
            init.constant_(self.att_gate_2.bias, 1.0)

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)

        if self.language_attention and self.LSTMN:
            self.Intergrate2vector = Intergrate2vector(opt, opt.Intergrate2vector)
            self.Intergrate_instead = opt.Intergrate_instead
            self.lang_att_before_pro = opt.lang_att_before_pro

        # dynamically nonlocal
        self.nonlocal_dy = opt.nonlocal_dy
        if self.nonlocal_dy:
            self.dy_nonlocal = NonLocalBlock(opt, self.rnn_size)

        if self.input_feature == 'attention':
            self.attention0 = Attention(opt)

        if self.output_attention:
            self.attention2 = Attention(opt)
            self.outatt_affine = nn.Linear(opt.rnn_size, opt.rnn_size, bias=False)
            model_utils.xavier_normal('linear', self.outatt_affine)

        self.save_att_statics = opt.save_att_statics
        if opt.save_att_statics:
            self.att_statics_numfeat_path = opt.att_statics_numfeat_path
            self.att_statics_weights_l2_path = opt.att_statics_weights_l2_path
            self.att_statics_numfeat = {}
            self.att_statics_weights_l2 = {}
            self.dim_att_statics_numfeat_path = opt.dim_att_statics_numfeat_path
            self.dim_att_statics_weights_l2_path = opt.dim_att_statics_weights_l2_path
            self.dim_att_statics_numboxes = {}
            self.dim_att_statics_l2weight = {}

        # normalize attentioned feature
        if self.att_normalize_method == '2':
            self.att_bias = nn.Parameter(self.h2_affine.weight.data.new_zeros((1, self.rnn_size)))
        elif self.att_normalize_method == '3':
            self.att_BN = nn.BatchNorm1d(self.rnn_size, affine=False, momentum=0.01)
        elif self.att_normalize_method == '4-0':
            self.att_r = self.h2_affine.weight.data.new_ones(1).cuda()
            self.att_bias = self.h2_affine.weight.data.new_zeros(1).cuda()
        elif self.att_normalize_method == '4-1':
            self.att_r = nn.Parameter(self.h2_affine.weight.data.new_ones(1))
            self.att_bias = self.h2_affine.weight.data.new_zeros(1).cuda()
        elif self.att_normalize_method == '4-2':
            self.att_r = nn.Parameter(self.h2_affine.weight.data.new_ones(1))
            self.att_bias = nn.Parameter(self.h2_affine.weight.data.new_zeros(1))
        elif self.att_normalize_method is not None and '5' in self.att_normalize_method:
            self.att_normalize_rate = 1
            self.att_mean = nn.Linear(1, self.encoded_feat_size)
            self.att_std = nn.Linear(1, self.encoded_feat_size)
            # initialization
            self.att_mean.weight.data.zero_()
            self.att_mean.bias.data.zero_()
            self.att_std.weight.data.fill_(opt.att_std_weight_init)
            self.att_std.bias.data.fill_(opt.att_std_bias_init)
            if self.att_normalize_method == '5-0' or self.att_normalize_method == '5-1':
                del self.att_mean
                self.att_mean = lambda x: 0
            if self.att_normalize_method == '5-1' or self.att_normalize_method == '5-3':
                self.att_linear_project = Linear_Project(self.encoded_feat_size, has_bias=True)
            else:
                self.att_linear_project = lambda x: x

        elif self.att_normalize_method is not None and '6' in self.att_normalize_method:
            self.att_normalize_rate = 1
            self.att_norm = att_normalization(opt)
            if self.att_normalize_method == '6-1' or self.att_normalize_method == '6-1-0' or self.att_normalize_method == '6-3':
                self.att_linear_project = Linear_Project(self.encoded_feat_size, has_bias=True)
            else:
                self.att_linear_project = lambda x: x
        elif self.att_normalize_method is not None and '9' in self.att_normalize_method:
            self.att_BN = nn.BatchNorm1d(self.encoded_feat_size, affine=False)
            if self.att_normalize_method == '9-0':
                self.attstd_linear_project = Linear_Project(1, has_bias=True)
            elif self.att_normalize_method == '9-1':
                self.attstd_linear_project = Linear_Project(self.encoded_feat_size, has_bias=True)
            self.att_linear_project = Linear_Project(self.encoded_feat_size, has_bias=True)
        elif self.att_normalize_method == 't1':
            self.att_linear_project = Linear_Project(self.encoded_feat_size, has_bias=True)
        elif self.att_normalize_method == 't2':
            self.att_BN = nn.BatchNorm1d(self.encoded_feat_size, affine=False, momentum=0.01)
            self.att_linear_project = Linear_Project(self.encoded_feat_size, has_bias=True)
        elif self.att_normalize_method == 't3':
            self.att_BN = nn.BatchNorm1d(self.encoded_feat_size, affine=True, momentum=0.01)
        elif self.att_normalize_method == 't4':
            self.att_BN = nn.BatchNorm1d(self.encoded_feat_size, affine=False, momentum=0.01)
            self.att_bias = nn.Parameter(self.h2_affine.weight.data.new_zeros((1, self.encoded_feat_size)))

        if self.skip_connection == 'HW':
            self.HW_connection = HW_connection(num_dim=self.rnn_size, normal=False)
        elif self.skip_connection == 'HW-normal':
            self.HW_connection = HW_connection(num_dim=self.rnn_size, normal=True)


    def forward(self, xt, fc_feats, att_feats, p_att_feats, p0_att_feats, p2_att_feats, state, att_masks=None, std_feat=None, mean_feat=None):
        pre_states = state[1:]
        step = len(pre_states)
        if self.LSTMN and step > 0:
            xi = torch.cat([_[4].unsqueeze(1) for _ in pre_states], 1)  # [batch_size, step, rnn_size]

        # --start-------first LSTM-------- #
        if self.LSTMN:
            h1 = state[-1][0]
            h2 = state[-1][2]
            c2 = state[-1][3]
        else:
            h1 = state[0][0]
            h2 = state[0][1]
            c2 = state[1][1]

        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        weight0 = att_masks.view(-1, att_size).float()
        weight0 = weight0 / weight0.sum(1, keepdim=True)  # normalize to 1
        if self.input_feature == 'attention':
            fc_feats, weight0 = self.attention0(h1, att_feats, p0_att_feats, att_masks)  # batch_size * rnn_size

        if self.noh2pre:
            att_lstm_input = torch.cat([fc_feats, xt], 1)  # [batch_size, rnn_size + input_encoding_size]
        else:
            if self.pre == 'h':
                pre = h2
            elif self.pre == 'c':
                pre = c2
            prev_h = F.dropout(pre, self.drop_prob_rnn1, self.training)  # [batch_size, rnn_size]
            att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        if self.LSTMN:
            if step < 2 or not self.LSTMN_att:
                h1 = state[-1][0]
                c1 = state[-1][1]
                layer1_weights = h1.new_zeros((h1.size(0), 1))
            else:
                last_att_h1 = state[0][0]
                hi = torch.cat([_[0].unsqueeze(1) for _ in pre_states], 1)  # [batch_size, step, rnn_size]
                ci = torch.cat([_[1].unsqueeze(1) for _ in pre_states], 1)  # [batch_size, step, rnn_size]
                if self.input_first_att == 1:
                    input_first_att = xt
                elif self.input_first_att == 2:
                    input_first_att = torch.cat([h2, xt], 1)
                elif self.input_first_att == 3:
                    input_first_att = torch.cat([h2, fc_feats, xt], 1)
                h1, c1, layer1_weights = self.intra_att_att_lstm(input_first_att, last_att_h1, hi, ci, xi)   # batch * rnn_size, batch * num_hidden
        else:
            h1 = state[0][0]
            c1 = state[1][0]

        if self.attention_gate:
            att_lstm_input = torch.mul(att_lstm_input, F.sigmoid(
                self.att_gate_1(torch.cat((att_lstm_input, h1), 1))))

        h_att, c_att = self.att_lstm(att_lstm_input, (h1, c1))  # both are [batch_size, rnn_size]
        # --end-------first LSTM-------- #

        # --start-------second LSTM-------- #
        att, weight = self.attention(h_att, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size
        # normalize attention:
        # compute l2
        if self.att_normalize_method is not None:
            if self.att_normalize_method not in ['4-0', '4-1', '4-2'] and self.att_normalize_method != 't1' and self.att_normalize_method != 't3':
                batch_size = att_feats.size(0)
                l2_weight = att_feats.new_ones(batch_size)       # batch_size
                num_feats = att_masks.sum(dim=1).int()
                for i in range(batch_size):
                    l2_weight[i] = weight[i][:num_feats[i]].norm().item()
                l2_weight = l2_weight*self.att_normalize_rate        # batch_size
                l2_weight = l2_weight.unsqueeze(1)              # batch_size * 1
            # method 1: attention/l2
            if self.att_normalize_method == '1':
                att = att/l2_weight
            # method 2: attention/l2 + bias(parameter), previous attfeat project's BN is set to no bias
            elif self.att_normalize_method == '2':
                att = att/l2_weight + self.att_bias
            # method 3: (attention-running_mean)/l2 + running_mean
            elif self.att_normalize_method == '3':
                self.att_BN(att)
                att = (att - self.att_BN.running_mean)/l2_weight + self.att_BN.running_mean
            # method 4:
            elif self.att_normalize_method in ['4-0', '4-1', '4-2']:
                att = (att-att.mean(dim=1, keepdim=True)) / att.std(dim=1, keepdim=True) * self.att_r + self.att_bias
            # method 5:
            elif '5' in self.att_normalize_method:
                att = (att - self.att_mean(l2_weight)) / self.att_std(l2_weight)
                att = self.att_linear_project(att)
            # method 6:
            elif '6' in self.att_normalize_method:
                att = self.att_linear_project(self.att_norm(l2_weight, att))
            # method 9:
            elif '9' in self.att_normalize_method:
                self.att_BN(att)
                mean = self.att_BN.running_mean
                std = torch.mm(l2_weight, std_feat.unsqueeze(0))
                std = self.attstd_linear_project(std)
                att = (att - mean)/std
                att = self.att_linear_project(att)
            # method 10:
            elif self.att_normalize_method =='10':
                att = (att - mean_feat) / l2_weight + mean_feat
            # method t1
            elif self.att_normalize_method == 't1':
                att = (att - att.mean(dim=1, keepdim=True)) / att.std(dim=1, keepdim=True)
                att = self.att_linear_project(att)
            # method t2
            elif self.att_normalize_method == 't2':
                self.att_BN(att)
                att = (att - self.att_BN.running_mean) / l2_weight
                att = self.att_linear_project(att)
            # method t3
            elif self.att_normalize_method == 't3':
                att = self.att_BN(att)
            # method t4
            elif self.att_normalize_method == 't4':
                self.att_BN(att)
                att = (att - self.att_BN.running_mean) / l2_weight + self.att_bias


        if self.drop_attfeat_location == 'after_attention':
            att = F.dropout(att, self.drop_prob_attfeat, self.training)

        droped_h_att = F.dropout(h_att, self.drop_prob_rnn1, self.training)
        lang_lstm_input = torch.cat([att, droped_h_att], 1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        if self.LSTMN:
            if step < 2 or not self.LSTMN_lang:
                h2 = state[-1][2]
                c2 = state[-1][3]
                layer2_weights = h2.new_zeros((h2.size(0), 1))
            else:
                last_att_h2 = state[0][1]
                hi = torch.cat([_[2].unsqueeze(1) for _ in pre_states], 1)  # [batch_size, step, rnn_size]
                ci = torch.cat([_[3].unsqueeze(1) for _ in pre_states], 1)  # [batch_size, step, rnn_size]
                if self.input_second_att == 1:
                    input_second_att = h_att
                elif self.input_second_att == 2:
                    input_second_att = torch.cat([att, h_att], 1)
                h2, c2, layer2_weights = self.intra_att_lang_lstm(input_second_att, last_att_h2, hi,
                                                                 ci, xi)  # batch * rnn_size, batch * num_hidden
        if self.attention_gate:
            lang_lstm_input = torch.mul(lang_lstm_input, F.sigmoid(self.att_gate_2(torch.cat((lang_lstm_input, h2), 1))))

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (h2, c2))  # batch*rnn_size
        # --end------second LSTM--------#

        # --start-------generate output--------#
        lang_weights = torch.zeros_like(h2)
        if self.language_attention:
            if self.LSTMN:
                if self.lang_att_before_pro:
                    p_h_lang = h_lang
                else:
                    p_h_lang = self.h2_affine(self.drop(h_lang))

                if step > 0:
                    p_h_pre, lang_weights = self.sen_attention(p_h_lang, xi)            # batch * rnn_size, batch * num_hidden
                    inter_v, prob_v1 = self.Intergrate2vector(p_h_pre, p_h_lang)        # batch*rnn_size, batch*1
                    lang_weights = prob_v1 * lang_weights
                    lang_weights = torch.cat((lang_weights, (1.0 - prob_v1)), 1)        # batch * num_hidden
                else:
                    inter_v = p_h_lang
                    lang_weights = torch.zeros_like(h2)

                if self.lang_att_before_pro:
                    affined_linear = self.h2_affine(self.drop(inter_v))
                else:
                    affined_linear = inter_v  # batch*rnn_size

            else:
                pre_sentinal = state[2:]
                # sentinal = F.dropout(sentinal, self.drop_prob_rnn, self.training)
                sentinal = torch.cat([_[0].unsqueeze(1) for _ in pre_sentinal],
                                     1)  # [batch_size, num_recurrent, rnn_size]
                weighted_sentinal, lang_weights = self.sen_attention(h_lang, sentinal)  # batch_size * rnn_size
                affined_linear = self.h2_affine(self.drop(h_lang)) + self.ws_affine(
                        self.drop(weighted_sentinal))  # batch_size * rnn_size
        else:
            if self.skip_connection == 'CAT':
                affined_linear = self.h2_affine(self.drop(h_lang)) + self.h1_affine(self.drop(h_att))
            elif self.skip_connection == 'RES':
                affined_linear = self.h2_affine(self.drop(h_lang + h_att))
            elif self.skip_connection in ['HW', 'HW-normal']:
                affined_linear, trans_gate, carry_gate = self.HW_connection(h_att, h_lang)      # batch*rnn_size,batch*1,batch*1
                affined_linear = self.h2_affine(self.drop(affined_linear))
                lang_weights = torch.cat((trans_gate, carry_gate), dim=1)      # batch*2
            else:
                affined_linear = self.h2_affine(self.drop(h_lang))

        if self.output_attention:
            att_output, weight2 = self.attention2(h_lang, att_feats, p2_att_feats, att_masks)  # batch_size * rnn_size
            affined_linear += self.outatt_affine(self.drop(att_output))

        affined = self.tgh(affined_linear)

        # ---non-local--- #
        if self.nonlocal_dy:
            x = affined.unsqueeze(1)                            # batch*1*rnn_size
            if self.LSTMN and step > 0:
                x = torch.cat([xi, x], 1)                        # batch*(step+1)*rnn_size
            affined, lang_weights = self.dy_nonlocal(x)         # batch * rnn_size, batch * (step+1)
        # ---non-local--- #

        output = F.dropout(affined, self.drop_prob_output, self.training)  # batch_size * rnn_size
        # --end-------generate output--------#

        # --start-------generate state--------#
        if self.LSTMN:
            if self.language_attention:
                if self.Intergrate_instead:
                    state = (torch.stack([h1, h2, h1, h2, h1]),) + tuple(pre_states) + (
                    torch.stack([h_att, c_att, h_lang, c_lang, inter_v]),)
                else:
                    state = (torch.stack([h1, h2, h1, h2, h1]),) + tuple(pre_states) + (
                        torch.stack([h_att, c_att, h_lang, c_lang, p_h_lang]),)
            else:
                state = (torch.stack([h1, h2, h1, h2, h1]),) + tuple(pre_states) + (
                torch.stack([h_att, c_att, h_lang, c_lang, affined]),)
        else:
            if self.transfer_horizontal:
                state = (torch.stack([h_att, affined]), torch.stack([c_att, c_lang]))
            else:
                state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

            if self.language_attention:
                sentinal_current = self.sentinal_embed2(self.sentinal_embed1(h_lang))  # batch* rnn_size
                sentinal_current = self.sentinel_nonlinear(sentinal_current)  # batch* rnn_size
                state = state + tuple(pre_sentinal) + (torch.stack([sentinal_current, torch.zeros_like(sentinal_current)]),)
        # --end-------generate state--------#

        # visualization
        if step == 6:
            self.iteration += 1
            if self.iteration > 10 and self.iteration % 100 == 0:
                batch_size = att_feats.size(0)
                if self.input_feature == 'attention':
                    print('The attention distribution of the input attention:')
                    random.seed(123)
                    for i in random.sample(range(batch_size), 3):
                        print(weight0[i][att_masks[i] != 0])

                print('The attention distribution of the first LSTM attention:')
                random.seed(123)
                for i in random.sample(range(batch_size), 3):
                    print(weight[i][att_masks[i] != 0])

                if self.output_attention:
                    print('The attention distribution of the output attention:')
                    random.seed(123)
                    for i in random.sample(range(batch_size), 3):
                        print(weight2[i][att_masks[i] != 0])

            # ---generate a variance statics dictionary--- #
            if self.save_att_statics:
                batch_size = att_feats.size(0)
                num_feats = att_masks.sum(dim=1).int()
                for i in range(batch_size):
                    num_feat = num_feats[i].item()
                    l2_weight = weight[i][:num_feats[i]].norm().item()
                    l2_weight = int(l2_weight * 1e3 + 0.5)

                    att_std = att[i].std().item()
                    att_mean = att[i].mean().item()
                    feat_std = att_feats[i][:num_feats[i], :].std(dim=1)
                    feat_mean = att_feats[i][:num_feats[i], :].mean(dim=1)

                    self.generate_dict(num_feat, self.att_statics_numfeat, att_mean, att_std, feat_mean, feat_std, ('l2_weight', l2_weight))
                    self.generate_dict(l2_weight, self.att_statics_weights_l2, att_mean, att_std, feat_mean, feat_std, ('num_boxes', num_feat))

                    dim_feat = att_feats[i][:num_feats[i], :]
                    dim_att = att[i][:]

                    self.generate_dim_dict(num_feat, self.dim_att_statics_numboxes, dim_feat, dim_att, ('l2_weight', l2_weight))
                    self.generate_dim_dict(l2_weight, self.dim_att_statics_l2weight, dim_feat, dim_att, ('num_boxes', num_feat))

                if self.iteration % 100 == 0:
                    # json.dump(self.att_statics_numfeat, open(self.att_statics_numfeat_path, 'w'))
                    # json.dump(self.att_statics_weights_l2, open(self.att_statics_weights_l2_path, 'w'))

                    np.save(self.att_statics_numfeat_path, self.att_statics_numfeat)
                    np.save(self.att_statics_weights_l2_path, self.att_statics_weights_l2)

                    np.save(self.dim_att_statics_numfeat_path, self.dim_att_statics_numboxes)
                    np.save(self.dim_att_statics_weights_l2_path, self.dim_att_statics_l2weight)

            # ---generate a variance statics dictionary--- #

        # end visualization #

        return output, state, lang_weights

    def generate_dict(self, key, dict, att_mean, att_std, feat_mean, feat_std, other_pair):
        value_dict = dict.get(key,
                              {'att_std': incre_std_avg.incre_std_avg(), 'att_mean': incre_std_avg.incre_std_avg(),
                               'feat_std': incre_std_avg.incre_std_avg(),
                               'feat_mean': incre_std_avg.incre_std_avg(), other_pair[0]: incre_std_avg.incre_std_avg(),
                               'num': 0})
        value_dict['num'] += 1
        value_dict['att_std'].incre_in_value(att_std)
        value_dict['att_mean'].incre_in_value(att_mean)
        value_dict['feat_std'].incre_in_list(feat_std.tolist())
        value_dict['feat_mean'].incre_in_list(feat_mean.tolist())
        value_dict[other_pair[0]].incre_in_value(other_pair[1])
        dict[key] = value_dict

    def generate_dim_dict(self, key, dict, dim_feat, dim_att, other_pair):
        value_dict = dict.get(key,
                              {'num': 0, 'dim_feat': [incre_std_avg.incre_std_avg() for _ in range(self.encoded_feat_size)],
                               'dim_att': [incre_std_avg.incre_std_avg() for _ in range(self.encoded_feat_size)], other_pair[0]: incre_std_avg.incre_std_avg()})
        value_dict['num'] += 1
        for i in range(self.encoded_feat_size):
            value_dict['dim_feat'][i].incre_in_list(dim_feat[:, i].tolist())
            value_dict['dim_att'][i].incre_in_value(dim_att[i].item())
        value_dict[other_pair[0]].incre_in_value(other_pair[1])
        dict[key] = value_dict

    def average_hiddens(self, h_lang, hiddens):
        weighted_sentinal = torch.mean(hiddens, 1)  # batch_size * rnn_size
        return weighted_sentinal


class HW_connection(nn.Module):
    def __init__(self, num_dim, normal=False):
        super(HW_connection, self).__init__()
        self.transform_gate = nn.Sequential(nn.Linear(num_dim, 1), nn.Sigmoid())
        self.carry_gate = nn.Sequential(nn.Linear(num_dim, 1), nn.Sigmoid())
        self.normal = normal

        #initialization
        model_utils.xavier_normal('linear', self.transform_gate[0], self.carry_gate[0])

    def forward(self, input_1, input_2):
        # both inputs' size maybe batch*opt.rnn_size
        trans_gate = self.transform_gate(input_1)   # batch*1
        carry_gate = self.carry_gate(input_1)       # batch*1

        if self.normal == True:
            l2 = torch.cat((trans_gate, carry_gate), dim=1).norm(p=2, dim=1, keepdim=True)  # batch*1
            trans_gate = trans_gate/l2
            carry_gate = carry_gate/l2

        output = input_1 * trans_gate + input_2 * carry_gate  # batch*opt.rnn_size

        return output, trans_gate, carry_gate



class att_normalization(nn.Module):
    def __init__(self, opt):
        super(att_normalization, self).__init__()
        self.att_norm_reg_paras_path = opt.att_norm_reg_paras_path
        self.encoded_feat_size = opt.encoded_feat_size
        self.att_normalize_method = opt.att_normalize_method

        self.att_norm_reg_paras = torch.Tensor(np.load(self.att_norm_reg_paras_path))

        self.att_norm_reg_mean_w = self.att_norm_reg_paras[0].unsqueeze(0).cuda()
        self.att_norm_reg_mean_b = self.att_norm_reg_paras[1].unsqueeze(0).cuda()
        self.att_norm_reg_std_w = self.att_norm_reg_paras[2].unsqueeze(0).cuda()
        self.att_norm_reg_std_b = self.att_norm_reg_paras[3].unsqueeze(0).cuda()

        if self.att_normalize_method == '6-2' or self.att_normalize_method == '6-3':
            self.att_norm_reg_mean_w = nn.Parameter(self.att_norm_reg_mean_w)
            self.att_norm_reg_mean_b = nn.Parameter(self.att_norm_reg_mean_b)
            self.att_norm_reg_std_w = nn.Parameter(self.att_norm_reg_std_w)
            self.att_norm_reg_std_b = nn.Parameter(self.att_norm_reg_std_b)

    def forward(self, l2_weight, attentioned_feat):
        '''
        :param l2_ewight: size is batch*1
        :param attentioned_feat: batch*encoded_feat_size
        :return: batch*encoded_feat_size
        '''
        att_mean = torch.mm(l2_weight, self.att_norm_reg_mean_w) + self.att_norm_reg_mean_b
        att_std = torch.mm(l2_weight, self.att_norm_reg_std_w) + self.att_norm_reg_std_b
        attentioned_feat = (attentioned_feat - att_mean) / att_std

        return attentioned_feat


class Linear_Project(nn.Module):
    def __init__(self, dim, has_bias=True):
        super(Linear_Project, self).__init__()
        self.dim = dim
        self.has_bias = has_bias

        self.weight = nn.Parameter(torch.ones(self.dim))
        self.bias = torch.zeros(self.dim).cuda()
        if self.has_bias:
            self.bias = nn.Parameter(self.bias)

    def forward(self, input):
        projected = self.weight * input + self.bias

        return projected



class TestCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TestCore, self).__init__()
        self.iteration = 0
        self.drop_prob_rnn1 = opt.drop_prob_rnn1
        self.drop_prob_rnn2 = opt.drop_prob_rnn2
        self.drop_prob_output = opt.drop_prob_output
        self.rnn_size = opt.rnn_size
        self.input_encoding_size = opt.input_encoding_size
        self.language_attention = opt.language_attention
        self.project_hidden = opt.project_hidden
        self.add_2_layer_hidden = opt.add_2_layer_hidden
        self.attention_gate = opt.attention_gate
        self.directly_add_2_layer = opt.directly_add_2_layer
        self.LSTMN = opt.LSTMN
        self.LSTMN_att = opt.LSTMN_att
        self.LSTMN_lang = opt.LSTMN_lang
        self.LSTMN_lang_att_score_method = opt.LSTMN_lang_att_score_method
        self.input_first_att = opt.input_first_att
        self.input_second_att = opt.input_second_att
        self.lstm_layer_norm = opt.lstm_layer_norm
        self.layer_norm = opt.layer_norm
        self.norm_input = opt.norm_input
        self.norm_output = opt.norm_output
        self.norm_hidden = opt.norm_hidden
        self.noh2pre = opt.noh2pre
        self.pre = opt.pre
        self.input_feature = opt.input_feature
        self.output_attention = opt.output_attention

        if self.noh2pre:
            inputsize = opt.input_encoding_size + opt.rnn_size
        else:
            inputsize = opt.input_encoding_size + opt.rnn_size * 2


        self.att_lstm = nn.RNNCell(inputsize, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.RNNCell(opt.rnn_size, opt.rnn_size)  # h^1_t, \hat v

        if self.LSTMN:
            if self.input_first_att == 1:
                input_first_att_size = self.input_encoding_size
            elif self.input_first_att == 2:
                input_first_att_size = self.input_encoding_size + self.rnn_size
            elif self.input_first_att == 3:
                input_first_att_size = self.input_encoding_size + 2*self.rnn_size
            self.intra_att_att_lstm = IntraAttention(opt, input_first_att_size, opt.LSTMN_att_att_score_method)

            if self.input_second_att == 1:
                input_second_att_size = self.rnn_size
            if self.input_second_att == 2:
                input_second_att_size = 2*self.rnn_size
            self.intra_att_lang_lstm = IntraAttention(opt, input_second_att_size, opt.LSTMN_lang_att_score_method)

        self.attention = Attention(opt)
        if opt.weighted_hidden:
            self.sen_attention = SentinalAttention(opt)
        else:
            self.sen_attention = self.average_hiddens

        # -------generate sentinal--------#
        self.transfer_horizontal = opt.transfer_horizontal
        self.sentinal_embed1 = nn.Linear(opt.rnn_size, opt.rnn_size, bias=False)
        self.sentinal_embed2 = lambda x: x

        # output
        self.h2_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.ws_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.drop = nn.Dropout(self.drop_prob_rnn2)

        if opt.nonlinear == 'relu':
            self.tgh = nn.ReLU()
            model_utils.kaiming_normal('relu', 0, self.h2_affine, self.ws_affine)
        elif opt.nonlinear == 'prelu':
            self.tgh = nn.PReLU()
            model_utils.kaiming_normal('leaky_relu', 0.25, self.h2_affine, self.ws_affine)
        elif opt.nonlinear == 'lecun_tanh':
            self.tgh = lambda x: 1.7159 * F.tanh((2.0 / 3.0) * x)
            model_utils.xavier_normal('tanh', self.h2_affine, self.ws_affine)
        elif opt.nonlinear == 'maxout':
            del self.h2_affine, self.ws_affine
            self.h2_affine = nn.Linear(opt.rnn_size, 2 * opt.rnn_size)
            self.ws_affine = nn.Linear(opt.rnn_size, 2 * opt.rnn_size)
            self.tgh = lambda x: torch.max(x.narrow(1, 0, self.rnn_size),
                                           x.narrow(1, self.rnn_size, self.rnn_size))
            model_utils.xavier_normal('linear', self.h2_affine, self.ws_affine)
        elif opt.nonlinear == 'tanh':
            self.tgh = nn.Tanh()
            model_utils.xavier_normal('tanh', self.h2_affine, self.ws_affine)
        elif opt.nonlinear == 'x':
            self.tgh = lambda x: x
            model_utils.xavier_normal('linear', self.h2_affine, self.ws_affine)

        if opt.sentinel_nonlinear == 'relu':
            self.sentinel_nonlinear = nn.ReLU()
            model_utils.kaiming_normal('relu', 0, self.sentinal_embed1)
        elif opt.sentinel_nonlinear == 'prelu':
            self.sentinel_nonlinear = nn.PReLU()
            model_utils.kaiming_normal('leaky_relu', 0.25, self.sentinal_embed1)
        elif opt.sentinel_nonlinear == 'lecun_tanh':
            self.sentinel_nonlinear = lambda x: 1.7159 * F.tanh((2.0 / 3.0) * x)
            model_utils.xavier_normal('tanh', self.sentinal_embed1)
        elif opt.sentinel_nonlinear == 'maxout':
            del self.sentinal_embed1
            self.sentinal_embed1 = nn.Linear(opt.rnn_size, 2 * opt.rnn_size, bias=False)
            self.sentinel_nonlinear = lambda x: torch.max(x.narrow(1, 0, self.rnn_size),
                                           x.narrow(1, self.rnn_size, self.rnn_size))
            model_utils.xavier_normal('linear', self.sentinal_embed1)
        elif opt.sentinel_nonlinear == 'tanh':
            self.sentinel_nonlinear = nn.Tanh()
            model_utils.xavier_normal('tanh', self.sentinal_embed1)
        elif opt.sentinel_nonlinear == 'x':
            del self.sentinal_embed1
            self.sentinal_embed1 = lambda x: x
            self.sentinel_nonlinear = lambda x: x

        if self.add_2_layer_hidden:
            self.h1_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
            self.beta = ((opt.rnn_size + opt.rnn_size)/(2*opt.rnn_size + float(opt.rnn_size)))**0.5
            model_utils.xavier_normal(opt.nonlinear, self.h1_affine)
            self.h1_affine.weight.data = self.h1_affine.weight * self.beta
            self.h2_affine.weight.data = self.h2_affine.weight * self.beta

        if not self.project_hidden:
            del self.h2_affine
            self.h2_affine = lambda x: x
            del self.drop
            self.drop = nn.Dropout(0)
            if self.add_2_layer_hidden:
                del self.h1_affine
                self.h1_affine = lambda x: x

        if self.attention_gate:
            self.att_gate_1 = nn.Linear(opt.rnn_size + inputsize,
                                             inputsize)
            self.att_gate_2 = nn.Linear(3 * opt.rnn_size, 2 * opt.rnn_size)

            init.orthogonal_(self.att_gate_1.weight)
            init.orthogonal_(self.att_gate_2.weight)
            init.constant_(self.att_gate_1.bias, 1.0)
            init.constant_(self.att_gate_2.bias, 1.0)

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)

        if self.language_attention and self.LSTMN:
            self.Intergrate2vector = Intergrate2vector(opt, opt.Intergrate2vector)
            self.Intergrate_instead = opt.Intergrate_instead
            self.lang_att_before_pro = opt.lang_att_before_pro

        # dynamically nonlocal
        self.nonlocal_dy = opt.nonlocal_dy
        self.nonlocal_residual = opt.nonlocal_residual
        if self.nonlocal_dy:
            self.dy_nonlocal_att = NonLocalBlock(opt, self.rnn_size, self.rnn_size, self.nonlocal_residual)
            self.dy_nonlocal_lang = NonLocalBlock(opt, self.rnn_size, self.rnn_size, self.nonlocal_residual)

        if self.input_feature == 'attention':
            self.attention0 = Attention(opt)

        if self.output_attention:
            self.attention2 = Attention(opt)
            self.outatt_affine = nn.Linear(opt.rnn_size, opt.rnn_size, bias=False)
            model_utils.xavier_normal('linear', self.outatt_affine)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, p0_att_feats, p2_att_feats, state, att_masks=None, std_feat=None, mean_feat=None):
        pre_states = state[1:]
        step = len(pre_states)
        if self.LSTMN and step > 0:
            xi = torch.cat([_[4].unsqueeze(1) for _ in pre_states], 1)  # [batch_size, step, rnn_size]

        # --start-------first LSTM--------#
        if self.LSTMN:
            h1 = state[-1][0]
            h2 = state[-1][2]
            c2 = state[-1][3]
        else:
            h1 = state[0][0]
            h2 = state[0][1]
            c2 = state[1][1]

        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        weight0 = att_masks.view(-1, att_size).float()
        weight0 = weight0 / weight0.sum(1, keepdim=True)  # normalize to 1
        if self.input_feature == 'attention':
            fc_feats, weight0 = self.attention0(h1, att_feats, p0_att_feats, att_masks)  # batch_size * rnn_size

        if self.noh2pre:
            att_lstm_input = torch.cat([fc_feats, xt], 1)  # [batch_size, rnn_size + input_encoding_size]
        else:
            if self.pre == 'h':
                pre = h2
            elif self.pre == 'c':
                pre = c2
            prev_h = F.dropout(pre, self.drop_prob_rnn1, self.training)  # [batch_size, rnn_size]
            att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        if self.LSTMN:
            if step < 2 or not self.LSTMN_att:
                h1 = state[-1][0]
                c1 = state[-1][1]
                layer1_weights = h1.new_zeros((h1.size(0), 1))
            else:
                last_att_h1 = state[0][0]
                hi = torch.cat([_[0].unsqueeze(1) for _ in pre_states], 1)  # [batch_size, step, rnn_size]
                ci = torch.cat([_[1].unsqueeze(1) for _ in pre_states], 1)  # [batch_size, step, rnn_size]
                if self.input_first_att == 1:
                    input_first_att = xt
                elif self.input_first_att == 2:
                    input_first_att = torch.cat([h2, xt], 1)
                elif self.input_first_att == 3:
                    input_first_att = torch.cat([h2, fc_feats, xt], 1)
                h1, c1, layer1_weights = self.intra_att_att_lstm(input_first_att, last_att_h1, hi, ci, xi)   # batch * rnn_size, batch * num_hidden
            # ---non-local--- #
            if self.nonlocal_dy and step > 0:
                h1i = torch.cat([_[0].unsqueeze(1) for _ in pre_states], 1)     # [batch_size, step, rnn_size]
                h1, layer1_weights = self.dy_nonlocal_att(h1i)     # batch * rnn_size, batch * (step+1)
            # ---non-local--- #
        else:
            h1 = state[0][0]
            c1 = state[1][0]

        if self.attention_gate:
            att_lstm_input = torch.mul(att_lstm_input, F.sigmoid(
                self.att_gate_1(torch.cat((att_lstm_input, h1), 1))))

        h_att = self.att_lstm(att_lstm_input, h1)  # both are [batch_size, rnn_size]
        # --end-------first LSTM-------- #

        # --start-------second LSTM-------- #
        att, weight = self.attention(h_att, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size

        droped_h_att = F.dropout(h_att, self.drop_prob_rnn1, self.training)
        lang_lstm_input = torch.cat([att, droped_h_att], 1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        if self.LSTMN:
            if step < 2 or not self.LSTMN_lang:
                h2 = state[-1][2]
                c2 = state[-1][3]
                layer2_weights = h2.new_zeros((h2.size(0), 1))
            else:
                last_att_h2 = state[0][1]
                hi = torch.cat([_[2].unsqueeze(1) for _ in pre_states], 1)  # [batch_size, step, rnn_size]
                ci = torch.cat([_[3].unsqueeze(1) for _ in pre_states], 1)  # [batch_size, step, rnn_size]
                if self.input_second_att == 1:
                    input_second_att = h_att
                elif self.input_second_att == 2:
                    input_second_att = torch.cat([att, h_att], 1)
                h2, c2, layer2_weights = self.intra_att_lang_lstm(input_second_att, last_att_h2, hi,
                                                                 ci, xi)  # batch * rnn_size, batch * num_hidden
        # ---non-local--- #
        if self.nonlocal_dy and step > 0:
            h2i = torch.cat([_[2].unsqueeze(1) for _ in pre_states], 1)  # [batch_size, step, rnn_size]
            h2, layer2_weights = self.dy_nonlocal_lang(h2i)  # batch * rnn_size, batch * (step+1)
        # ---non-local--- #

        if self.attention_gate:
            lang_lstm_input = torch.mul(lang_lstm_input, F.sigmoid(self.att_gate_2(torch.cat((lang_lstm_input, h2), 1))))

        h_lang = self.lang_lstm(droped_h_att, h2)  # batch*rnn_size
        # --end------second LSTM--------#

        # --start-------generate output--------#
        if self.add_2_layer_hidden:
            affined_linear = self.h2_affine(self.drop(h_lang)) + self.h1_affine(self.drop(h_att))
        elif self.directly_add_2_layer:
            affined_linear = self.h2_affine(self.drop((h_lang + h_att) / 1.4142))
        else:
            affined_linear = self.h2_affine(self.drop(h_lang))
        if self.LSTMN:
            lang_weights = (layer1_weights + layer2_weights) / 2.0
        else:
            lang_weights = torch.zeros_like(h2)

        if self.output_attention:
            att_output, weight2 = self.attention2(h_lang, att_feats, p2_att_feats, att_masks)  # batch_size * rnn_size
            affined_linear += self.outatt_affine(self.drop(att_output))

        affined = self.tgh(affined_linear)

        output = F.dropout(affined, self.drop_prob_output, self.training)  # batch_size * rnn_size
        # --end-------generate output--------#

        # --start-------generate state--------#
        if self.LSTMN:
            if self.language_attention:
                if self.Intergrate_instead:
                    state = (torch.stack([h1, h2, h1, h2, h1]),) + tuple(pre_states) + (
                    torch.stack([h_att, h_att, h_lang, h_lang, affined_linear]),)
                else:
                    state = (torch.stack([h1, h2, h1, h2, h1]),) + tuple(pre_states) + (
                        torch.stack([h_att, h_att, h_lang, h_lang, affined_linear]),)
            else:
                state = (torch.stack([h1, h2, h1, h2, h1]),) + tuple(pre_states) + (
                torch.stack([h_att, h_att, h_lang, h_lang, affined]),)
        else:
            if self.transfer_horizontal:
                state = (torch.stack([h_att, affined]), torch.stack([h_att, h_lang]))
            else:
                state = (torch.stack([h_att, h_lang]), torch.stack([h_att, h_lang]))

            if self.language_attention:
                sentinal_current = self.sentinal_embed2(self.sentinal_embed1(h_lang))  # batch* rnn_size
                sentinal_current = self.sentinel_nonlinear(sentinal_current)  # batch* rnn_size
                state = state + tuple(pre_states) + (torch.stack([sentinal_current, torch.zeros_like(sentinal_current)]),)
        # --end-------generate state--------#

        # visualization
        if step == 6:
            self.iteration += 1
            if self.iteration > 10 and self.iteration % 100 == 0:
                batch_size = att_feats.size(0)
                if self.input_feature == 'attention':
                    print('The attention distribution of the input attention:')
                    random.seed(123)
                    for i in random.sample(range(batch_size), 3):
                        print(weight0[i][att_masks[i] != 0])

                # print('The attention distribution of the first LSTM attention:')
                # random.seed(123)
                # for i in random.sample(range(batch_size), 3):
                #     print(weight[i][att_masks[i] != 0])

                if self.output_attention:
                    print('The attention distribution of the output attention:')
                    random.seed(123)
                    for i in random.sample(range(batch_size), 3):
                        print(weight2[i][att_masks[i] != 0])

        return output, state, lang_weights

    def average_hiddens(self, h_lang, hiddens):
        weighted_sentinal = torch.mean(hiddens, 1)  # batch_size * rnn_size
        return weighted_sentinal


############################################################################
# Notice:
# StackAtt and DenseAtt are models that I randomly designed.
# They are not related to any paper.
############################################################################

from .FCModel import LSTMCore


class StackAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(StackAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        # self.att0 = Attention(opt)
        self.att1 = Attention(opt)
        self.att2 = Attention(opt)

        opt_input_encoding_size = opt.input_encoding_size
        opt.input_encoding_size = opt.input_encoding_size + opt.rnn_size
        self.lstm0 = LSTMCore(opt)  # att_feat + word_embedding
        opt.input_encoding_size = opt.rnn_size * 2
        self.lstm1 = LSTMCore(opt)
        self.lstm2 = LSTMCore(opt)
        opt.input_encoding_size = opt_input_encoding_size

        # self.emb1 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.emb2 = nn.Linear(opt.rnn_size, opt.rnn_size)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        # att_res_0 = self.att0(state[0][-1], att_feats, p_att_feats, att_masks)
        h_0, state_0 = self.lstm0(torch.cat([xt, fc_feats], 1), [state[0][0:1], state[1][0:1]])
        att_res_1 = self.att1(h_0, att_feats, p_att_feats, att_masks)
        h_1, state_1 = self.lstm1(torch.cat([h_0, att_res_1], 1), [state[0][1:2], state[1][1:2]])
        att_res_2 = self.att2(h_1 + self.emb2(att_res_1), att_feats, p_att_feats, att_masks)
        h_2, state_2 = self.lstm2(torch.cat([h_1, att_res_2], 1), [state[0][2:3], state[1][2:3]])

        return h_2, [torch.cat(_, 0) for _ in zip(state_0, state_1, state_2)]


class DenseAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(DenseAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        # self.att0 = Attention(opt)
        self.att1 = Attention(opt)
        self.att2 = Attention(opt)

        opt_input_encoding_size = opt.input_encoding_size
        opt.input_encoding_size = opt.input_encoding_size + opt.rnn_size
        self.lstm0 = LSTMCore(opt)  # att_feat + word_embedding
        opt.input_encoding_size = opt.rnn_size * 2
        self.lstm1 = LSTMCore(opt)
        self.lstm2 = LSTMCore(opt)
        opt.input_encoding_size = opt_input_encoding_size

        # self.emb1 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.emb2 = nn.Linear(opt.rnn_size, opt.rnn_size)

        # fuse h_0 and h_1
        self.fusion1 = nn.Sequential(nn.Linear(opt.rnn_size * 2, opt.rnn_size),
                                     nn.ReLU(),
                                     nn.Dropout(opt.drop_prob_lm))
        # fuse h_0, h_1 and h_2
        self.fusion2 = nn.Sequential(nn.Linear(opt.rnn_size * 3, opt.rnn_size),
                                     nn.ReLU(),
                                     nn.Dropout(opt.drop_prob_lm))

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        # att_res_0 = self.att0(state[0][-1], att_feats, p_att_feats, att_masks)
        h_0, state_0 = self.lstm0(torch.cat([xt, fc_feats], 1), [state[0][0:1], state[1][0:1]])
        att_res_1 = self.att1(h_0, att_feats, p_att_feats, att_masks)
        h_1, state_1 = self.lstm1(torch.cat([h_0, att_res_1], 1), [state[0][1:2], state[1][1:2]])
        att_res_2 = self.att2(h_1 + self.emb2(att_res_1), att_feats, p_att_feats, att_masks)
        h_2, state_2 = self.lstm2(torch.cat([self.fusion1(torch.cat([h_0, h_1], 1)), att_res_2], 1),
                                  [state[0][2:3], state[1][2:3]])

        return self.fusion2(torch.cat([h_0, h_1, h_2], 1)), [torch.cat(_, 0) for _ in zip(state_0, state_1, state_2)]


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Sequential(*((nn.Linear(self.rnn_size, self.att_hid_size),) + (
            (nn.BatchNorm1d(self.att_hid_size),) if opt.BN_other else ())))
        self.alpha_net = nn.Linear(self.att_hid_size, 1, bias=False)

        # initialization
        self.beta = ((self.att_hid_size + self.rnn_size) / (float(self.att_hid_size) + 2 * self.rnn_size)) ** 0.5
        model_utils.xavier_normal('tanh', self.h2att[0])
        self.h2att[0].weight.data = self.h2att[0].weight*self.beta
        model_utils.kaiming_normal('relu', 0, self.alpha_net)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)  # 36
        att = p_att_feats.view(-1, att_size, self.att_hid_size)  # [batch_size, 36, att_hid_size]

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = F.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot, dim=1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))  # batch * att_size * rnn_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * rnn_size

        return att_res, weight


class Intergrate2vector(nn.Module):
    def __init__(self, opt, inter_method):
        super(Intergrate2vector, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.inter_method = inter_method
        if self.inter_method == 'concat':
            self.v1_pro = nn.Sequential(*((nn.Linear(self.rnn_size, self.att_hid_size, bias=False),) + (
                (nn.BatchNorm1d(self.att_hid_size),) if opt.BN_other else ())))
            self.v2_pro = nn.Sequential(*((nn.Linear(self.rnn_size, self.att_hid_size, bias=False),) + (
                (nn.BatchNorm1d(self.att_hid_size),) if opt.BN_other else ())))
            if opt.Intergrate_ele_wise:
                self.alpha_net = nn.Linear(self.att_hid_size, self.rnn_size, bias=True)
            else:
                self.alpha_net = nn.Linear(self.att_hid_size, 1, bias=False)
            # initialization
            self.beta1 = ((self.rnn_size + self.att_hid_size) / (2 * self.rnn_size + float(self.att_hid_size))) ** 0.5
            model_utils.xavier_normal('tanh', self.v1_pro[0], self.v2_pro[0])
            self.v1_pro[0].weight.data = self.v1_pro[0].weight*self.beta1
            self.v2_pro[0].weight.data = self.v2_pro[0].weight * self.beta1
            model_utils.xavier_normal('sigmoid', self.alpha_net)
        elif self.inter_method == 'self_attention':
            self.v_pro = nn.Linear(self.rnn_size, self.att_hid_size)
            self.alpha_1 = nn.Linear(self.att_hid_size, 1, bias=False)
            self.alpha_2 = nn.Linear(self.att_hid_size, 1, bias=False)
            # initialization
            model_utils.xavier_normal('tanh', self.v_pro)
            model_utils.kaiming_normal('relu', 0, self.alpha_1, self.alpha_2)
        elif self.inter_method == 'self_attention_tying_alpha':
            self.v_pro = nn.Linear(self.rnn_size, self.att_hid_size)
            self.alpha_1 = self.alpha_2 = nn.Linear(self.att_hid_size, 1, bias=False)
            # initialization
            model_utils.xavier_normal('tanh', self.v_pro)
            model_utils.kaiming_normal('relu', 0, self.alpha_1, self.alpha_2)

    def forward(self, v1, v2):
        # size of both is batch*rnn_size

        if self.inter_method == 'concat':
            prob_v1 = self.concat_prob(v1, v2)              # batch * 1 or batch * rnn_size
        elif 'self_attention' in self.inter_method:
            prob_v1 = self.self_attention_prob(v1, v2)      # batch * 1

        inter_v = prob_v1 * v1 + (1.0 - prob_v1) * v2

        prob_v1 = torch.mean(prob_v1, 1, True)              # batch * 1

        return inter_v, prob_v1     # batch*rnn_size, batch*1 or batch * rnn_size

    def self_attention_prob(self, v1, v2):
        v1_score = self.alpha_1(F.tanh(self.v_pro(v1)))  # batch * 1
        v2_score = self.alpha_2(F.tanh(self.v_pro(v2)))  # batch * 1
        score = torch.cat((v1_score, v2_score), 1)  # batch * 2
        weights = F.softmax(score, dim=1)       # batch * 2

        return weights[:, 0].unsqueeze(1)       # batch * 1

    def concat_prob(self, v1, v2):
        # size of both is batch * rnn_size
        p_v1 = self.v1_pro(v1)  # batch * att_hid_size
        p_v2 = self.v2_pro(v2)  # batch * att_hid_size
        score = self.alpha_net(F.tanh(p_v1 + p_v2))  # batch * 1 or batch * rnn_size
        prob = F.sigmoid(score)     # batch * 1 or batch * rnn_size

        return prob                 # batch * 1 or batch * rnn_size


class SentinalAttention(nn.Module):
    def __init__(self, opt):
        super(SentinalAttention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.lang_no_h0 = opt.lang_no_h0

        self.att_score_method = opt.att_score_method
        if self.att_score_method == 'mlp':
            self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
            self.alpha_net = nn.Linear(self.att_hid_size, 1, bias=False)
            self.senti2att = nn.Linear(self.rnn_size, self.att_hid_size)
            # initialization
            self.beta1 = ((self.rnn_size + self.att_hid_size)/(2*self.rnn_size + float(self.att_hid_size)))**0.5
            model_utils.xavier_normal('tanh', self.h2att, self.senti2att)
            self.h2att.weight.data = self.h2att.weight*self.beta1
            self.senti2att.weight.data = self.senti2att.weight * self.beta1
            model_utils.kaiming_normal('relu', 0, self.alpha_net)
        elif self.att_score_method == 'general':
            self.general_att = nn.Linear(self.rnn_size, self.rnn_size, bias=False)
            model_utils.kaiming_normal('relu', 0, self.general_att)
        elif self.att_score_method == 'cosine':
            self.cos = nn.CosineSimilarity(dim=2)
        elif self.att_score_method == 'scaled_dot':
            self.scale = nn.Parameter(torch.Tensor([22.6]))

    def forward(self, h, sentinal):
        if self.lang_no_h0 and sentinal.size(1) > 1:
            sentinal = sentinal[:, 1:, :]

        if self.att_score_method == 'mlp':
            dot_senti = self.mlp_att_score(h, sentinal)
        elif self.att_score_method == 'dot':
            dot_senti = self.dot_att_score(h, sentinal)
        elif self.att_score_method == 'general':
            dot_senti = self.general_att_score(h, sentinal)
        elif self.att_score_method == 'cosine':
            dot_senti = self.cosine_att_score(h, sentinal)
        elif self.att_score_method == 'scaled_dot':
            dot_senti = self.scaled_dot_att_score(h, sentinal)

        weight_senti = F.softmax(dot_senti, dim=1)  # batch * num_hidden
        weight_senti = weight_senti.unsqueeze(1)  # batch * 1 * num_hidden
        att_res_senti = torch.bmm(weight_senti, sentinal).squeeze(1)  # batch * rnn_size
        # --end----recurrent attention-----#

        return att_res_senti, weight_senti.squeeze(1)  # batch * rnn_size, batch * num_hidden

    def mlp_att_score(self, h, sentinal):
        # sentinal's shape is batch * num_hidden * rnn_size
        num_hidden = sentinal.size(1)
        # --start----recurrent attention (including sentinal attention, recurrent hidden attention, recurrent sentinal attention)-----#
        att_h = self.h2att(h)  # batch * att_hid_size
        att_senti = self.senti2att(sentinal)  # batch * num_hidden * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att_senti)  # batch * num_hidden * att_hid_size
        dot_senti = att_senti + att_h  # batch * num_hidden * att_hid_size
        dot_senti = F.tanh(dot_senti)  # batch * num_hidden * att_hid_size
        dot_senti = dot_senti.view(-1, self.att_hid_size)  # (batch*num_hidden) * att_hid_size
        dot_senti = self.alpha_net(dot_senti)  # (batch*num_hidden) * 1
        dot_senti = dot_senti.view(-1, num_hidden)  # batch * num_hidden
        return dot_senti

    def dot_att_score(self, h, sentinal):
        # sentinal's shape is batch * num_hidden * rnn_size
        h = h.unsqueeze(1)      # batch*1*rnn_size
        sentinal = torch.transpose(sentinal, 1, 2)      # batch*rnn_size*num_hidden

        return torch.bmm(h, sentinal).squeeze(1)       # batch * num_hidden

    def general_att_score(self, h, sentinal):
        # sentinal's shape is batch * num_hidden * rnn_size
        h = h.unsqueeze(1)      # batch*1*rnn_size
        h = self.general_att(h)       # batch*1*rnn_size
        sentinal = torch.transpose(sentinal, 1, 2)      # batch*rnn_size*num_hidden

        return torch.bmm(h, sentinal).squeeze(1)/(self.rnn_size**0.5)       # batch * num_hidden

    def cosine_att_score(self, h, sentinal):
        # sentinal's shape is batch * num_hidden * rnn_size
        h = h.unsqueeze(1).expand_as(sentinal)      # batch * num_hidden * rnn_size

        return self.cos(h, sentinal)       # batch * num_hidden

    def scaled_dot_att_score(self, h, sentinal):
        # sentinal's shape is batch * num_hidden * rnn_size
        h = h.unsqueeze(1)      # batch*1*rnn_size
        sentinal = torch.transpose(sentinal, 1, 2)      # batch*rnn_size*num_hidden

        return torch.bmm(h, sentinal).squeeze(1)/self.scale      # batch * num_hidden


class IntraAttention(nn.Module):
    def __init__(self, opt, xt_dimension, att_score_method):
        super(IntraAttention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.xt_dimension = xt_dimension
        self.LSTMN_last_att_hidden = opt.LSTMN_last_att_hidden
        self.LSTM_att_hi = opt.LSTM_att_hi
        self.LSTMN_att_key_hi_ci = opt.LSTMN_att_key_hi_ci
        self.distance_sensitive_bias = opt.distance_sensitive_bias
        self.distance_bias_9_1 = opt.distance_bias_9_1
        self.distance_sensitive_coefficient = opt.distance_sensitive_coefficient
        self.word_sensitive_bias = opt.word_sensitive_bias
        self.word_sensitive_coefficient = opt.word_sensitive_coefficient
        self.word_sensitive_hi_xi = opt.word_sensitive_hi_xi
        self.not_use_first = opt.not_use_first
        self.position_sentive_inter = opt.position_sentive_inter
        self.att_score_method = att_score_method

        self.beta1 = ((self.rnn_size + self.att_hid_size) / (
                    float(self.att_hid_size) + self.xt_dimension + 2 * self.rnn_size)) ** 0.5
        self.beta2 = ((self.rnn_size + self.att_hid_size) / (
                float(self.att_hid_size) + self.xt_dimension + self.rnn_size)) ** 0.5
        self.beta3 = ((self.xt_dimension + self.att_hid_size) / (
                float(self.att_hid_size) + self.xt_dimension + 2 * self.rnn_size)) ** 0.5
        self.beta4 = ((self.xt_dimension + self.att_hid_size) / (
                float(self.att_hid_size) + self.xt_dimension + self.rnn_size)) ** 0.5

        self.xt2att = nn.Linear(self.xt_dimension, self.att_hid_size)
        self.hl2att = nn.Linear(self.rnn_size, self.att_hid_size, bias=False)

        if self.position_sentive_inter:
            self.hi2atts = nn.ModuleList([nn.Linear(self.rnn_size, self.att_hid_size, bias=True) for _ in range(16)])
            for i in range(len(self.hi2atts)):
                if i == 0:
                    nonlinear = 'linear'
                else:
                    nonlinear = 'tanh'
                model_utils.xavier_normal(nonlinear, self.hi2atts[i])
                if self.LSTMN_last_att_hidden:
                    self.hi2atts[i].weight.data = self.hi2atts[i].weight * self.beta1
                else:
                    self.hi2atts[i].weight.data = self.hi2atts[i].weight * self.beta2
        else:
            self.hi2att = nn.Linear(self.rnn_size, self.att_hid_size, bias=False)
            model_utils.xavier_normal('tanh', self.hi2att)
            if self.LSTMN_last_att_hidden:
                self.hi2att.weight.data = self.hi2att.weight * self.beta1
            else:
                self.hi2att.weight.data = self.hi2att.weight * self.beta2

        self.alpha_net = nn.Linear(self.att_hid_size, 1, bias=False)
        # initialization
        model_utils.xavier_normal('tanh', self.xt2att, self.hl2att)
        model_utils.kaiming_normal('relu', 0, self.alpha_net)
        self.hl2att.weight.data = self.hl2att.weight * self.beta1
        if self.LSTMN_last_att_hidden:
            self.xt2att.weight.data = self.xt2att.weight * self.beta3
        else:
            self.xt2att.weight.data = self.xt2att.weight * self.beta4

        if not self.LSTMN_last_att_hidden:
            del self.hl2att

        if self.word_sensitive_bias:
            self.word_bias_projection = nn.Sequential(nn.Linear(self.rnn_size, self.rnn_size, bias=False), nn.Tanh(),
                                                      nn.Linear(self.rnn_size, 1, bias=False))
            model_utils.xavier_normal('tanh', self.word_bias_projection[0], self.word_bias_projection[2])

        if self.word_sensitive_coefficient:
            self.word_coefficient_projection = nn.Sequential(nn.Linear(self.rnn_size, self.rnn_size, bias=False),
                                                             nn.Tanh(),
                                                             nn.Linear(self.rnn_size, 1, bias=False))
            model_utils.xavier_normal('tanh', self.word_coefficient_projection[0], self.word_coefficient_projection[2])

        if self.distance_sensitive_bias:
            if self.distance_bias_9_1:
                self.bias_1_9 = nn.Parameter(self.alpha_net.weight.data.new_zeros((1, 9)))
                self.bias_10_ = nn.Parameter(self.alpha_net.weight.data.new_zeros((1, 1)))
            else:
                self.bias = nn.Parameter(self.alpha_net.weight.data.new_zeros((1, 16)))

        if self.distance_sensitive_coefficient:
            self.coefficient = nn.Parameter(self.alpha_net.weight.data.new_ones((1, 16)))

        if self.att_score_method == 'general':
            self.general_att = nn.Linear(self.rnn_size, self.rnn_size, bias=False)
            model_utils.kaiming_normal('relu', 0, self.general_att)
        elif self.att_score_method == 'scaled_dot':
            self.scale = nn.Parameter(torch.Tensor([float(self.rnn_size)**0.5]))

    def forward(self, xt, last_att_hl, hi, ci, xi):
        # xt, batch * input_encoding_size
        # last_att_hl, batch * rnn_size
        # hi, batch * num_hidden * rnn_size
        # ci, batch * num_hidden * rnn_size
        # xi, batch * num_hidden * input_encoding_size

        if self.not_use_first:
            hi = hi[:, 1:, :]
            ci = ci[:, 1:, :]
            xi = xi[:, 1:, :]

        num_hidden = ci.size(1)

        # relevance score
        if self.LSTMN_att_key_hi_ci == 0:
            key = hi
        else:
            key = ci

        if self.att_score_method == 'mlp':
            att_score = self.mlp_att_score(xt, last_att_hl, key)  # batch * num_hidden
        elif self.att_score_method == 'general':
            att_score = self.general_att_score(xt, key)
        elif self.att_score_method == 'scaled_dot':
            att_score = self.scaled_dot_att_score(xt, key)

        if self.word_sensitive_hi_xi == 1:
            key = xi
        else:
            key = hi

        # add word-sensitive coefficient
        if self.word_sensitive_coefficient:
            att_score = att_score * self.word_coefficient_projection(key).squeeze(2)

        # add distance-sensitive coefficient
        if self.distance_sensitive_coefficient:
            att_score = att_score * self.coefficient[:, -num_hidden:].expand_as(att_score)

        # add word-sensitive bias
        if self.word_sensitive_bias:
            att_score = att_score + self.word_bias_projection(key).squeeze(2)

        # add distance-sensitive bias
        if self.distance_sensitive_bias:
            if self.distance_bias_9_1:
                if num_hidden < 10:
                    self.bias = self.bias_1_9.data
                else:
                    self.bias = torch.cat((self.bias_10_.expand(-1, 7), self.bias_1_9), 1)
            att_score = att_score + self.bias[:, -num_hidden:].expand_as(att_score)

        # softmax
        weights = F.softmax(att_score, dim=1)  # batch * num_hidden
        weights = weights.unsqueeze(1)  # batch * 1 * num_hidden

        if self.LSTM_att_hi:
            # weighted summation
            att_hi = torch.bmm(weights, hi).squeeze(1)  # batch * rnn_size
        else:
            att_hi = hi[:, -1, :]

        att_ci = torch.bmm(weights, ci).squeeze(1)  # batch * rnn_size
        # --end----recurrent attention-----#

        return att_hi, att_ci, weights.squeeze(1)  # batch * rnn_size, batch * num_hidden

    def mlp_att_score(self, xt, last_att_hl, hi):
        # hi's shape is batch * num_hidden * rnn_size
        num_hidden = hi.size(1)

        att_xt = self.xt2att(xt)  # batch * att_hid_size

        if self.position_sentive_inter:
            att_hi = []
            for i in range(num_hidden):
                att_hi.append(self.hi2atts[i](hi[:, i, :]).unsqueeze(1))      # batch * att_hid_size
            att_hi = torch.cat(att_hi, 1)  # batch * num_hidden * att_hid_size
        else:
            att_hi = self.hi2att(hi)  # batch * num_hidden * att_hid_size


        att_xt = att_xt.unsqueeze(1).expand_as(att_hi)  # batch * num_hidden * att_hid_size

        if self.LSTMN_last_att_hidden:
            att_hl = self.hl2att(last_att_hl)  # batch * att_hid_size
            att_hl = att_hl.unsqueeze(1).expand_as(att_hi)  # batch * num_hidden * att_hid_size
            att_score = att_xt + att_hl + att_hi  # batch * num_hidden * att_hid_size
        else:
            att_score = att_xt + att_hi  # batch * num_hidden * att_hid_size

        att_score = F.tanh(att_score)  # batch * num_hidden * att_hid_size
        att_score = att_score.view(-1, self.att_hid_size)  # (batch*num_hidden) * att_hid_size
        att_score = self.alpha_net(att_score)  # (batch*num_hidden) * 1
        att_score = att_score.view(-1, num_hidden)  # batch * num_hidden

        return att_score

    def general_att_score(self, h, sentinal):
        # sentinal's shape is batch * num_hidden * rnn_size
        h = h.unsqueeze(1)      # batch*1*rnn_size
        h = self.general_att(h)       # batch*1*rnn_size
        sentinal = torch.transpose(sentinal, 1, 2)      # batch*rnn_size*num_hidden

        return torch.bmm(h, sentinal).squeeze(1)/(float(self.rnn_size)**0.5)       # batch * num_hidden

    def scaled_dot_att_score(self, h, sentinal):
        # sentinal's shape is batch * num_hidden * rnn_size
        h = h.unsqueeze(1)      # batch*1*rnn_size
        sentinal = torch.transpose(sentinal, 1, 2)      # batch*rnn_size*num_hidden

        return torch.bmm(h, sentinal).squeeze(1)/self.scale      # batch * num_hidden


class NonLocalBlock(nn.Module):
    def __init__(self, opt, in_size, inter_size=None, residual=True):
        super(NonLocalBlock, self).__init__()

        self.in_size = in_size
        self.inter_size = inter_size
        self.nonlocal_dy_bn = opt.nonlocal_dy_bn
        self.residual = residual
        self.project_g = opt.project_g

        if self.inter_size is None:
            self.inter_size = in_size // 2

        if self.project_g:
            self.g = nn.Linear(self.in_size, self.inter_size)
        else:
            self.g = lambda x: x
        self.theta = nn.Linear(self.in_size, self.inter_size)
        self.phi = nn.Linear(self.in_size, self.inter_size)

        if self.nonlocal_dy_bn:
            self.W = nn.Sequential(
                nn.Linear(self.inter_size, self.in_size),
                nn.BatchNorm1d(self.in_size)
            )
            model_utils.xavier_normal('linear', self.W[0])
            if opt.nonlocal_dy_insert:
                nn.init.constant_(self.W[1].weight, 0)
                nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Linear(self.inter_size, self.in_size)
            model_utils.xavier_normal('linear', self.W)
            if opt.nonlocal_dy_insert:
                nn.init.constant_(self.W.weight, 0)
                nn.init.constant_(self.W.bias, 0)

        # initialization
        model_utils.kaiming_normal('relu', 0, self.theta, self.phi)
        if self.project_g:
            model_utils.xavier_normal('linear', self.g)

        # normalization
        self.nonlocal_normalize_method = opt.nonlocal_normalize_method

    def forward(self, x):
        '''
        :param x: (batch, (step+1), rnn_size)
        :return:
        '''

        g_x = self.g(x)                                   # batch*(step+1)*inter_size

        phi_x = self.phi(x[:, -1, :]).unsqueeze(1)  # batch*1*inter_size

        theta_x = self.theta(x)                           # batch*(step+1)*inter_size
        theta_x = torch.transpose(theta_x, 1, 2)          # batch*inter_size*(step+1)

        f = torch.bmm(phi_x, theta_x).squeeze(1) / (self.inter_size ** 0.5)     # batch*(step+1)

        f_div_C = F.softmax(f, dim=1)   # batch * (step+1)
        z = torch.bmm(f_div_C.unsqueeze(1), g_x).squeeze(1).contiguous()  # batch * inter_size

        if self.nonlocal_normalize_method == '1':
            l2_weight = f_div_C.norm(p=2, dim=1, keepdim=True)  # batch * 1
            z = z / l2_weight

        if self.residual:
            W_y = self.W(z)             # batch * rnn_size
            z = W_y + x[:, -1, :]

        return z, f_div_C.squeeze(1)  # batch * rnn_size, batch * (step+1)


class Att2in2Core(nn.Module):
    def __init__(self, opt):
        super(Att2in2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        # self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        # self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        # Build a LSTM
        self.a2c = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + \
                       self.a2c(att_res)
        in_transform = torch.max( \
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state


class Att2inCore(Att2in2Core):
    def __init__(self, opt):
        super(Att2inCore, self).__init__(opt)
        del self.a2c
        self.a2c = nn.Linear(self.att_feat_size, 2 * self.rnn_size)


"""
Note this is my attempt to replicate att2all model in self-critical paper.
However, this is not a correct replication actually. Will fix it.
"""


class Att2all2Core(nn.Module):
    def __init__(self, opt):
        super(Att2all2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        # self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        # self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        # Build a LSTM
        self.a2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1]) + self.a2h(att_res)
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
        in_transform = torch.max( \
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state


class AdaAttModel(AttModel):
    def __init__(self, opt):
        super(AdaAttModel, self).__init__(opt)
        self.core = AdaAttCore(opt)


# AdaAtt with maxout lstm
class AdaAttMOModel(AttModel):
    def __init__(self, opt):
        super(AdaAttMOModel, self).__init__(opt)
        self.core = AdaAttCore(opt, True)


class Att2in2Model(AttModel):
    def __init__(self, opt):
        super(Att2in2Model, self).__init__(opt)
        self.core = Att2in2Core(opt)
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x: x


class Att2all2Model(AttModel):
    def __init__(self, opt):
        super(Att2all2Model, self).__init__(opt)
        self.core = Att2all2Core(opt)
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x: x


class TopDownModel(AttModel):
    def __init__(self, opt):
        super(TopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownCore(opt)


class TopDownOriginalModel(AttModel):
    def __init__(self, opt):
        super(TopDownOriginalModel, self).__init__(opt)
        del self.fc_embed, self.att_embed
        self.fc_embed = nn.BatchNorm1d(self.fc_feat_size)
        self.att_embed = nn.BatchNorm1d(self.att_feat_size)
        del self.ctx2att
        self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)

        self.num_layers = 2
        self.core = TopDownOriginalCore(opt)

        # initialization
        model_utils.xavier_normal('tanh', self.ctx2att)


class TopDownUpCatWeightedHiddenModel_3(AttModel):
    def __init__(self, opt):
        super(TopDownUpCatWeightedHiddenModel_3, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownUpCatWeightedHiddenCore3(opt)


class TestModel(AttModel):
    def __init__(self, opt):
        super(TestModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TestCore(opt)


class BottomUpModel(AttModel):
    def __init__(self, opt):
        super(BottomUpModel, self).__init__(opt)
        self.num_layers = 2
        self.core = BottomUpCore(opt)

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, average_att_feat, state):
        # 'it' contains a word index
        xt = self.embed(it)  # [batch_size, input_encoding_size]
        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, average_att_feat,
                                  att_masks)  # batch*rn_size
        logprobs = F.log_softmax(self.logit(output), dim=1)  # batch*(vocab_size+1)

        return logprobs, state


class StackAttModel(AttModel):
    def __init__(self, opt):
        super(StackAttModel, self).__init__(opt)
        self.num_layers = 3
        self.core = StackAttCore(opt)


class DenseAttModel(AttModel):
    def __init__(self, opt):
        super(DenseAttModel, self).__init__(opt)
        self.num_layers = 3
        self.core = DenseAttCore(opt)


class Att2inModel(AttModel):
    def __init__(self, opt):
        super(Att2inModel, self).__init__(opt)
        del self.embed, self.fc_embed, self.att_embed
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.fc_embed = self.att_embed = lambda x: x
        del self.ctx2att
        self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.core = Att2inCore(opt)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)
