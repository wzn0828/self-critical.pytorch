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
import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from models import model_utils
from .CaptionModel import CaptionModel


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
        self.input_encoding_size = opt.input_encoding_size
        # self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0  # Schedule sampling probability

        self.embed = nn.Sequential(*((nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                      nn.ReLU(),) +
                                     ((nn.BatchNorm1d(self.input_encoding_size),) if opt.emb_use_bn else ()) +
                                     (nn.Dropout(self.drop_prob_lm),)
                                     ))
        self.fc_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.fc_feat_size),) if opt.fc_use_bn else ()) +
                (nn.Linear(self.fc_feat_size, self.rnn_size),
                 nn.ReLU(),) +
                ((nn.BatchNorm1d(self.rnn_size),) if opt.fc_use_bn == 2 else ()) +
                (nn.Dropout(self.drop_prob_lm),)))

        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.rnn_size),
                 nn.ReLU(),) +
                ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ()) +
                (nn.Dropout(self.drop_prob_lm),)))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in
                          range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(
                *(reduce(lambda x, y: x + y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

        # initialization
        nn.init.normal_(self.embed[0].weight, mean=0, std=0.01)
        model_utils.kaiming_normal('relu', 0, filter(lambda x: 'linear' in str(type(x)), self.fc_embed)[0],
                                   filter(lambda x: 'linear' in str(type(x)), self.att_embed)[0], self.logit)
        model_utils.xavier_uniform('tanh', self.ctx2att)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
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
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        att_masks_expand = att_masks.unsqueeze(2).expand_as(att_feats)
        att_feats_masked = att_feats * att_masks_expand
        lens = att_masks.data.long().sum(1)
        lens = lens.unsqueeze(1).expand(lens.size(0), att_feats.size(-1))
        average_att_feat = att_feats_masked.sum(1) / lens.data.float()  # batch*rnn_size

        return fc_feats, att_feats, p_att_feats, att_masks, average_att_feat

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

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, average_att_feat = self._prepare_feature(fc_feats,
                                                                                                     att_feats,
                                                                                                     att_masks)  # p_fc_feats [batch_size, rnn_size], p_att_feats [batch_size, 36, rnn_size], pp_att_feats [batch_size, 36, att_hid_size]
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

            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks,
                                                    average_att_feat, state)  # batch*(vocab_size+1)
            outputs[:, i] = output

            att_hiddens.append(state[0][0])
            lan_hiddens.append(state[0][1])
            att_sentinals.append(state[-1][0])
            lang_sentinals.append(state[-1][1])

        # return outputs, p_fc_feats, p_att_feats, torch.stack(att_hiddens), torch.stack(lan_hiddens), torch.stack(
        #     sentinals)
        return outputs, p_fc_feats, torch.stack(att_hiddens), torch.stack(lan_hiddens), torch.stack(
            att_sentinals), torch.stack(lang_sentinals)

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, average_att_feat, state):
        # 'it' contains a word index
        xt = self.embed(it)  # [batch_size, input_encoding_size]

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)  # batch*rn_size
        logprobs = F.log_softmax(self.logit(output), dim=1)  # batch*(vocab_size+1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, average_att_feat = self._prepare_feature(fc_feats,
                                                                                                     att_feats,
                                                                                                     att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k + 1].expand(beam_size, p_fc_feats.size(1))
            tmp_att_feats = p_att_feats[k:k + 1].expand(*((beam_size,) + p_att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = pp_att_feats[k:k + 1].expand(*((beam_size,) + pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k + 1].expand(
                *((beam_size,) + p_att_masks.size()[1:])).contiguous() if att_masks is not None else None
            tmp_average_att_feat = average_att_feat[k:k + 1].expand(beam_size, average_att_feat.size(1))

            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                          tmp_att_masks, tmp_average_att_feat, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                  tmp_att_masks, tmp_average_att_feat, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)  # (2*batch*rnn_size, 2*batch*rnn_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, average_att_feat = self._prepare_feature(fc_feats, att_feats,
                                                                                   att_masks)  # batch*rnn_size, batch*36*rnn_size, batch*36*att_hid_size

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)  # batch*16
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)  # batch*16
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks,average_att_feat,
                                                      state)  # batch*(vocab_size+1), (2*batch*rnn_size, 2*batch*rnn_size)

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


class TopDownOriginal2Core(TopDownOriginalCore):
    def __init__(self, opt):
        super(TopDownOriginal2Core, self).__init__(opt)
        del self.lang_lstm
        self.lang_lstm = nn.LSTMCell(2 * opt.rnn_size, opt.rnn_size)  # h^1_t, \hat v

        # initialization
        model_utils.lstm_init(self.lang_lstm)


class TopDownSentinalAffine2Core(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownSentinalAffine2Core, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = AttentionRecurrent(opt)

        # -------generate sentinal--------#
        self.i2h_2 = nn.Linear(opt.rnn_size * 2, opt.rnn_size)
        self.h2h_2 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.sentinal_embed1 = nn.Sequential(*(
                (nn.Linear(opt.rnn_size, opt.rnn_size),
                 nn.ReLU(),) +
                ((nn.BatchNorm1d(opt.rnn_size),) if opt.use_bn else ()) +
                (nn.Dropout(self.drop_prob_lm),)))
        self.sentinal_embed2 = nn.Sequential(*(
                (nn.Linear(opt.rnn_size, opt.rnn_size),
                 nn.ReLU(),) +
                ((nn.BatchNorm1d(opt.rnn_size),) if opt.use_bn else ()) +
                (nn.Dropout(self.drop_prob_lm),)))

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)
        model_utils.xavier_uniform('sigmoid', self.i2h_2, self.h2h_2)
        model_utils.kaiming_normal('relu', 0, filter(lambda x: 'linear' in str(type(x)), self.sentinal_embed1)[0],
                                   filter(lambda x: 'linear' in str(type(x)), self.sentinal_embed2)[0])

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        sentinal = state[2][0].unsqueeze(1)  # [batch_size, 1, rnn_size]

        prev_h = state[0][-1]  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        att = self.attention(h_att, att_feats, p_att_feats, sentinal,
                             att_masks)  # h, att_feats, p_att_feats, sentinal, att_masks=None

        lang_lstm_input = torch.cat([att, h_att], 1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate sentinal--------#
        ada_gate_point = F.sigmoid(self.i2h_2(lang_lstm_input) + self.h2h_2(prev_h))  # batch*rnn_size
        # sentinal = F.dropout(ada_gate_point * F.tanh(c_lang), self.drop_prob_lm, self.training)     # batch*rnn_size
        sentinal = ada_gate_point * F.tanh(c_lang)  # batch*rnn_size
        sentinal = self.sentinal_embed2(self.sentinal_embed1(sentinal))
        state = state + (torch.stack([sentinal, torch.zeros_like(sentinal)]),)
        # --end-------generate sentinal--------#

        return output, state


class TopDownRecurrentHiddenCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownRecurrentHiddenCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = AttentionRecurrent(opt)

        # #-------generate sentinal--------#
        self.sentinal_embed = nn.Sequential(nn.Linear(opt.rnn_size, opt.rnn_size),
                                            nn.ReLU(),
                                            nn.Dropout(self.drop_prob_lm))

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)
        model_utils.kaiming_normal('relu', 0, filter(lambda x: 'linear' in str(type(x)), self.sentinal_embed)[0])

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        sentinal = state[2]  # [batch_size, num_recurrent, rnn_size]

        prev_h = state[0][-1]  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        att = self.attention(h_att, att_feats, p_att_feats, sentinal,
                             att_masks)  # h, att_feats, p_att_feats, sentinal, att_masks=None

        lang_lstm_input = torch.cat([att, h_att], 1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate recurrent--------#
        # ada_gate_point = F.sigmoid(self.i2h_2(lang_lstm_input) + self.h2h_2(prev_h))      # batch*rnn_size
        # sentinal = F.dropout(ada_gate_point * F.tanh(c_lang), self.drop_prob_lm, self.training)     # batch*rnn_size
        sentinal = torch.cat([sentinal, self.sentinal_embed(h_lang.unsqueeze(1))],
                             1)  # [batch_size, num_recurrent + 1, rnn_size]
        state = state + (sentinal,)
        # --end-------generate recurrent--------#

        return output, state


class TopDownWeightedHiddenCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownWeightedHiddenCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn
        self.rnn_size = opt.rnn_size

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = AttentionRecurrent(opt)
        self.sen_attention = SentinalAttention(opt)

        # -------generate sentinal--------#
        self.sentinal_embed1 = nn.Linear(opt.rnn_size, 2 * opt.rnn_size, bias=False)
        self.sentinal_embed2 = lambda x: x

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)
        model_utils.xavier_normal('linear', self.sentinal_embed1)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        pre_sentinal = state[2:]
        sentinal = torch.cat([_[0].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        # sentinal = F.dropout(sentinal, self.drop_prob_rnn, self.training)

        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        weighted_sentinal = self.sen_attention(h_att, sentinal)  # batch_size * rnn_size
        att = self.attention(h_att, att_feats, p_att_feats, weighted_sentinal.unsqueeze(1),
                             att_masks)  # h, att_feats, p_att_feats, sentinal, att_masks=None

        lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_rnn, self.training)],
                                    1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate recurrent--------#
        sentinal_current = self.sentinal_embed2(self.sentinal_embed1(h_lang))  # batch* 2rnn_size
        sentinal_current = torch.max(sentinal_current.narrow(1, 0, self.rnn_size),
                                     sentinal_current.narrow(1, self.rnn_size, self.rnn_size))  # batch* rnn_size
        state = state + tuple(pre_sentinal) + (torch.stack([sentinal_current, torch.zeros_like(sentinal_current)]),)
        # --end-------generate recurrent--------#

        return output, state


class TopDownCatWeightedSentinalCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCatWeightedSentinalCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)
        self.sen_attention = SentinalAttention(opt)

        # -------generate sentinal--------#
        self.i2h_2 = nn.Linear(opt.rnn_size * 3, opt.rnn_size)
        self.h2h_2 = nn.Linear(opt.rnn_size, opt.rnn_size)

        self.sentinal_embed1 = self.sentinal_embed2 = lambda x: x

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)
        model_utils.xavier_uniform('sigmoid', self.i2h_2, self.h2h_2)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        pre_sentinal = state[2:]
        sentinal = torch.cat([_[0].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        # sentinal = F.dropout(sentinal, self.drop_prob_rnn, self.training)

        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        weighted_sentinal = self.sen_attention(h_att, sentinal)  # batch_size * rnn_size
        att = self.attention(h_att, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size

        lang_lstm_input = torch.cat([att, weighted_sentinal, F.dropout(h_att, self.drop_prob_rnn, self.training)],
                                    1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate recurrent--------#
        ada_gate_point = F.sigmoid(self.i2h_2(lang_lstm_input) + self.h2h_2(prev_h))  # batch*rnn_size
        # sentinal_current = F.dropout(ada_gate_point * F.tanh(c_lang), self.drop_prob_lm, self.training)     # batch*rnn_size
        sentinal_current = ada_gate_point * F.tanh(c_lang)  # batch*rnn_size
        sentinal_current = self.sentinal_embed2(self.sentinal_embed1(sentinal_current))
        state = state + tuple(pre_sentinal) + (torch.stack([sentinal_current, torch.zeros_like(sentinal_current)]),)
        # --end-------generate recurrent--------#

        return output, state


class TopDownCatWeightedHiddenCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCatWeightedHiddenCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn
        self.rnn_size = opt.rnn_size

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)
        self.sen_attention = SentinalAttention(opt)

        # -------generate sentinal--------#
        self.sentinal_embed1 = nn.Linear(opt.rnn_size, 2 * opt.rnn_size, bias=False)
        self.sentinal_embed2 = lambda x: x

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)
        model_utils.xavier_normal('linear', self.sentinal_embed1)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        pre_sentinal = state[2:]
        sentinal = torch.cat([_[0].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        # sentinal = F.dropout(sentinal, self.drop_prob_rnn, self.training)

        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        weighted_sentinal = self.sen_attention(h_att, sentinal)  # batch_size * rnn_size
        att = self.attention(h_att, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size

        lang_lstm_input = torch.cat([att, weighted_sentinal, F.dropout(h_att, self.drop_prob_rnn, self.training)],
                                    1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate recurrent--------#
        sentinal_current = self.sentinal_embed2(self.sentinal_embed1(h_lang))  # batch* 2rnn_size
        sentinal_current = torch.max(sentinal_current.narrow(1, 0, self.rnn_size),
                                     sentinal_current.narrow(1, self.rnn_size, self.rnn_size))  # batch* rnn_size
        state = state + tuple(pre_sentinal) + (torch.stack([sentinal_current, torch.zeros_like(sentinal_current)]),)
        # --end-------generate recurrent--------#

        return output, state


class TopDownCatWeightedSentinalBaseAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCatWeightedSentinalBaseAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)
        self.sen_attention = SentinalAttention(opt)

        # -------generate sentinal--------#
        self.i2h_2 = nn.Linear(opt.rnn_size * 3, opt.rnn_size)
        self.h2h_2 = nn.Linear(opt.rnn_size, opt.rnn_size)

        self.sentinal_embed1 = self.sentinal_embed2 = lambda x: x

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)
        model_utils.xavier_uniform('sigmoid', self.i2h_2, self.h2h_2)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, average_att_feat, att_masks=None):
        pre_sentinal = state[2:]
        sentinal = torch.cat([_[0].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        # sentinal = F.dropout(sentinal, self.drop_prob_rnn, self.training)

        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        weighted_sentinal = self.sen_attention(average_att_feat, sentinal)  # batch_size * rnn_size
        att = self.attention(h_att, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size

        lang_lstm_input = torch.cat([att, weighted_sentinal, F.dropout(h_att, self.drop_prob_rnn, self.training)],
                                    1)  # batch_size * 3rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate recurrent--------#
        ada_gate_point = F.sigmoid(self.i2h_2(lang_lstm_input) + self.h2h_2(prev_h))  # batch*rnn_size
        # sentinal_current = F.dropout(ada_gate_point * F.tanh(c_lang), self.drop_prob_lm, self.training)     # batch*rnn_size
        sentinal_current = ada_gate_point * F.tanh(c_lang)  # batch*rnn_size
        sentinal_current = self.sentinal_embed2(self.sentinal_embed1(sentinal_current))
        state = state + tuple(pre_sentinal) + (torch.stack([sentinal_current, torch.zeros_like(sentinal_current)]),)
        # --end-------generate recurrent--------#

        return output, state


class TopDownUpCatWeightedSentinalCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownUpCatWeightedSentinalCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)
        self.sen_attention = SentinalAttention(opt)

        # -------generate sentinal--------#
        self.i2h_2 = nn.Linear(opt.rnn_size * 2, opt.rnn_size)
        self.h2h_2 = nn.Linear(opt.rnn_size, opt.rnn_size)

        self.sentinal_embed1 = self.sentinal_embed2 = lambda x: x

        # output
        self.h2_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.ws_affine = nn.Linear(opt.rnn_size, opt.rnn_size)

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)
        model_utils.xavier_uniform('sigmoid', self.i2h_2, self.h2h_2)
        model_utils.xavier_normal('linear', self.h2_affine, self.ws_affine)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        pre_sentinal = state[2:]
        sentinal = torch.cat([_[0].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        # sentinal = F.dropout(sentinal, self.drop_prob_rnn, self.training)

        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size

        lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_rnn, self.training)],
                                    1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        weighted_sentinal = self.sen_attention(h_lang, sentinal)  # batch_size * rnn_size

        affined = self.h2_affine(h_lang) + self.ws_affine(weighted_sentinal)  # batch_size * rnn_size

        output = F.dropout(affined, self.drop_prob_lm, self.training)  # batch_size * rnn_size

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate recurrent--------#
        ada_gate_point = F.sigmoid(self.i2h_2(lang_lstm_input) + self.h2h_2(prev_h))  # batch*rnn_size
        # sentinal_current = F.dropout(ada_gate_point * F.tanh(c_lang), self.drop_prob_lm, self.training)     # batch*rnn_size
        sentinal_current = ada_gate_point * F.tanh(c_lang)  # batch*rnn_size
        sentinal_current = self.sentinal_embed2(self.sentinal_embed1(sentinal_current))
        state = state + tuple(pre_sentinal) + (torch.stack([sentinal_current, torch.zeros_like(sentinal_current)]),)
        # --end-------generate recurrent--------#

        return output, state


class TopDownUpCatWeightedHiddenCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownUpCatWeightedHiddenCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn
        self.rnn_size = opt.rnn_size

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)
        self.sen_attention = SentinalAttention(opt)

        # -------generate sentinal--------#
        self.sentinal_embed1 = nn.Linear(opt.rnn_size, 2 * opt.rnn_size, bias=False)
        self.sentinal_embed2 = lambda x: x

        # output
        self.h2_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.ws_affine = nn.Linear(opt.rnn_size, opt.rnn_size)

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)
        model_utils.xavier_normal('linear', self.h2_affine, self.ws_affine, self.sentinal_embed1)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        pre_sentinal = state[2:]
        sentinal = torch.cat([_[0].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        # sentinal = F.dropout(sentinal, self.drop_prob_rnn, self.training)

        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size

        lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_rnn, self.training)],
                                    1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        weighted_sentinal = self.sen_attention(h_lang, sentinal)  # batch_size * rnn_size

        affined = self.h2_affine(h_lang) + self.ws_affine(weighted_sentinal)  # batch_size * rnn_size

        output = F.dropout(affined, self.drop_prob_lm, self.training)  # batch_size * rnn_size

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate recurrent--------#
        sentinal_current = self.sentinal_embed2(self.sentinal_embed1(h_lang))  # batch* 2rnn_size
        sentinal_current = torch.max(sentinal_current.narrow(1, 0, self.rnn_size),
                                     sentinal_current.narrow(1, self.rnn_size, self.rnn_size))  # batch* rnn_size
        state = state + tuple(pre_sentinal) + (torch.stack([sentinal_current, torch.zeros_like(sentinal_current)]),)
        # --end-------generate recurrent--------#

        return output, state


class TopDownUpCatWeightedHiddenCore1(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownUpCatWeightedHiddenCore1, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn
        self.rnn_size = opt.rnn_size

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)
        self.sen_attention = SentinalAttention(opt)

        # -------generate sentinal--------#
        self.sentinal_embed1 = nn.Linear(opt.rnn_size, 2 * opt.rnn_size, bias=False)
        self.sentinal_embed2 = lambda x: x

        # output
        self.h2_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.ws_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.drop = nn.Dropout(self.drop_prob_lm)

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)
        model_utils.xavier_normal('linear', self.h2_affine, self.ws_affine, self.sentinal_embed1)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        pre_sentinal = state[2:]
        sentinal = torch.cat([_[0].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        # sentinal = F.dropout(sentinal, self.drop_prob_rnn, self.training)

        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size

        lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_rnn, self.training)],
                                    1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        weighted_sentinal = self.sen_attention(h_lang, sentinal)  # batch_size * rnn_size

        affined = self.h2_affine(self.drop(h_lang)) + self.ws_affine(
            self.drop(weighted_sentinal))  # batch_size * rnn_size

        output = F.dropout(affined, self.drop_prob_lm, self.training)  # batch_size * rnn_size

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate recurrent--------#
        sentinal_current = self.sentinal_embed2(self.sentinal_embed1(h_lang))  # batch* 2rnn_size
        sentinal_current = torch.max(sentinal_current.narrow(1, 0, self.rnn_size),
                                     sentinal_current.narrow(1, self.rnn_size, self.rnn_size))  # batch* rnn_size
        state = state + tuple(pre_sentinal) + (torch.stack([sentinal_current, torch.zeros_like(sentinal_current)]),)
        # --end-------generate recurrent--------#

        return output, state


class TopDownUpCatWeightedHiddenCore2(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownUpCatWeightedHiddenCore2, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn
        self.rnn_size = opt.rnn_size

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)
        self.sen_attention = SentinalAttention(opt)

        # -------generate sentinal--------#
        self.sentinal_embed1 = nn.Linear(opt.rnn_size, 2 * opt.rnn_size, bias=False)
        self.sentinal_embed2 = lambda x: x

        # output
        self.h2_affine = nn.Linear(opt.rnn_size, int(opt.rnn_size / 2))
        self.ws_affine = nn.Linear(opt.rnn_size, int(opt.rnn_size / 2))
        self.drop = nn.Dropout(self.drop_prob_lm)

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)
        model_utils.xavier_normal('linear', self.h2_affine, self.ws_affine, self.sentinal_embed1)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        pre_sentinal = state[2:]
        sentinal = torch.cat([_[0].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        # sentinal = F.dropout(sentinal, self.drop_prob_rnn, self.training)

        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size

        lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_rnn, self.training)],
                                    1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        weighted_sentinal = self.sen_attention(h_lang, sentinal)  # batch_size * rnn_size

        affined = torch.cat([self.h2_affine(self.drop(h_lang)), self.ws_affine(self.drop(weighted_sentinal))],
                            1)  # batch_size * rnn_size

        output = F.dropout(affined, self.drop_prob_lm, self.training)  # batch_size * rnn_size

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate recurrent--------#
        sentinal_current = self.sentinal_embed2(self.sentinal_embed1(h_lang))  # batch* 2rnn_size
        sentinal_current = torch.max(sentinal_current.narrow(1, 0, self.rnn_size),
                                     sentinal_current.narrow(1, self.rnn_size, self.rnn_size))  # batch* rnn_size
        state = state + tuple(pre_sentinal) + (torch.stack([sentinal_current, torch.zeros_like(sentinal_current)]),)
        # --end-------generate recurrent--------#

        return output, state


class TopDownUpCatWeightedHiddenCore3(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownUpCatWeightedHiddenCore3, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn
        self.drop_prob_output = opt.drop_prob_output
        self.rnn_size = opt.rnn_size

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)
        if opt.weighted_hidden:
            self.sen_attention = SentinalAttention(opt)
        else:
            self.sen_attention = self.average_hiddens

        # -------generate sentinal--------#
        self.sentinal_embed1 = nn.Linear(opt.rnn_size, opt.rnn_size, bias=False)
        self.sentinal_embed2 = lambda x: x

        # output
        self.h2_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.ws_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.drop = nn.Dropout(self.drop_prob_output)

        if opt.nonlinear == 'relu':
            self.tgh = nn.ReLU()
            model_utils.kaiming_normal('relu', 0, self.h2_affine, self.ws_affine)
        elif opt.nonlinear == 'prelu':
            self.tgh = nn.PReLU()
            model_utils.kaiming_normal('leaky_relu', 0.25, self.h2_affine, self.ws_affine)
        elif opt.nonlinear == 'lecun_tanh':
            self.tgh = lambda x: 1.7159 * F.tanh((2.0 / 3.0) * x)
            model_utils.xavier_uniform('tanh', self.h2_affine, self.ws_affine)
        elif opt.nonlinear == 'maxout':
            del self.h2_affine, self.ws_affine
            self.h2_affine = nn.Linear(opt.rnn_size, 2 * opt.rnn_size)
            self.ws_affine = nn.Linear(opt.rnn_size, 2 * opt.rnn_size)
            self.tgh = lambda x: torch.max(x.narrow(1, 0, self.rnn_size),
                                           x.narrow(1, self.rnn_size, self.rnn_size))
            model_utils.xavier_normal('linear', self.h2_affine, self.ws_affine)
        elif opt.nonlinear == 'tanh':
            self.tgh = nn.Tanh()
            model_utils.xavier_normal('linear', self.h2_affine, self.ws_affine)


        if opt.sentinel_nonlinear == 'relu':
            self.sentinel_nonlinear = nn.ReLU()
            model_utils.kaiming_normal('relu', 0, self.sentinal_embed1)
        elif opt.sentinel_nonlinear == 'prelu':
            self.sentinel_nonlinear = nn.PReLU()
            model_utils.kaiming_normal('leaky_relu', 0.25, self.sentinal_embed1)
        elif opt.sentinel_nonlinear == 'lecun_tanh':
            self.sentinel_nonlinear = lambda x: 1.7159 * F.tanh((2.0 / 3.0) * x)
            model_utils.xavier_uniform('tanh', self.sentinal_embed1)
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

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        pre_sentinal = state[2:]
        sentinal = torch.cat([_[0].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        # sentinal = F.dropout(sentinal, self.drop_prob_rnn, self.training)

        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size

        lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_rnn, self.training)],
                                    1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        weighted_sentinal = self.sen_attention(h_lang, sentinal)  # batch_size * rnn_size

        affined = self.tgh(
            self.h2_affine(self.drop(h_lang)) + self.ws_affine(self.drop(weighted_sentinal)))  # batch_size * rnn_size

        output = F.dropout(affined, self.drop_prob_lm, self.training)  # batch_size * rnn_size

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate recurrent--------#
        sentinal_current = self.sentinal_embed2(self.sentinal_embed1(h_lang))  # batch* rnn_size
        sentinal_current = self.sentinel_nonlinear(sentinal_current)  # batch* rnn_size
        state = state + tuple(pre_sentinal) + (torch.stack([sentinal_current, torch.zeros_like(sentinal_current)]),)
        # --end-------generate recurrent--------#

        return output, state

    def average_hiddens(self, h_lang, hiddens):
        weighted_sentinal = torch.mean(hiddens, 1)  # batch_size * rnn_size
        return weighted_sentinal

class TopDownUpCatAverageHiddenCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownUpCatAverageHiddenCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn
        self.drop_prob_output = opt.drop_prob_output
        self.rnn_size = opt.rnn_size

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)
        # self.sen_attention = SentinalAttention(opt)

        # -------generate sentinal--------#
        self.sentinal_embed1 = nn.Linear(opt.rnn_size, 2 * opt.rnn_size, bias=False)
        self.sentinal_embed2 = lambda x: x

        # output
        self.h2_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.ws_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.drop = nn.Dropout(self.drop_prob_output)
        self.tgh = nn.Tanh()

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)
        model_utils.xavier_normal('linear', self.sentinal_embed1)
        model_utils.xavier_uniform('tanh', self.h2_affine, self.ws_affine)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        pre_sentinal = state[2:]
        sentinal = torch.cat([_[0].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        # sentinal = F.dropout(sentinal, self.drop_prob_rnn, self.training)

        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size

        lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_rnn, self.training)],
                                    1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        # weighted_sentinal = self.sen_attention(h_lang, sentinal)  # batch_size * rnn_size
        weighted_sentinal = torch.mean(sentinal, 1)  # batch_size * rnn_size

        affined = self.tgh(
            self.h2_affine(self.drop(h_lang)) + self.ws_affine(self.drop(weighted_sentinal)))  # batch_size * rnn_size

        output = F.dropout(affined, self.drop_prob_lm, self.training)  # batch_size * rnn_size

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate recurrent--------#
        sentinal_current = self.sentinal_embed2(self.sentinal_embed1(h_lang))  # batch* 2rnn_size
        sentinal_current = torch.max(sentinal_current.narrow(1, 0, self.rnn_size),
                                     sentinal_current.narrow(1, self.rnn_size, self.rnn_size))  # batch* rnn_size
        state = state + tuple(pre_sentinal) + (torch.stack([sentinal_current, torch.zeros_like(sentinal_current)]),)
        # --end-------generate recurrent--------#

        return output, state


class TopDown2LayerUpCatWeightedHiddenCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDown2LayerUpCatWeightedHiddenCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn
        self.drop_prob_output = opt.drop_prob_output
        self.rnn_size = opt.rnn_size

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)
        self.sen_attention_lang = SentinalAttention(opt)

        self.lang_first = opt.lang_first
        # -------generate sentinal--------#
        self.sentinal_embed_lang = lambda x: x

        # -------generate sentinal att layer--------#
        self.sen_attention_att = SentinalAttention(opt)
        self.sentinal_embed_att = lambda x: x
        self.h1_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.ws_att_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.drop_att = nn.Dropout(opt.drop_prob_att)

        # output
        self.h2_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.ws_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.drop = nn.Dropout(self.drop_prob_output)

        self.tgh = nn.Tanh()
        model_utils.xavier_normal('linear', self.h2_affine, self.ws_affine, self.h1_affine, self.ws_att_affine)

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)


    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        pre_sentinal = state[2:]
        sentinal_att = torch.cat([_[0].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        sentinal_lang = torch.cat([_[1].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        # sentinal = F.dropout(sentinal, self.drop_prob_rnn, self.training)

        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        # att layer history output
        weighted_sentinal_att = self.sen_attention_att(h_att, sentinal_att)  # batch_size * rnn_size
        h_att_ws = self.tgh(self.h1_affine(self.drop_att(h_att)) + self.ws_att_affine(
            self.drop_att(weighted_sentinal_att)))  # batch_size * rnn_size

        if self.lang_first:
            att = self.attention(h_att_ws, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size
        else:
            att = self.attention(h_att, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size

        lang_lstm_input = torch.cat([att, F.dropout(h_att_ws, self.drop_prob_rnn, self.training)],
                                    1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        weighted_sentinal = self.sen_attention_lang(h_lang, sentinal_lang)  # batch_size * rnn_size

        affined = self.tgh(
            self.h2_affine(self.drop(h_lang)) + self.ws_affine(self.drop(weighted_sentinal)))  # batch_size * rnn_size

        output = F.dropout(affined, self.drop_prob_lm, self.training)  # batch_size * rnn_size

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate recurrent--------#
        sentinal_att = self.sentinal_embed_att(h_att)  # batch* rnn_size
        sentinal_lang = self.sentinal_embed_lang(h_lang)  # batch* rnn_size

        state = state + tuple(pre_sentinal) + (torch.stack([sentinal_att, sentinal_lang]),)
        # --end-------generate recurrent--------#

        return output, state


class TopDownAttLayerUpCatWeightedHiddenCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownAttLayerUpCatWeightedHiddenCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn
        self.drop_prob_output = opt.drop_prob_output
        self.rnn_size = opt.rnn_size

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)
        # self.sen_attention_lang = SentinalAttention(opt)

        self.lang_first = opt.lang_first
        # -------generate sentinal--------#
        # self.sentinal_embed_lang = lambda x: x

        # -------generate sentinal att layer--------#
        self.sen_attention_att = SentinalAttention(opt)
        self.sentinal_embed_att = lambda x: x
        self.h1_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.ws_att_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.drop_att = nn.Dropout(opt.drop_prob_att)

        # output
        # self.h2_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        # self.ws_affine = nn.Linear(opt.rnn_size, opt.rnn_size)
        # self.drop = nn.Dropout(self.drop_prob_output)

        self.tgh = nn.Tanh()
        model_utils.xavier_normal('linear', self.h1_affine, self.ws_att_affine)

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)


    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        pre_sentinal = state[2:]
        sentinal_att = torch.cat([_[0].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        # sentinal_lang = torch.cat([_[1].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        # sentinal = F.dropout(sentinal, self.drop_prob_rnn, self.training)

        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        # att layer history output
        weighted_sentinal_att = self.sen_attention_att(h_att, sentinal_att)  # batch_size * rnn_size
        h_att_ws = self.tgh(self.h1_affine(self.drop_att(h_att)) + self.ws_att_affine(
            self.drop_att(weighted_sentinal_att)))  # batch_size * rnn_size

        if self.lang_first:
            att = self.attention(h_att_ws, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size
        else:
            att = self.attention(h_att, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size

        lang_lstm_input = torch.cat([att, F.dropout(h_att_ws, self.drop_prob_rnn, self.training)],
                                    1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        # weighted_sentinal = self.sen_attention_lang(h_lang, sentinal_lang)  # batch_size * rnn_size

        # affined = self.tgh(
        #     self.h2_affine(self.drop(h_lang)) + self.ws_affine(self.drop(weighted_sentinal)))  # batch_size * rnn_size

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)  # batch_size * rnn_size

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate recurrent--------#
        sentinal_att = self.sentinal_embed_att(h_att)  # batch* rnn_size
        # sentinal_lang = self.sentinal_embed_lang(h_lang)  # batch* rnn_size

        state = state + tuple(pre_sentinal) + (torch.stack([sentinal_att, torch.zeros_like(sentinal_att)]),)
        # --end-------generate recurrent--------#

        return output, state


class TopDownUpCatWeightedHiddenCore4(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownUpCatWeightedHiddenCore4, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn
        self.rnn_size = opt.rnn_size

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)
        self.sen_attention = SentinalAttention(opt)

        # -------generate sentinal--------#
        self.sentinal_embed1 = nn.Linear(opt.rnn_size, 2 * opt.rnn_size, bias=False)
        self.sentinal_embed2 = lambda x: x

        # output
        self.h2_affine = nn.Linear(opt.rnn_size, int(opt.rnn_size / 2))
        self.ws_affine = nn.Linear(opt.rnn_size, int(opt.rnn_size / 2))
        self.drop = nn.Dropout(self.drop_prob_lm)
        self.tgh = nn.Tanh()

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)
        model_utils.xavier_normal('linear', self.sentinal_embed1)
        model_utils.xavier_uniform('tanh', self.h2_affine, self.ws_affine)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        pre_sentinal = state[2:]
        sentinal = torch.cat([_[0].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        # sentinal = F.dropout(sentinal, self.drop_prob_rnn, self.training)

        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size

        lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_rnn, self.training)],
                                    1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        weighted_sentinal = self.sen_attention(h_lang, sentinal)  # batch_size * rnn_size

        affined = self.tgh(torch.cat([self.h2_affine(self.drop(h_lang)), self.ws_affine(self.drop(weighted_sentinal))],
                                     1))  # batch_size * rnn_size

        output = F.dropout(affined, self.drop_prob_lm, self.training)  # batch_size * rnn_size

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate recurrent--------#
        sentinal_current = self.sentinal_embed2(self.sentinal_embed1(h_lang))  # batch* 2rnn_size
        sentinal_current = torch.max(sentinal_current.narrow(1, 0, self.rnn_size),
                                     sentinal_current.narrow(1, self.rnn_size, self.rnn_size))  # batch* rnn_size
        state = state + tuple(pre_sentinal) + (torch.stack([sentinal_current, torch.zeros_like(sentinal_current)]),)
        # --end-------generate recurrent--------#

        return output, state


class TopDownUpAddWeightedSentinalCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownUpAddWeightedSentinalCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)
        self.sen_attention = SentinalAttention(opt)

        # -------generate sentinal--------#
        self.i2h_2 = nn.Linear(opt.rnn_size * 2, opt.rnn_size)
        self.h2h_2 = nn.Linear(opt.rnn_size, opt.rnn_size)

        self.sentinal_embed1 = self.sentinal_embed2 = lambda x: x

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)
        model_utils.xavier_uniform('sigmoid', self.i2h_2, self.h2h_2)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        pre_sentinal = state[2:]
        sentinal = torch.cat([_[0].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        # sentinal = F.dropout(sentinal, self.drop_prob_rnn, self.training)

        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size

        lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_rnn, self.training)],
                                    1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        weighted_sentinal = self.sen_attention(h_lang, sentinal)  # batch_size * rnn_size
        output = F.dropout(h_lang + weighted_sentinal, self.drop_prob_lm, self.training)  # batch_size * rnn_size

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate recurrent--------#
        ada_gate_point = F.sigmoid(self.i2h_2(lang_lstm_input) + self.h2h_2(prev_h))  # batch*rnn_size
        # sentinal_current = F.dropout(ada_gate_point * F.tanh(c_lang), self.drop_prob_lm, self.training)     # batch*rnn_size
        sentinal_current = ada_gate_point * F.tanh(c_lang)  # batch*rnn_size
        sentinal_current = self.sentinal_embed2(self.sentinal_embed1(sentinal_current))
        state = state + tuple(pre_sentinal) + (torch.stack([sentinal_current, torch.zeros_like(sentinal_current)]),)
        # --end-------generate recurrent--------#

        return output, state


class TopDownUpCatWeightedSentinalBaseAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownUpCatWeightedSentinalBaseAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)
        self.sen_attention = SentinalAttention(opt)

        # -------generate sentinal--------#
        self.i2h_2 = nn.Linear(opt.rnn_size * 2, opt.rnn_size)
        self.h2h_2 = nn.Linear(opt.rnn_size, opt.rnn_size)

        self.sentinal_embed1 = self.sentinal_embed2 = lambda x: x

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)
        model_utils.xavier_uniform('sigmoid', self.i2h_2, self.h2h_2)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, average_att_feat, att_masks=None):
        pre_sentinal = state[2:]
        sentinal = torch.cat([_[0].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        # sentinal = F.dropout(sentinal, self.drop_prob_rnn, self.training)

        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size

        lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_rnn, self.training)],
                                    1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        weighted_sentinal = self.sen_attention(average_att_feat, sentinal)  # batch_size * rnn_size
        output = torch.cat([F.dropout(h_lang, self.drop_prob_lm, self.training),
                            F.dropout(weighted_sentinal, self.drop_prob_lm, self.training)],
                           1)  # batch_size * 2rnn_size

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate recurrent--------#
        ada_gate_point = F.sigmoid(self.i2h_2(lang_lstm_input) + self.h2h_2(prev_h))  # batch*rnn_size
        # sentinal_current = F.dropout(ada_gate_point * F.tanh(c_lang), self.drop_prob_lm, self.training)     # batch*rnn_size
        sentinal_current = ada_gate_point * F.tanh(c_lang)  # batch*rnn_size
        sentinal_current = self.sentinal_embed2(self.sentinal_embed1(sentinal_current))
        state = state + tuple(pre_sentinal) + (torch.stack([sentinal_current, torch.zeros_like(sentinal_current)]),)
        # --end-------generate recurrent--------#

        return output, state


class TopDownUpAddWeightedSentinalBaseAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownUpAddWeightedSentinalBaseAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_rnn = opt.drop_prob_rnn

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)
        self.sen_attention = SentinalAttention(opt)

        # -------generate sentinal--------#
        self.i2h_2 = nn.Linear(opt.rnn_size * 2, opt.rnn_size)
        self.h2h_2 = nn.Linear(opt.rnn_size, opt.rnn_size)

        self.sentinal_embed1 = self.sentinal_embed2 = lambda x: x

        # initialization
        model_utils.lstm_init(self.att_lstm)
        model_utils.lstm_init(self.lang_lstm)
        model_utils.xavier_uniform('sigmoid', self.i2h_2, self.h2h_2)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, average_att_feat, att_masks=None):
        pre_sentinal = state[2:]
        sentinal = torch.cat([_[0].unsqueeze(1) for _ in pre_sentinal], 1)  # [batch_size, num_recurrent, rnn_size]
        # sentinal = F.dropout(sentinal, self.drop_prob_rnn, self.training)

        prev_h = F.dropout(state[0][-1], self.drop_prob_rnn, self.training)  # [batch_size, rnn_size]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [batch_size, 2*rnn_size + input_encoding_size]

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # both are [batch_size, rnn_size]

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)  # batch_size * rnn_size

        lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_rnn, self.training)],
                                    1)  # batch_size * 2rnn_size
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # batch*rnn_size

        weighted_sentinal = self.sen_attention(average_att_feat, sentinal)  # batch_size * rnn_size
        output = F.dropout(h_lang + weighted_sentinal, self.drop_prob_lm, self.training)  # batch_size * rnn_size

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        # --start-------generate recurrent--------#
        ada_gate_point = F.sigmoid(self.i2h_2(lang_lstm_input) + self.h2h_2(prev_h))  # batch*rnn_size
        # sentinal_current = F.dropout(ada_gate_point * F.tanh(c_lang), self.drop_prob_lm, self.training)     # batch*rnn_size
        sentinal_current = ada_gate_point * F.tanh(c_lang)  # batch*rnn_size
        sentinal_current = self.sentinal_embed2(self.sentinal_embed1(sentinal_current))
        state = state + tuple(pre_sentinal) + (torch.stack([sentinal_current, torch.zeros_like(sentinal_current)]),)
        # --end-------generate recurrent--------#

        return output, state


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

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1, bias=False)

        # initialization
        model_utils.xavier_uniform('tanh', self.h2att)
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

        return att_res


class AttentionRecurrent(nn.Module):
    def __init__(self, opt):
        super(AttentionRecurrent, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1, bias=False)

        # ----sentinal attention-----#
        if opt.caption_model == 'topdown_original_weighted_sentinel':
            self.senti2att = nn.Linear(opt.att_feat_size, self.att_hid_size)
        else:
            self.senti2att = nn.Linear(self.rnn_size, self.att_hid_size)

        # initialization
        model_utils.xavier_uniform('tanh', self.h2att, self.senti2att)
        model_utils.kaiming_normal('relu', 0, self.alpha_net)

    def forward(self, h, att_feats, p_att_feats, sentinal, att_masks=None):
        # The p_att_feats here is already projected
        num_hidden = sentinal.size(1)
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)  # 36
        att = p_att_feats.view(-1, att_size, self.att_hid_size)  # [batch_size, 36, att_hid_size]

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = F.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot, dim=1)  # batch * (att_size)
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))  # batch * att_size * rnn_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * rnn_size

        # --start----recurrent attention (including sentinal attention, recurrent hidden attention, recurrent sentinal attention)-----#
        att_h = self.h2att(h)  # batch * att_hid_size
        att_senti = self.senti2att(sentinal)  # batch * num_hidden * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att_senti)  # batch * num_hidden * att_hid_size
        dot_senti = att_senti + att_h  # batch * num_hidden * att_hid_size
        dot_senti = F.tanh(dot_senti)  # batch * num_hidden * att_hid_size
        dot_senti = dot_senti.view(-1, self.att_hid_size)  # (batch*num_hidden) * att_hid_size
        dot_senti = self.alpha_net(dot_senti)  # (batch*num_hidden) * 1
        dot_senti = dot_senti.view(-1, num_hidden)  # batch * num_hidden

        dot_union = torch.cat([dot, dot_senti], 1)  # batch * (att_size+num_hidden)
        weight_senti = F.softmax(dot_union, dim=1)  # batch * (att_size+num_hidden)
        beta = weight_senti[:, att_size:]  # batch * num_hidden
        beta = beta.unsqueeze(1)  # batch * 1 * num_hidden
        beta_sum = beta.sum(2)  # batch * 1
        att_res_senti = (1 - beta_sum) * att_res + torch.bmm(beta, sentinal).squeeze(1)  # batch * rnn_size
        # --end----recurrent attention-----#

        return att_res_senti  # batch * rnn_size


class SentinalAttention(nn.Module):
    def __init__(self, opt):
        super(SentinalAttention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1, bias=False)

        # ----sentinal attention-----#
        self.senti2att = nn.Linear(self.rnn_size, self.att_hid_size)

        # initialization
        model_utils.xavier_uniform('tanh', self.h2att, self.senti2att)
        model_utils.kaiming_normal('relu', 0, self.alpha_net)

    def forward(self, h, sentinal):
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

        weight_senti = F.softmax(dot_senti, dim=1)  # batch * num_hidden
        weight_senti = weight_senti.unsqueeze(1)  # batch * 1 * num_hidden

        att_res_senti = torch.bmm(weight_senti, sentinal).squeeze(1)  # batch * rnn_size
        # --end----recurrent attention-----#

        return att_res_senti  # batch * rnn_size


class O_SentinalAttention(SentinalAttention):
    def __init__(self, opt):
        super(O_SentinalAttention, self).__init__(opt)

        # ----sentinal attention-----#
        del self.senti2att
        self.senti2att = nn.Linear(opt.att_feat_size, self.att_hid_size)

        # initialization
        model_utils.xavier_uniform('tanh', self.senti2att)


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
        model_utils.xavier_uniform('tanh', self.ctx2att)


class TopDownOriginal2Model(AttModel):
    def __init__(self, opt):
        super(TopDownOriginal2Model, self).__init__(opt)
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x: x
        self.num_layers = 2
        self.core = TopDownOriginal2Core(opt)


class TopDownSentinalAffine2Model(AttModel):
    def __init__(self, opt):
        super(TopDownSentinalAffine2Model, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownSentinalAffine2Core(opt)


class TopDownRecurrentHiddenModel(AttModel):
    def __init__(self, opt):
        super(TopDownRecurrentHiddenModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownRecurrentHiddenCore(opt)


class TopDownWeightedHiddenModel(AttModel):
    def __init__(self, opt):
        super(TopDownWeightedHiddenModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownWeightedHiddenCore(opt)


class TopDownCatWeightedSentinalModel(AttModel):
    def __init__(self, opt):
        super(TopDownCatWeightedSentinalModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownCatWeightedSentinalCore(opt)


class TopDownCatWeightedHiddenModel(AttModel):
    def __init__(self, opt):
        super(TopDownCatWeightedHiddenModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownCatWeightedHiddenCore(opt)


class TopDownUpCatWeightedSentinalModel(AttModel):
    def __init__(self, opt):
        super(TopDownUpCatWeightedSentinalModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownUpCatWeightedSentinalCore(opt)


class TopDownUpCatWeightedHiddenModel(AttModel):
    def __init__(self, opt):
        super(TopDownUpCatWeightedHiddenModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownUpCatWeightedHiddenCore(opt)


class TopDownUpCatWeightedHiddenModel_1(AttModel):
    def __init__(self, opt):
        super(TopDownUpCatWeightedHiddenModel_1, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownUpCatWeightedHiddenCore1(opt)


class TopDownUpCatWeightedHiddenModel_2(AttModel):
    def __init__(self, opt):
        super(TopDownUpCatWeightedHiddenModel_2, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownUpCatWeightedHiddenCore2(opt)


class TopDownUpCatWeightedHiddenModel_3(AttModel):
    def __init__(self, opt):
        super(TopDownUpCatWeightedHiddenModel_3, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownUpCatWeightedHiddenCore3(opt)


class TopDownUpCatWeightedHiddenModel_4(AttModel):
    def __init__(self, opt):
        super(TopDownUpCatWeightedHiddenModel_4, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownUpCatWeightedHiddenCore4(opt)


class TopDownUpCatAverageHiddenModel(AttModel):
    def __init__(self, opt):
        super(TopDownUpCatAverageHiddenModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownUpCatAverageHiddenCore(opt)


class TopDown2LayerUpCatWeightedHiddenModel(AttModel):
    def __init__(self, opt):
        super(TopDown2LayerUpCatWeightedHiddenModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDown2LayerUpCatWeightedHiddenCore(opt)


class TopDownAttLayerUpCatWeightedHiddenModel(AttModel):
    def __init__(self, opt):
        super(TopDownAttLayerUpCatWeightedHiddenModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownAttLayerUpCatWeightedHiddenCore(opt)


class TopDownUpAddWeightedSentinalModel(AttModel):
    def __init__(self, opt):
        super(TopDownUpAddWeightedSentinalModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownUpAddWeightedSentinalCore(opt)


class TopDownUpCatWeightedSentinalBaseAttModel(AttModel):
    def __init__(self, opt):
        super(TopDownUpCatWeightedSentinalBaseAttModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownUpCatWeightedSentinalBaseAttCore(opt)

        del self.logit
        self.logit = nn.Linear(2 * self.rnn_size, self.vocab_size + 1)
        model_utils.kaiming_normal('relu', 0, self.logit)

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, average_att_feat, state):
        # 'it' contains a word index
        xt = self.embed(it)  # [batch_size, input_encoding_size]

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, average_att_feat,
                                  att_masks)  # batch*rn_size
        logprobs = F.log_softmax(self.logit(output), dim=1)  # batch*(vocab_size+1)

        return logprobs, state


class TopDownUpAddWeightedSentinalBaseAttModel(AttModel):
    def __init__(self, opt):
        super(TopDownUpAddWeightedSentinalBaseAttModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownUpAddWeightedSentinalBaseAttCore(opt)

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, average_att_feat, state):
        # 'it' contains a word index
        xt = self.embed(it)  # [batch_size, input_encoding_size]

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, average_att_feat,
                                  att_masks)  # batch*rn_size
        logprobs = F.log_softmax(self.logit(output), dim=1)  # batch*(vocab_size+1)

        return logprobs, state


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




class TopDownCatWeightedSentinalBaseAttModel(AttModel):
    def __init__(self, opt):
        super(TopDownCatWeightedSentinalBaseAttModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownCatWeightedSentinalBaseAttCore(opt)

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
