from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import math
import time
import os
from six.moves import cPickle
from copy import deepcopy

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward


try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def variables_histogram(data, iteration, outputs, tb_summary_writer, opt):
    # results, p_fc_feats, p_att_feats, att_hiddens, lan_hiddens, sentinels = outputs
    results, p_fc_feats, p_att_feats, att_hiddens, lan_hiddens, att_sentinels, lang_sentinels, lang_weights = outputs
    if opt.tensorboard_mid_variables:
        # add original fc_feats histogram
        tb_summary_writer.add_histogram('fc_feat', data['fc_feats'], iteration)
        # add original att_feats histogram
        tb_summary_writer.add_histogram('att_feat', data['att_feats'], iteration)
        # add affined fc_feats histogram
        tb_summary_writer.add_histogram('p_fc_feat', p_fc_feats, iteration)
        # add affined att_feat histogram
        tb_summary_writer.add_histogram('p_att_feat', p_att_feats, iteration)

        # add att_hiddens histogram
        for i in range(len(att_hiddens)):
            tb_summary_writer.add_histogram('att_hiddens/_'+str(i+1), att_hiddens[i], iteration)
        # add lan_hiddens histogram
        for i in range(len(lan_hiddens)):
            tb_summary_writer.add_histogram('lan_hiddens/_'+str(i+1), lan_hiddens[i], iteration)

        # add att_sentinel histogram
        tb_summary_writer.add_histogram('att_sentinel', att_sentinels, iteration)
        # add lang_sentinel histogram
        tb_summary_writer.add_histogram('lang_sentinel', lang_sentinels, iteration)
    if opt.tensorboard_lang_weights:
        # add lang_weights histogram
        tb_summary_writer.add_histogram('lang_weights', lang_weights, iteration)
        print('language weights when the hidden number is 6:')
        print(lang_weights)

# def flatten_params(model):
#     return torch.cat([param.data.view(-1) for param in model.parameters()], 0)

def load_params(model, avg_p_list):
    for p, avg_p in zip(model.parameters(), avg_p_list):
        p.data.copy_(avg_p)

def eva_ave_model(avg_param, best_val_score_ave_model, crit, infos, iteration, loader, model, opt, tb_summary_writer):
    original_param = deepcopy(list(p.data for p in model.parameters()))  # save current params
    load_params(model, avg_param)  # load the average

    # eval model on training data
    if opt.train_eval_split is not None:
        eval_kwargs = {'split': opt.train_eval_split,
                       'dataset': opt.input_json}
        eval_kwargs.update(vars(opt))
        train_eval_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)
        # Write train_eval result into summary
        add_summary_value(tb_summary_writer, 'validation_loss/train_ave_model', train_eval_loss, iteration)
        if lang_stats is not None:
            for k, v in lang_stats.items():
                add_summary_value(tb_summary_writer, k + '/train_ave_model', v, iteration)

    # eval model on valid data
    if opt.val_split is not None:
        eval_kwargs = {'split': opt.val_split,
                       'dataset': opt.input_json}
        eval_kwargs.update(vars(opt))
        val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)
        # Write validation result into summary
        add_summary_value(tb_summary_writer, 'validation_loss/val_ave_model', val_loss, iteration)
        if lang_stats is not None:
            for k, v in lang_stats.items():
                add_summary_value(tb_summary_writer, k + '/val_ave_model', v, iteration)

    # eval model on test data
    if opt.test_split is not None:
        eval_kwargs = {'split': opt.test_split,
                       'dataset': opt.input_json}
        eval_kwargs.update(vars(opt))
        val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)
        # Write validation result into summary
        add_summary_value(tb_summary_writer, 'validation_loss/test_ave_model', val_loss, iteration)
        if lang_stats is not None:
            for k, v in lang_stats.items():
                add_summary_value(tb_summary_writer, k + '/test_ave_model', v, iteration)

    # Save model if is improving on validation result
    if opt.language_eval == 1:
        current_score = lang_stats['CIDEr']
    else:
        current_score = - val_loss
    best_flag = False
    if best_val_score_ave_model is None or current_score >= best_val_score_ave_model:
        best_val_score_ave_model = current_score
        best_flag = True
    checkpoint_path = os.path.join(opt.checkpoint_path, 'ave_model.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))
    # Dump miscalleous informations
    infos['best_val_score_ave_model'] = best_val_score_ave_model
    with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '.pkl'), 'wb') as f:
        cPickle.dump(infos, f)
    if best_flag:
        checkpoint_path = os.path.join(opt.checkpoint_path, 'ave-best.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print("model saved to {}".format(checkpoint_path))
        with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '-ave_model-best.pkl'), 'wb') as f:
            cPickle.dump(infos, f)
    load_params(model, original_param)  # restore parameters

    return best_val_score_ave_model, infos

def eva_original_model(best_val_score, crit, epoch, histories, infos, iteration, loader, loss_history, lr_history, model, opt,
                       optimizer, ss_prob_history, tb_summary_writer, val_result_history):
    # eval model on training data
    if opt.train_eval_split is not None:
        eval_kwargs = {'split': opt.train_eval_split,
                       'dataset': opt.input_json}
        eval_kwargs.update(vars(opt))
        train_eval_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)
        # Write train_eval result into summary
        add_summary_value(tb_summary_writer, 'validation_loss/train', train_eval_loss, iteration)
        if lang_stats is not None:
            for k, v in lang_stats.items():
                add_summary_value(tb_summary_writer, k + '/train', v, iteration)

    # eval model on valid data
    if opt.val_split is not None:
        eval_kwargs = {'split': opt.val_split,
                       'dataset': opt.input_json}
        eval_kwargs.update(vars(opt))
        val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)
        # Write validation result into summary
        add_summary_value(tb_summary_writer, 'validation_loss/val', val_loss, iteration)
        if lang_stats is not None:
            for k, v in lang_stats.items():
                add_summary_value(tb_summary_writer, k + '/val', v, iteration)
        val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

    # eval model on test data
    if opt.test_split is not None:
        eval_kwargs = {'split': opt.test_split,
                       'dataset': opt.input_json}
        eval_kwargs.update(vars(opt))
        val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)
        # Write validation result into summary
        add_summary_value(tb_summary_writer, 'validation_loss/test', val_loss, iteration)
        if lang_stats is not None:
            for k, v in lang_stats.items():
                add_summary_value(tb_summary_writer, k + '/test', v, iteration)
        val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}


    # Save model if is improving on validation result
    if opt.language_eval == 1:
        current_score = lang_stats['CIDEr']
    else:
        current_score = - val_loss

    best_flag = False
    # if True: # if true
    if best_val_score is None or current_score >= best_val_score:
        best_val_score = current_score
        best_flag = True

    checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))
    optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
    torch.save(optimizer.state_dict(), optimizer_path)
    # Dump miscalleous informations
    infos['iter'] = iteration
    infos['epoch'] = epoch
    infos['iterators'] = loader.iterators
    infos['split_ix'] = loader.split_ix
    infos['best_val_score'] = best_val_score
    infos['opt'] = opt
    infos['vocab'] = loader.get_vocab()
    histories['val_result_history'] = val_result_history
    histories['loss_history'] = loss_history
    histories['lr_history'] = lr_history
    histories['ss_prob_history'] = ss_prob_history
    with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '.pkl'), 'wb') as f:
        cPickle.dump(infos, f)
    with open(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + '.pkl'), 'wb') as f:
        cPickle.dump(histories, f)
    if best_flag:
        checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print("model saved to {}".format(checkpoint_path))
        with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '-best.pkl'), 'wb') as f:
            cPickle.dump(infos, f)

    return best_val_score, histories, infos


def train(opt):
    assert opt.annfile is not None and len(opt.annfile)>0

    print('Checkpoint path is ' + opt.checkpoint_path)
    print('This program is using GPU '+ str(os.environ['CUDA_VISIBLE_DEVICES']))
    # Deal with feature things before anything
    opt.use_att = utils.if_use_att(opt.caption_model)
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        if opt.load_best:
            info_path = os.path.join(opt.start_from, 'infos_' + opt.id + '-best.pkl')
        else:
            info_path = os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl')
        with open(info_path) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    if opt.learning_rate_decay_start is None:
        opt.learning_rate_decay_start = infos.get('opt', None).learning_rate_decay_start
    # if opt.load_best:
    #     opt.self_critical_after = epoch
    elif opt.learning_rate_decay_start == -1 and opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
        opt.learning_rate_decay_start = epoch

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    # loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
        best_val_score_ave_model = infos.get('best_val_score_ave_model', None)

    model = models.setup(opt).cuda()
    dp_model = torch.nn.DataParallel(model)

    update_lr_flag = True
    # Assure in training mode
    dp_model.train()

    crit = utils.LanguageModelCriterion(opt.XE_eps)
    rl_crit = utils.RewardCriterion()

    # build_optimizer
    optimizer = build_optimizer(model, opt)

    # Load the optimizer
    if opt.load_opti and vars(opt).get('start_from', None) is not None and opt.load_best==0 and os.path.isfile(os.path.join(opt.start_from, "optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    # initialize the running average of parameters
    avg_param = deepcopy(list(p.data for p in model.parameters()))

    # make evaluation using original model
    best_val_score, histories, infos = eva_original_model(best_val_score, crit, epoch, histories, infos, iteration,
                                                              loader, loss_history, lr_history,
                                                              model, opt, optimizer, ss_prob_history, tb_summary_writer,
                                                              val_result_history)

    while True:
        if update_lr_flag:
            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                if opt.lr_decay == 'exp':
                    frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                    decay_factor = opt.learning_rate_decay_rate ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
                elif opt.lr_decay == 'cosine':
                    lr_epoch = min((epoch-opt.learning_rate_decay_start), opt.lr_max_epoch)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * lr_epoch / opt.lr_max_epoch))
                    decay_factor = (1 - opt.lr_cosine_decay_base) * cosine_decay + opt.lr_cosine_decay_base
                    opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate

            lr = [opt.current_lr]
            if opt.att_normalize_method is not None and '6' in opt.att_normalize_method:
                lr = [opt.current_lr, opt.lr_ratio*opt.current_lr]

            utils.set_lr(optimizer, lr)
            print('learning rate is: ' + str(lr))

            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            update_lr_flag = False

        # Update the iteration
        iteration += 1

        # Load data from train split (0)
        data = loader.get_batch(opt.train_split)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp
        
        optimizer.zero_grad()
        if not sc_flag:
            output = dp_model(fc_feats, att_feats, labels, att_masks)
            # calculate loss
            loss = crit(output[0], labels[:,1:], masks[:,1:])

            # add some middle variable histogram
            if iteration % (4*opt.losses_log_every) == 0:
                outputs = [_.data.cpu().numpy() if _ is not None else None for _ in output]
                variables_histogram(data, iteration, outputs, tb_summary_writer, opt)

        else:
            gen_result, sample_logprobs = dp_model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
            reward = get_self_critical_reward(dp_model, fc_feats, att_feats, att_masks, data, gen_result, opt)
            loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())

        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_max_norm)
        # add_summary_value(tb_summary_writer, 'grad_L2_norm', grad_norm, iteration)

        optimizer.step()
        train_loss = loss.item()
        torch.cuda.synchronize()
        end = time.time()

        # compute the running average of parameters
        for p, avg_p in zip(model.parameters(), avg_param):
            avg_p.mul_(opt.beta).add_((1.0-opt.beta), p.data)

        if iteration%10==0:
            if not sc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, end - start))
            else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, np.mean(reward[:,0]), end - start))

        # Update the epoch
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:,0]), iteration)

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        if opt.tensorboard_weights_grads and (iteration % (8*opt.losses_log_every) == 0):
            # add weights histogram to tensorboard summary
            for name, param in model.named_parameters():
                if (opt.tensorboard_parameters_name is None or sum([p_name in name for p_name in opt.tensorboard_parameters_name]) > 0) and param.grad is not None:
                    tb_summary_writer.add_histogram('Weights_' + name.replace('.', '/'), param, iteration)
                    tb_summary_writer.add_histogram('Grads_' + name.replace('.', '/'), param.grad, iteration)

        if opt.tensorboard_buffers and (iteration % (opt.losses_log_every) == 0):
            for name, buffer in model.named_buffers():
                if (opt.tensorboard_buffers_name is None or sum([p_name in name for p_name in opt.tensorboard_buffers_name]) > 0) and buffer is not None:
                    add_summary_value(tb_summary_writer, name.replace('.', '/'), buffer, iteration)

        if opt.distance_sensitive_coefficient and iteration % (4 * opt.losses_log_every) == 0:
                print('The coefficient in intra_att_att_lstm is as follows:')
                print(model.core.intra_att_att_lstm.coefficient.data.cpu().tolist())
                print('The coefficient in intra_att_lang_lstm is as follows:')
                print(model.core.intra_att_lang_lstm.coefficient.data.cpu().tolist())
        if opt.distance_sensitive_bias and iteration % (4 * opt.losses_log_every) == 0:
                print('The bias in intra_att_att_lstm is as follows:')
                print(model.core.intra_att_att_lstm.bias.data.cpu().tolist())
                print('The bias in intra_att_lang_lstm is as follows:')
                print(model.core.intra_att_lang_lstm.bias.data.cpu().tolist())

        # make evaluation using original model
        if (iteration % opt.save_checkpoint_every == 0):
            best_val_score, histories, infos = eva_original_model(best_val_score, crit, epoch, histories, infos, iteration, loader, loss_history, lr_history,
                               model, opt, optimizer, ss_prob_history, tb_summary_writer, val_result_history)

        # make evaluation with the averaged parameters model
        if iteration > opt.ave_threshold and (iteration % opt.save_checkpoint_every == 0):
            best_val_score_ave_model, infos = eva_ave_model(avg_param, best_val_score_ave_model, crit, infos, iteration, loader, model, opt,
                          tb_summary_writer)

        # # Stop if reaching max epochs
        # if epoch >= opt.max_epochs and opt.max_epochs != -1:
        #     break

        if iteration >= opt.max_iter:
            break


def build_optimizer(model, opt):
    optimized = [{'params': list(model.parameters())}]
    # if opt.att_normalize_method is '6', finetuning will be needed
    if opt.att_normalize_method is not None and '6' in opt.att_normalize_method:
        fine_tuned = list(model.core.lang_lstm.parameters())
        fine_tuned.extend(list(model.core.h2_affine.parameters()))
        fine_tuned.extend(list(model.logit.parameters()))
        if opt.att_normalize_method == '6-1' or opt.att_normalize_method == '6-1-0' or opt.att_normalize_method == '6-3':
            fine_tuned.extend(model.core.att_linear_project.parameters())

        optimized = list(set(optimized[0]['params']).difference(set(fine_tuned)))
        optimized = [{'params': optimized},
                     {'params': fine_tuned}]

    optimizer = utils.build_optimizer(optimized, opt)

    return optimizer


opt = opts.parse_opt()

#----for my local set----#
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# net
opt.id = 'BUTD-RL_cosine50-lrdecay_lstm_adaptive-prodis_online7'
opt.caption_model = 'topdown_up_cat_weighted_hidden_3'
opt.encoded_feat_size = 512
opt.input_encoding_size = 512
opt.rnn_size = 512
opt.drop_prob_embed = 0
opt.drop_prob_fcfeat = 0.3
opt.drop_prob_attfeat = 0.3
opt.drop_prob_rnn1 = 0
opt.drop_prob_rnn2 = 0
opt.drop_prob_output = 0.3
opt.drop_prob_attention = 0

opt.use_bn = 2
opt.fc_use_bn = 2
opt.emb_use_bn = 0
opt.BN_other = False

opt.load_best = 0
opt.self_critical_after = 0
opt.start_from = 'Experiments/Experiments_hdd/Visualized/BUTD-RL_cosine50-lrdecay_lstm_adaptive-prodis_online7'

opt.cappro = False
opt.cappro_method = 'dis'       #  'pro', 'dis', 'pro-dis'
opt.cappro_pro_normal = False
opt.cappro_dis_normal = False

opt.linearprodis = True
opt.prodis_method = 'adaptive-prodis'           # pro, dis, adaptive-prodis, adaptive-prodis-detach

opt.visatt_RMSfilter = False
opt.visatt_RMSfilter_sum1 = False

opt.save_att_statics = False

opt.att_normalize_method = None      # '1', '2', '3', None
opt.fc_normalize_method = None
opt.att_std_weight_init = 0.25
opt.att_std_bias_init = 0.15
opt.att_normalize_rate = 1
opt.lr_ratio = 15
opt.att_norm_reg_paras_path = None

opt.tensorboard_lang_weights = False
opt.tensorboard_weights_grads = True
opt.tensorboard_parameters_name = ['logit.weight', 'att_lstm.weight_ih', 'att_lstm.weight_hh', 'lang_lstm.weight_ih',
                                   'lang_lstm.weight_hh', 'fc_embed.1.weight', 'att_embed.1.weight']
opt.tensorboard_buffers = True
opt.tensorboard_buffers_name = ['a1_min', 'a2_max']
opt.tensorboard_mid_variables = False
opt.print_lang_weights = False

opt.nonlinear = 'tanh'   # relu, prelu, lecun_tanh, maxout, tanh,
opt.sentinel_nonlinear = 'x'    # relu, prelu, lecun_tanh, maxout, tanh, x

opt.noh2pre = True
opt.pre = 'h'   # h or c

opt.input_feature = 'fc'        # fc, att_mean, attention

opt.output_attention = False

opt.nonlocal_dy = False
opt.project_g = False
opt.nonlocal_residual = False
opt.nonlocal_dy_bn = False
opt.nonlocal_dy_insert = False

opt.language_attention = False
opt.weighted_hidden = True
opt.transfer_horizontal = False
opt.att_score_method = 'general'    # mlp, dot, general, cosine, scaled_dot
opt.lang_no_h0 = False   # language attention use or not h0
opt.Intergrate2vector = 'concat'    # concat, self_attention, self_attention_tying_alpha
opt.Intergrate_instead = True
opt.lang_att_before_pro = True      # if true, use language attention on h_lang, otherwise, on projected h_lang

opt.emb_relu = False

opt.tie_weights = False

opt.project_hidden = True

opt.attention_gate = False

opt.skip_connection = None    # 'RES' ,'CAT', ['HW', 'HW-normal', 'HW_1', 'HW_1-normal', 'HW-softmax']
opt.skip_ele = False
opt.skip_sum_1 = False

opt.LSTMN = True
opt.LSTMN_att = False
opt.LSTMN_att_att_score_method = 'mlp'      # only mlp
opt.LSTMN_lang = False
opt.LSTMN_lang_att_score_method = 'general'     # mlp, general, scaled_dot
opt.LSTMN_last_att_hidden = False
opt.LSTM_att_hi = False
opt.adaptive_t0 = False
opt.LSTMN_att_key_hi_ci = 0         # 0 indicates 'hi', 1 indicates 'ci'
opt.input_first_att = 2             # 1: only word embedding, 2: 1 + last h_2_t-1, 3: 2 + global_fc
opt.input_second_att = 1            # 1: only hidden state of the first lstm, 2: 1 + attended visual feature

opt.lstm_layer_norm = True
opt.layer_norm = False
opt.norm_input = False
opt.norm_output = False
opt.norm_hidden = False

opt.LN_out_embedding = False

opt.not_use_first = False  # LSTMN use or not first timestep

opt.position_sentive_inter = False

opt.word_sensitive_bias = False
opt.word_sensitive_coefficient = False
opt.word_sensitive_hi_xi = 0

opt.distance_sensitive_bias = False
opt.distance_bias_9_1 = False
opt.distance_sensitive_coefficient = False

# data
opt.specified_id = []
opt.dataset = 'coco'
opt.annfile = '/home/wzn/PycharmProjects/self-critical.pytorch/coco-caption/annotations/captions_val2014.json'
opt.input_json = 'data/MSCOCO/cocotalk.json'
opt.input_fc_dir = '/home/wzn/Datasets/ImageCaption/MSCOCO/detection_features/trainval_20-100/trainval_fc'
opt.input_att_dir = '/home/wzn/Datasets/ImageCaption/MSCOCO/detection_features/trainval_20-100/trainval_att'
opt.input_label_h5 = 'data/MSCOCO/cocotalk_label.h5'
opt.cached_tokens = '/home/wzn/PycharmProjects/self-critical.pytorch/data/MSCOCO/coco-train-idxs.p'
opt.train_split = 'raw_train'
opt.train_eval_split = None
opt.val_split = 'online_val'
opt.test_split = None

# optimization
opt.load_opti = False
opt.lr_decay = 'cosine'        # 'exp', 'cosine'
opt.lr_max_epoch = 35
opt.lr_cosine_decay_base = 0.05
opt.batch_size = 64
opt.optim = 'adam'
opt.learning_rate = 2e-5
opt.learning_rate_decay_start = -1
opt.learning_rate_decay_every = 10
opt.optim_alpha = 0.9
opt.max_iter = 600000
opt.beam_size = 5
opt.ave_threshold = 600000
opt.XE_eps = 0

opt.checkpoint_path = 'Experiments/Experiments_hdd/Visualized/BUTD-RL_cosine50-lrdecay_lstm_adaptive-prodis_online7'
# opt.scheduled_sampling_start = 0
opt.save_checkpoint_every = 6000
opt.val_images_use = 2000
opt.test_images_use = 5000
opt.train_eval_images_use = 4000
opt.max_epochs = math.ceil(float(opt.max_iter)/(113287/opt.batch_size))
opt.language_eval = 1

models.LinearProDis.prodis_method = opt.prodis_method
#----for my local set----#

train(opt)
