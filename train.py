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

def train(opt):
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
    if opt.load_best:
        opt.self_critical_after = epoch

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model = models.setup(opt).cuda()
    dp_model = torch.nn.DataParallel(model)

    update_lr_flag = True
    # Assure in training mode
    dp_model.train()

    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()

    optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and opt.load_best==0 and os.path.isfile(os.path.join(opt.start_from, "optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    while True:
        if update_lr_flag:
            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr)

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

        # opt.current_lr = opt.learning_rate * ((1 - float(iteration) / opt.max_iter) ** opt.power)
        # utils.set_lr(optimizer, opt.current_lr)

        # Update the iteration
        iteration += 1
                
        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        # print('Read data:', time.time() - start)

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
                # results, p_fc_feats, p_att_feats, att_hiddens, lan_hiddens, sentinels = outputs
                results, p_fc_feats, att_hiddens, lan_hiddens, att_sentinels, lang_sentinels = outputs
                # add original fc_feats histogram
                tb_summary_writer.add_histogram('fc_feat', data['fc_feats'], iteration)

                # add original att_feats histogram
                tb_summary_writer.add_histogram('att_feat',  data['att_feats'], iteration)

                # add affined fc_feats histogram
                tb_summary_writer.add_histogram('p_fc_feat', p_fc_feats, iteration)

                # # add affined att_feat histogram
                # tb_summary_writer.add_histogram('p_att_feat', p_att_feats, iteration)

                # add att_hiddens histogram
                tb_summary_writer.add_histogram('att_hiddens', att_hiddens, iteration)

                # add lan_hiddens histogram
                tb_summary_writer.add_histogram('lan_hiddens', lan_hiddens, iteration)

                # add att_sentinel histogram
                tb_summary_writer.add_histogram('att_sentinel', att_sentinels, iteration)

                # add lang_sentinel histogram
                tb_summary_writer.add_histogram('lang_sentinel', lang_sentinels, iteration)

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

        if (iteration % (8*opt.losses_log_every) == 0):
            # add weights histogram to tensorboard summary
            for name, param in model.named_parameters():
                if param.grad is not None:
                    tb_summary_writer.add_histogram('Weights_' + name.replace('.', '/'), param, iteration)
                    tb_summary_writer.add_histogram('Grads_' + name.replace('.', '/'), param.grad, iteration)

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):

            # eval model on training data
            eval_kwargs = {'split': 'train_eval',
                           'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            train_eval_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)

            # Write train_eval result into summary
            add_summary_value(tb_summary_writer, 'validation_loss/train', train_eval_loss, iteration)
            if lang_stats is not None:
                for k, v in lang_stats.items():
                    add_summary_value(tb_summary_writer, k+'/train', v, iteration)

            # eval model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)

            # Write validation result into summary
            add_summary_value(tb_summary_writer, 'validation_loss/val', val_loss, iteration)
            if lang_stats is not None:
                for k,v in lang_stats.items():
                    add_summary_value(tb_summary_writer, k+'/val', v, iteration)
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True: # if true
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
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # # Stop if reaching max epochs
        # if epoch >= opt.max_epochs and opt.max_epochs != -1:
        #     break

        if iteration >= opt.max_iter:
            break

opt = opts.parse_opt()


#----for my local set----#
# net
opt.id = 'topdown_original'
opt.caption_model = 'topdown_original'
opt.rnn_size = 1000
opt.input_encoding_size = 1000

# data
opt.input_json = 'data/cocotalk.json'
opt.input_fc_dir = '/home/wzn/Datasets/ImageCaption/MSCOCO/detection_features/trainval_36/trainval_36_fc'
opt.input_att_dir = '/home/wzn/Datasets/ImageCaption/MSCOCO/detection_features/trainval_36/trainval_36_att'
opt.input_label_h5 = 'data/cocotalk_label.h5'

# optimization
opt.batch_size = 32
opt.optim = 'adam'
opt.learning_rate = 5e-4
opt.optim_alpha = 0.9
opt.max_iter = 240000
opt.power = None
opt.weight_decay = 0.0
opt.grad_max_norm = None
opt.beam_size = 5
opt.learning_rate_decay_start = 0
opt.drop_prob_lm = 0.6

opt.checkpoint_path = 'Experiments/top-down-original-debug'
opt.scheduled_sampling_start = 0
opt.save_checkpoint_every = 6000
opt.val_images_use = 5000
opt.max_epochs = math.ceil(float(opt.max_iter)/(113287/opt.batch_size))
opt.language_eval = 1
#----for my local set----#


train(opt)
