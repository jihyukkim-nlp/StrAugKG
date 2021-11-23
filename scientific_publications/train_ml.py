import torch.nn as nn
from pykp.masked_loss import masked_cross_entropy
from utils.statistics import LossStatistics
from utils.time_log import time_since
from evaluate import evaluate_loss
import time
import math
import logging
import torch
import sys
import os
from utils.report import export_train_and_valid_loss
import numpy as np

EPS = 1e-8

def train_model(model, optimizer, criterion, train_data_loader, valid_data_loader, opt):
	'''
	generator = SequenceGenerator(model,
								  eos_idx=opt.word2idx[pykp.io.EOS_WORD],
								  beam_size=opt.beam_size,
								  max_sequence_length=opt.max_sent_length
								  )
	'''
	logging.info('======================  Start Training  =========================')

	total_batch = -1
	early_stop_flag = False

	total_train_loss_statistics = LossStatistics()
	report_train_loss_statistics = LossStatistics()
	report_train_ppl = []
	report_valid_ppl = []
	report_train_loss = []
	report_valid_loss = []
	best_valid_ppl = float('inf')
	best_valid_loss = float('inf')
	num_stop_dropping = 0

	if opt.train_from:  # opt.train_from:
		#TODO: load the training state
		raise ValueError("Not implemented the function of load from trained model")
		pass

	model.train()

	for epoch in range(opt.start_epoch, opt.epochs+1):
		if early_stop_flag:
			break

		# TODO: progress bar
		#progbar = Progbar(logger=logging, title='Training', target=len(train_data_loader), batch_size=train_data_loader.batch_size,total_examples=len(train_data_loader.dataset.examples))

		for batch_i, batch in enumerate(train_data_loader):
			total_batch += 1

			# Training
			if opt.train_ml:
				batch_loss_stat, decoder_dist = train_one_batch(batch, model, optimizer, opt, batch_i)
				report_train_loss_statistics.update(batch_loss_stat)
				total_train_loss_statistics.update(batch_loss_stat)
				#logging.info("one_batch")
				#report_loss.append(('train_ml_loss', loss_ml))
				#report_loss.append(('PPL', loss_ml))

				# Brief report
				'''
				if batch_i % opt.report_every == 0:
					brief_report(epoch, batch_i, one2one_batch, loss_ml, decoder_log_probs, opt)
				'''

			#progbar.update(epoch, batch_i, report_loss)

			# Checkpoint, decay the learning rate if validation loss stop dropping, apply early stopping if stop decreasing for several epochs.
			# Save the model parameters if the validation loss improved.
			if total_batch % 4000 == 0:
				print("Epoch %d; batch: %d; total batch: %d" % (epoch, batch_i, total_batch))
				sys.stdout.flush()

			if epoch >= opt.start_checkpoint_at:
				if (opt.checkpoint_interval == -1 and batch_i == len(train_data_loader) - 1) or \
						(opt.checkpoint_interval > -1 and total_batch > 1 and total_batch % opt.checkpoint_interval == 0):
					if opt.train_ml:
						# test the model on the validation dataset for one epoch
						valid_loss_stat = evaluate_loss(valid_data_loader, model, opt)
						model.train()
						current_valid_loss = valid_loss_stat.xent()
						current_valid_ppl = valid_loss_stat.ppl()
						print("Enter check point!")
						sys.stdout.flush()

						current_train_ppl = report_train_loss_statistics.ppl()
						current_train_loss = report_train_loss_statistics.xent()

						# debug
						if math.isnan(current_valid_loss) or math.isnan(current_train_loss):
							logging.info(
								"NaN valid loss. Epoch: %d; batch_i: %d, total_batch: %d" % (epoch, batch_i, total_batch))
							exit()

						if current_valid_loss < best_valid_loss: # update the best valid loss and save the model parameters
							print("Valid loss drops")
							sys.stdout.flush()
							best_valid_loss = current_valid_loss
							best_valid_ppl = current_valid_ppl
							num_stop_dropping = 0
							[os.remove(os.path.join(opt.model_path, _)) for _ in os.listdir(opt.model_path) if _.endswith(".model")]
							check_pt_model_path = os.path.join(opt.model_path, '%s.epoch=%d.batch=%d.total_batch=%d' % (opt.exp, epoch, batch_i, total_batch) + '.model')
							torch.save(  # save model parameters
								model.state_dict(),
								open(check_pt_model_path, 'wb')
							)
							logging.info('Saving checkpoint to %s' % check_pt_model_path)
							torch.save(  # save model parameters
								model.state_dict(),
								open(os.path.join(opt.model_path, 'best.model'), 'wb')
							)

						else:
							print("Valid loss does not drop")
							sys.stdout.flush()
							num_stop_dropping += 1
							# decay the learning rate by a factor
							for i, param_group in enumerate(optimizer.param_groups):
								old_lr = float(param_group['lr'])
								new_lr = old_lr * opt.learning_rate_decay
								if old_lr - new_lr > EPS:
									param_group['lr'] = new_lr

						# log loss, ppl, and time
						#print("check point!")
						#sys.stdout.flush()
						logging.info('Epoch: %d; batch idx: %d; total batches: %d' % (epoch, batch_i, total_batch))
						logging.info(
							'avg training ppl: %.3f; avg validation ppl: %.3f; best validation ppl: %.3f' % (
								current_train_ppl, current_valid_ppl, best_valid_ppl))
						logging.info(
							'avg training loss: %.3f; avg validation loss: %.3f; best validation loss: %.3f' % (
								current_train_loss, current_valid_loss, best_valid_loss))

						report_train_ppl.append(current_train_ppl)
						report_valid_ppl.append(current_valid_ppl)
						report_train_loss.append(current_train_loss)
						report_valid_loss.append(current_valid_loss)

						if num_stop_dropping >= opt.early_stop_tolerance:
							logging.info('Have not increased for %d check points, early stop training' % num_stop_dropping)
							early_stop_flag = True
							break
						report_train_loss_statistics.clear()

	# export the training curve
	train_valid_curve_path = opt.exp_path + '/train_valid_curve'
	export_train_and_valid_loss(report_train_loss, report_valid_loss, report_train_ppl, report_valid_ppl, opt.checkpoint_interval, train_valid_curve_path)
	#logging.info('Overall average training loss: %.3f, ppl: %.3f' % (total_train_loss_statistics.xent(), total_train_loss_statistics.ppl()))


def train_one_batch(batch, model, optimizer, opt, batch_i):
	nodes, nodes_oov, nodes_mask, src_adj, ret_adj, title_adj, src_mask, ret_mask, title_mask, trg, trg_oov, trg_mask, trg_lens, oov_lists, nodes_str_list = batch

	#?@ debugging
	# print(f'nodes={nodes}')
	# print(f'nodes_oov={nodes_oov}')
	# print(f'nodes_mask={nodes_mask}')
	# print(f'src_adj={src_adj}')
	# print(f'ret_adj={ret_adj}')
	# print(f'title_adj={title_adj}')
	# print(f'src_mask={src_mask}')
	# print(f'ret_mask={ret_mask}')
	# print(f'title_mask={title_mask}')
	# print(f'trg={trg}')
	# print(f'trg_oov={trg_oov}')
	# print(f'trg_mask={trg_mask}')
	# print(f'trg_lens={trg_lens}')
	# print(f'oov_lists={oov_lists}')
	# print(f'nodes_str_list={nodes_str_list}')
	# exit()

	batch_size = len(nodes)
	max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

	# move data to GPU if available
	nodes = nodes.to(opt.device)
	nodes_oov = nodes_oov.to(opt.device)
	nodes_mask = nodes_mask.to(opt.device)
	src_adj = src_adj.to(opt.device)
	ret_adj = ret_adj.to(opt.device)
	src_mask = src_mask.to(opt.device)
	ret_mask = ret_mask.to(opt.device)
	# 
	trg = trg.to(opt.device)
	trg_mask = trg_mask.to(opt.device)
	trg_oov = trg_oov.to(opt.device)

	if opt.use_title:
		title_mask = title_mask.to(opt.device)
		title_adj = title_adj.to(opt.device)


	optimizer.zero_grad()

	start_time = time.time()
	# if not opt.one2many:
	decoder_dist, h_t, attention_dist, encoder_final_state = \
		model(
			nodes=nodes, nodes_oov=nodes_oov, nodes_mask=nodes_mask, 
			src_adj=src_adj, ret_adj=ret_adj, 
			src_mask=src_mask, ret_mask=ret_mask,
			trg=trg, trg_oov=trg_oov, max_num_oov=max_num_oov, 
			title_mask=title_mask, title_adj=title_adj, 
		)
	forward_time = time_since(start_time)

	start_time = time.time()
	if opt.copy_attention:  # Compute the loss using target with oov words
		loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,)
	else:  # Compute the loss using target without oov words
		loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,)

	loss_compute_time = time_since(start_time)

	total_trg_tokens = sum(trg_lens)

	if math.isnan(loss.item()):
		print("Batch i: %d" % batch_i)
		raise ValueError("Loss is NaN")

	if opt.loss_normalization == "tokens": # use number of target tokens to normalize the loss
		normalization = total_trg_tokens
	elif opt.loss_normalization == 'batches': # use batch_size to normalize the loss
		normalization = nodes.size(0)
	else:
		raise ValueError('The type of loss normalization is invalid.')

	assert normalization > 0, 'normalization should be a positive number'

	start_time = time.time()
	# back propagation on the normalized loss
	loss.div(normalization).backward()
	backward_time = time_since(start_time)

	if opt.max_grad_norm > 0:
		grad_norm_before_clipping = nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
		# grad_norm_after_clipping = (sum([p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None])) ** (1.0 / 2)
		# logging.info('clip grad (%f -> %f)' % (grad_norm_before_clipping, grad_norm_after_clipping))

	optimizer.step()

	# construct a statistic object for the loss
	stat = LossStatistics(loss.item(), total_trg_tokens, n_batch=1, forward_time=forward_time, loss_compute_time=loss_compute_time, backward_time=backward_time)

	return stat, decoder_dist.detach()
