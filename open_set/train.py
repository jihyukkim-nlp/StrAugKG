import torch
import argparse
import config
import logging
import os
import json
from pykp.model import Seq2SeqModel
from torch.optim import Adam
import pykp

import train_ml

from utils.time_log import time_since
from utils.data_loader import load_data_and_vocab
import time
import numpy as np
import random


def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0

    if opt.delimiter_type == 0:
        opt.delimiter_word = pykp.io.SEP_WORD
    else:
        opt.delimiter_word = pykp.io.EOS_WORD

    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
        opt.model_path = opt.model_path % (opt.exp, opt.timemark)
    else:
        opt.exp_path = os.path.join(opt.exp_path, opt.exp)
        opt.model_path = opt.model_path % (opt.exp, "") # model/%s.%s -> model/this_is_some_thing-for-debugging.ml.
        opt.model_path = opt.model_path[:-1] # model/this_is_some_thing-for-debugging.ml. -> model/this_is_some_thing-for-debugging.ml

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)

    logging.info('EXP_PATH : ' + opt.exp_path)

    # dump the setting (opt) to disk in order to reuse easily
    if opt.train_from:
        opt = torch.load(
            open(os.path.join(opt.model_path, opt.exp + '.initial.config'), 'rb')
        )
    else:
        torch.save(opt,
                   open(os.path.join(opt.model_path, opt.exp + '.initial.config'), 'wb')
                   )
        json.dump(vars(opt), open(os.path.join(opt.model_path, opt.exp + '.initial.json'), 'w'))

    return opt


def init_optimizer_criterion(model, opt):
    criterion = torch.nn.NLLLoss(ignore_index=opt.word2idx[pykp.io.PAD_WORD]).to(opt.device)
    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    return optimizer, criterion


def init_model(opt):
    logging.info('======================  Model Parameters  =========================')

    if opt.copy_attention:
        logging.info('Training a seq2seq model with copy mechanism')
    else:
        logging.info('Training a seq2seq model')
    model = Seq2SeqModel(opt)
    print(model)

    return model.to(opt.device)


def main(opt):
    try:
        start_time = time.time()
        train_data_loader, valid_data_loader, word2idx, idx2word, vocab = load_data_and_vocab(opt, load_train=True)
        load_data_time = time_since(start_time)
        logging.info('Time for loading the data: %.1f' % load_data_time)
        start_time = time.time()
        model = init_model(opt)
        optimizer, criterion = init_optimizer_criterion(model, opt)
        train_ml.train_model(model, optimizer, criterion, train_data_loader, valid_data_loader, opt)
        training_time = time_since(start_time)
        logging.info('Time for training: %.1f' % training_time)
    except Exception as e:
        logging.exception("message")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.vocab_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    opt = parser.parse_args()
    opt = process_opt(opt)
    opt.input_feeding = False
    opt.copy_input_feeding = False

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    if not opt.one2many and opt.one2many_mode > 0:
        raise ValueError("You cannot choose one2many mode without the -one2many options.")

    if opt.one2many and opt.one2many_mode == 0:
        raise ValueError("If you choose one2many, you must specify the one2many mode.")

    if opt.one2many_mode == 1 and opt.num_predictions > 1:
        raise ValueError("If you set the one2many_mode to 1, the number of predictions should also be 1.")

    logging = config.init_logging(log_file=opt.exp_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)