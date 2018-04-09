import yaml
from model.las_model import Listener,Speller
from util.functions import batch_iterator
from util.kaldi_feat import KaldiReadIn,Language,OneHotEncode
import numpy as np
from torch.autograd import Variable
import torch
import sys
import time
#import pdb
# Load config file for experiment
#pdb.set_trace()
try:
    config_path = sys.argv[1]
    conf = yaml.load(open(config_path,'r'))
except:
    print('Usage: python3 run_exp.py <config file path>')

# Parameters loading
torch.manual_seed(conf['training_parameter']['seed'])
num_epochs = conf['training_parameter']['num_epochs']
use_pretrained = conf['training_parameter']['use_pretrained']
training_msg = 'epoch_{:2d}_step_{:3d}_TrLoss_{:.4f}_TrWER_{:.4f}'
epoch_end_msg = 'epoch_{:2d}_TrLoss_{:.4f}_TrWER_{:.4f}_ValidLoss_{:.4f}_ValidWER_{:.4f}_time_{:.2f}'
verbose_step = conf['training_parameter']['verbose_step']
tf_rate_upperbound = conf['training_parameter']['tf_rate_upperbound']
tf_rate_lowerbound = conf['training_parameter']['tf_rate_lowerbound']
tf_decay_epoch = conf['training_parameter']['tf_decay_epoch']

lang = Language(conf['model_parameter']['language_scp'])
output_class_dim = lang.n_words
conf['model_parameter']['output_class_dim']=lang.n_words

max_lab_len = conf['model_parameter']['max_label_len']
train_scp_path = conf['model_parameter']['train_scp_path']
train_lab_path = conf['model_parameter']['train_lab_path']
valid_scp_path = conf['model_parameter']['valid_scp_path']
valid_lab_path = conf['model_parameter']['valid_lab_path']
train_set = KaldiReadIn(lang, train_scp_path, train_lab_path, **conf['model_parameter'],**conf['training_parameter'])
valid_set = KaldiReadIn(lang, valid_scp_path, valid_lab_path, **conf['model_parameter'],**conf['training_parameter'])

# Construct LAS Model or load pretrained LAS model
if not use_pretrained:
    traing_log = open(conf['meta_variable']['training_log_dir']+conf['meta_variable']['experiment_name']+'.log','w')
    listener = Listener(**conf['model_parameter'])
    speller = Speller(**conf['model_parameter'])
else:
    traing_log = open(conf['meta_variable']['training_log_dir']+conf['meta_variable']['experiment_name']+'.log','a')
    listener = torch.load(conf['training_parameter']['pretrained_listener_path'])
    speller = torch.load(conf['training_parameter']['pretrained_speller_path'])

optimizer = torch.optim.Adam([{'params':listener.parameters()}, {'params':speller.parameters()}],
                             lr=conf['training_parameter']['learning_rate'])
listener_model_path = conf['meta_variable']['checkpoint_dir']+conf['meta_variable']['experiment_name']+'.listener'
speller_model_path = conf['meta_variable']['checkpoint_dir']+conf['meta_variable']['experiment_name']+'.speller'

###print arguments setting
for key in conf:
    print('{}:'.format(key))
    for para in conf[key]:
        print('{:50}:{}'.format(para,conf[key][para]))
    print('\n')
print('please check.',flush=True)

best_ler = 1.0
for epoch in range(num_epochs):
    epoch_head = time.time()
    tr_loss = 0.0
    tr_ler = []
    tt_loss = 0.0
    tt_ler = []
    valid_loss = 0.0
    valid_ler = []

    # Teacher forcing rate linearly decay after a certain epoch
    if epoch < tf_decay_epoch:
        tf_rate = 1.0
    else:
        tf_rate = tf_rate_upperbound - (tf_rate_upperbound-tf_rate_lowerbound)*((epoch - tf_decay_epoch)/(num_epochs - tf_decay_epoch))
    # Training
    ###train_set return:
    ###batch_data: (batch, T, fea_dim), batch_label: (batch, T), batch_length: (batch, T),true length of utterence
    for batch_index,(batch_data,batch_label,batch_length) in enumerate(train_set):
        batch_label = OneHotEncode(batch_label, max_lab_len, max_idx = output_class_dim)
        batch_data = torch.from_numpy(batch_data)
        batch_label = torch.from_numpy(batch_label)
        #if batch_index ==100:
        #    pdb.set_trace()
            #break
        #pdb.set_trace()
        batch_loss, batch_ler = batch_iterator(batch_data, batch_label, listener, speller, optimizer, 
                                               tf_rate, is_training=True, **conf['model_parameter'])
        tr_loss += batch_loss
        tr_ler.extend(batch_ler)
        if (batch_index+1) % verbose_step == 0:
            print(training_msg.format(epoch+1,batch_index+1,tr_loss[0]/(batch_index+1),sum(tr_ler)/len(tr_ler)),end='\n',flush=True)
    training_time = float(time.time()-epoch_head)
    
    # Validating
    for batch_index_val,(batch_data,batch_label,batch_length) in enumerate(valid_set):
        batch_label = OneHotEncode(batch_label, max_lab_len, max_idx = output_class_dim)
        batch_data = torch.from_numpy(batch_data)
        batch_label = torch.from_numpy(batch_label)
        batch_loss, batch_ler = batch_iterator(batch_data, batch_label, listener, speller, optimizer, 
                                               tf_rate, is_training=False, **conf['model_parameter'])
        valid_loss += batch_loss
        valid_ler.extend(batch_ler)
    
    # Logger
    print(epoch_end_msg.format(epoch+1,tr_loss[0]/(batch_index+1),sum(tr_ler)/len(tr_ler),
        valid_loss[0]/(batch_index_val+1),sum(valid_ler)/len(valid_ler),training_time),flush=True)
    print(epoch_end_msg.format(epoch+1,tr_loss[0]/(batch_index+1),sum(tr_ler)/len(tr_ler),
        valid_loss[0]/(batch_index_val+1),sum(valid_ler)/len(valid_ler),training_time),flush=True,file=traing_log)
    
    curr_save_path = listener_model_path + '.epoch{}'.format(epoch)
    torch.save(listener,curr_save_path)
    curr_save_path = speller_model_path + '.epoch{}'.format(epoch)
    torch.save(speller,curr_save_path)

    # Checkpoint
    # save checkpoint with the best ler
    if best_ler >= sum(valid_ler)/len(valid_ler):
        best_ler = sum(valid_ler)/len(valid_ler)
        torch.save(listener, listener_model_path)
        torch.save(speller, speller_model_path)
