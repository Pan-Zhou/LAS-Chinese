import yaml
from model.las_model import Listener,Speller
#from model.las_model_sampling import Listener,Speller
from util.kaldi_feat import KaldiReadIn,Language,OneHotEncode
from util.beamsearch import BeamSearch
from util.functions import LetterErrorRate
import numpy as np
from torch.autograd import Variable
import torch
import sys
import time
import pdb
# Load config file for experiment
#pdb.set_trace()
try:
    config_path = sys.argv[1]
    conf = yaml.load(open(config_path,'r'))
except:
    print('Usage: python3 run_decode.py <config file path>')

# Parameters loading
torch.manual_seed(conf['training_parameter']['seed'])
use_pretrained = conf['training_parameter']['use_pretrained']
epoch_end_msg = 'epoch_{:2d}_decodeWER_{:.4f}_time_{:.2f}'
lang = Language(conf['model_parameter']['language_scp'])
output_class_dim = lang.n_words
conf['model_parameter']['output_class_dim']=lang.n_words

max_lab_len = conf['model_parameter']['max_label_len']
valid_scp_path = conf['model_parameter']['valid_scp_path']
valid_lab_path = conf['model_parameter']['valid_lab_path']
valid_set = KaldiReadIn(lang, valid_scp_path, valid_lab_path, **conf['model_parameter'],**conf['training_parameter'])

# Construct LAS Model or load pretrained LAS model
#pdb.set_trace()
if not use_pretrained:
    traing_log = open(conf['meta_variable']['training_log_dir']+conf['meta_variable']['experiment_name']+'.log','w')
    listener = Listener(**conf['model_parameter'])
    speller = Speller(**conf['model_parameter'])
else:
    traing_log = open(conf['meta_variable']['training_log_dir']+conf['meta_variable']['experiment_name']+'.log','w')
    listener = torch.load(conf['training_parameter']['pretrained_listener_path'])
    speller = torch.load(conf['training_parameter']['pretrained_speller_path'])


###print arguments setting
for key in conf:
    print('{}:'.format(key))
    for para in conf[key]:
        print('{:50}:{}'.format(para,conf[key][para]))
    print('\n')
print('please check.',flush=True)

epoch_head = time.time()

    ###batch_data: (batch, T, fea_dim), batch_label: (batch, T), batch_length: (batch, T),true length of utterence
beam_search_decoder = BeamSearch(listener,speller,**conf['model_parameter'])    
beam_width = conf['model_parameter']['beam_width']
# Testing
errors =0
chars = 0
for batch_index_val,(batch_data,batch_label,batch_length,lab_len,utt_list) in enumerate(valid_set):
    #batch_label = OneHotEncode(batch_label, max(lab_len), max_idx = output_class_dim)
    batch_data = Variable(torch.from_numpy(batch_data))
    #batch_label = torch.from_numpy(batch_label)
    #pdb.set_trace()
    nbest_rst = beam_search_decoder.beam_search(batch_data)
    
    print('{}'.format(utt_list[0]),file=traing_log)
    tag =[]
    for idx in batch_label[0]:
        if idx ==1:
            break
        if idx ==0:
            continue
        tag.append(lang.index2word[idx])
    print('lab:{}'.format(' '.join(tag)),flush=True, file = traing_log)
    
    for i in range(beam_width):
        rec =[]
        for idx in nbest_rst[i][0]:
            if idx == 1:
                break
            if idx == 0:
                continue
            rec.append(lang.index2word[idx])
        print('rec,best{:2d}:{},{}'.format(i+1,' '.join(rec),nbest_rst[i][1]),flush=True, file = traing_log)
    #pdb.set_trace()
    cur_cer = LetterErrorRate(np.array(nbest_rst[0][0]).reshape(1,len(nbest_rst[0][0])),np.array(batch_label[0]).reshape(1,max(lab_len)))
    errors += cur_cer[0] * len(tag)
    chars += len(tag)
testing_time = float(time.time()-epoch_head)
    # Logger
print('test scp {}'.format(valid_scp_path),flush=True,file=traing_log)
print(epoch_end_msg.format(1,errors/chars,testing_time),flush=True,file=traing_log)
