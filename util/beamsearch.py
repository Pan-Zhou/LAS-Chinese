import numpy as np
import torch
from torch.autograd import Variable

class BeamSearch(object):
    def __init__(self, listener, speller, **kwargs):
        self.listener = listener
        self.speller = speller
        self.beam_width = kwargs['beam_width']
        self.max_decode_step = kwargs['max_label_len']
        self.use_gpu = kwargs['use_gpu']

    def check_all_done(self,seqs):
        for seq in seqs:
            if not seq[-1]:
                return False
        return True

    def logAdd(self, a, b):
        pass
    
    def beam_search_step(self,listener_feature, top_seqs):
        all_seqs = []
        for seq in top_seqs:
            if seq[0][-1] == 1:
                #seq[-1] = True
                all_seqs.append(seq)
                continue
            ##generate output distrubution use speller
            last_out = self.speller.embedding(Variable(torch.LongTensor( [seq[0][-1]] )).cuda() )
            last_context = seq[3] if len(seq[3].size())==3 else seq[3].unsqueeze(1)
            rnn_input = torch.cat([last_out.unsqueeze(1),last_context],dim=-1 )
            
            posterior, hidden_state, context, atten_score = self.speller.forward_step(rnn_input,seq[2],listener_feature)
            
            ##sort posterior, should use logAdd to avoid numerical underflow
            post_host = torch.exp(posterior.cpu().data)
            out_post =[(idx,post) for idx,post in enumerate(post_host[0])]
            out_post = sorted(out_post, key = lambda x:x[1],reverse=True)
            
            for i in range(self.beam_width):
                char_idx = out_post[i][0]
                char_post = out_post[i][1]
                ####use logAdd later...
                score = seq[1] *  char_post
                done = (char_idx == 1)
                rs_seq = [seq[0] + [char_idx],score, hidden_state, context,done]
                all_seqs.append(rs_seq)
        
        ####sort all seqs and keep N=beam width seqs 
        all_seqs = sorted(all_seqs,key=lambda x:x[1],reverse=True)
        topk_seqs = all_seqs[:self.beam_width]
        all_done = self.check_all_done(topk_seqs)
        
        return topk_seqs,all_done


    def beam_search(self,data,label=None):
        if self.use_gpu:
            data=data.cuda()
        listener_feature, hid_state0 = self.listener(data)
        ###init top_seqs, each seq contain [[unit idx], seq_score, hid_state, context, is_done]
        top_seqs =[ [[0],1.0,hid_state0, listener_feature[:,0:1,:],False] ]
        
        for step in range(self.max_decode_step):
            top_seqs, all_done = self.beam_search_step(listener_feature, top_seqs)
            if all_done:
                break
        
        return [seq[0:2] for seq in top_seqs]

