import torch
if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device
from torch.autograd import Variable    
import torch.nn as nn
import torch.nn.functional as F

from util.functions import TimeDistributed,CreateOnehotVariable
import numpy as np

# BLSTM layer for pBLSTM
# Step 1. Reduce time resolution to half
# Step 2. Run through BLSTM
# Note the input should have timestep%2 == 0
class pBLSTMLayer(nn.Module):
    def __init__(self,input_feature_dim,hidden_dim,rnn_unit='LSTM',dropout_rate=0.0):
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())

        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_unit(input_feature_dim*2,hidden_dim,1, bidirectional=True, 
                                   dropout=dropout_rate,batch_first=True)
    
    def forward(self,input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        # Reduce time resolution
        input_x = input_x.contiguous().view(batch_size,int(timestep/2),feature_dim*2)
        # Bidirectional RNN
        output,hidden = self.BLSTM(input_x)
        return output,hidden

# Listener is a pBLSTM stacking 3 layers to reduce time resolution 8 times
# Input shape should be [# of sample, timestep, features]
class Listener(nn.Module):
    def __init__(self, input_feature_dim, listener_hidden_dim, rnn_unit, use_gpu, dropout_rate=0.0, **kwargs):
        super(Listener, self).__init__()
        # Listener RNN layer
        self.BLSTM = nn.LSTM(input_feature_dim, listener_hidden_dim, 1, bidirectional=True, dropout=dropout_rate, batch_first=True)
        self.pLSTM_layer1 = pBLSTMLayer(listener_hidden_dim*2,listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate)
        self.pLSTM_layer2 = pBLSTMLayer(listener_hidden_dim*2,listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate)
        self.pLSTM_layer3 = pBLSTMLayer(listener_hidden_dim*2,listener_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate)
        self.use_gpu = use_gpu
        if self.use_gpu:
            self = self.cuda()

    def forward(self,input_x):
        output, _ = self.BLSTM(input_x)
        output, _ = self.pLSTM_layer1(output)
        output, _ = self.pLSTM_layer2(output)
        output0,hid0 = self.pLSTM_layer3(output[:,0:2,:])
        output, hidT = self.pLSTM_layer3(output)
        return output,(Variable(torch.transpose(output[:,0:1,:].data,0,1)),Variable(torch.cat((hid0[1][0:1,:,:].data,hidT[1][1:,:,:].data),2)))


# Speller specified in the paper
class Speller(nn.Module):
    def __init__(self, output_class_dim, embed_dim, use_listener_state, speller_hidden_dim, rnn_unit, speller_rnn_layer, use_gpu, max_label_len,
                 use_mlp_in_attention, mlp_dim_in_attention, mlp_activate_in_attention, num_heads, listener_hidden_dim, **kwargs):
        super(Speller, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())
        self.max_label_len = max_label_len
        self.use_gpu = use_gpu
        self.float_type = torch.torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.label_dim = output_class_dim
        self.use_listener_state = use_listener_state
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(output_class_dim, embed_dim)
        self.rnn_layer = self.rnn_unit(embed_dim+2*listener_hidden_dim,speller_hidden_dim,num_layers=speller_rnn_layer,batch_first=True)
        self.attention = Attention( mlp_preprocess_input=use_mlp_in_attention, preprocess_mlp_dim=mlp_dim_in_attention,
                                    activate=mlp_activate_in_attention, heads=num_heads, input_feature_dim=2*listener_hidden_dim)
        self.predict_hid = nn.Linear(speller_hidden_dim*2,512)
        self.nonlinear = nn.ReLU()
        self.character_distribution = nn.Linear(512,output_class_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        if self.use_gpu:
            self = self.cuda()

    # Stepwise operation of each sequence
    def forward_step(self,input_word, last_hidden_state,listener_feature):
        rnn_output, hidden_state = self.rnn_layer(input_word,last_hidden_state)
        attention_score, context = self.attention(rnn_output,listener_feature)
        concat_feature = torch.cat([rnn_output.squeeze(dim=1),context],dim=-1)
        raw_pred = self.softmax(self.character_distribution(self.nonlinear(self.predict_hid(concat_feature))))

        return raw_pred, hidden_state, context, attention_score

    def forward(self, listener_feature, hid_state=None, ground_truth=None, label_len=None, teacher_force_rate = 0.9):
        if ground_truth is None:
            teacher_force_rate = 0
        teacher_force = True if np.random.random_sample() < teacher_force_rate else False

        batch_size = listener_feature.size()[0]

        #output_word = CreateOnehotVariable(self.float_type(np.zeros((batch_size,1))),self.label_dim)
        output_word = self.embedding(Variable(torch.LongTensor(np.zeros((batch_size,1)) )).cuda())
        rnn_input = torch.cat([output_word,listener_feature[:,0:1,:]],dim=-1)

        if self.use_listener_state:
            ##use listener hid_state of t=0 as the history of speller
            hidden_state = hid_state
        else:
            hidden_state = None

        raw_pred_seq = []
        output_seq = []
        attention_record = []
        speller_max_step = max(label_len) if label_len is not None else self.max_label_len
        for step in range(speller_max_step):
            raw_pred, hidden_state, context, attention_score = self.forward_step(rnn_input, hidden_state, listener_feature)
            raw_pred_seq.append(raw_pred)
            attention_record.append(attention_score)
            # Teacher force - use ground truth as next step's input
            if teacher_force:
                output_word_onehot = ground_truth[:,step:step+1,:].type(self.float_type)
                output_word = self.embedding(torch.max(output_word_onehot,2)[1])
            else:
                output_word_post = raw_pred.unsqueeze(1)
                output_word = self.embedding(torch.max(output_word_post,2)[1])
            rnn_input = torch.cat([output_word,context.unsqueeze(1)],dim=-1)

        return raw_pred_seq,attention_record


# Attention mechanism
# Currently only 'dot' is implemented
# please refer to http://www.aclweb.org/anthology/D15-1166 section 3.1 for more details about Attention implementation
# Input : Decoder state                      with shape [batch size, 1, decoder hidden dimension]
#         Compressed feature from Listner    with shape [batch size, T, listener feature dimension]
# Output: Attention score                    with shape [batch size, T (attention score of each time step)]
#         Context vector                     with shape [batch size,  listener feature dimension]
#         (i.e. weighted (by attention score) sum of all timesteps T's feature)
class Attention(nn.Module):
    def __init__(self, mlp_preprocess_input, preprocess_mlp_dim, activate, heads=1, mode='dot', input_feature_dim=512):
        super(Attention,self).__init__()
        self.mode = mode.lower()
        self.mlp_preprocess_input = mlp_preprocess_input
        self.relu = nn.ReLU()
        self.num_heads = heads
        self.head_dim = int(input_feature_dim/heads)
        self.softmax = nn.Softmax(dim=-1)
        if mlp_preprocess_input:
            self.preprocess_mlp_dim  = preprocess_mlp_dim
            self.phi = nn.Linear(input_feature_dim,preprocess_mlp_dim)
            self.psi = nn.Linear(input_feature_dim,preprocess_mlp_dim)
            self.activate = getattr(F,activate)
            if self.mode == 'add':
                self.w = nn.Linear(preprocess_mlp_dim,1)

    def forward(self, decoder_state, listener_feature):
        if self.mlp_preprocess_input:
            comp_decoder_state = self.relu(self.phi(decoder_state))
            comp_listener_feature = self.relu(TimeDistributed(self.psi,listener_feature))
        else:
            comp_decoder_state = decoder_state
            comp_listener_feature = listener_feature

        temp=[]
        for idx in range(self.num_heads):
            st_idx = idx*self.head_dim
            ed_idx = (idx+1)*self.head_dim
            if self.mode == 'dot':
                energy = torch.bmm(comp_decoder_state[:,:,st_idx:ed_idx],comp_listener_feature.transpose(1, 2)[:,st_idx:ed_idx,:]).squeeze(dim=1)
            if self.mode == 'add':
                pass
            else:
                # TODO: other attention implementations
                pass
            attention_score_idx = self.softmax(energy)
            context_idx = torch.sum(listener_feature[:,:,st_idx:ed_idx]*attention_score_idx.unsqueeze(2).repeat(1,1,self.head_dim),dim=1)
            temp.append(context_idx)

        context = torch.cat(temp,1)
        return attention_score_idx,context




