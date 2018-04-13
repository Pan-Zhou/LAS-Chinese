# Copyright 2014    Yajie Miao    Carnegie Mellon University
# modified by Pan Zhou
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import gzip
import os
import sys,re
import time
import glob
import struct
import numpy


from util import smart_open, preprocess_feature_and_label, shuffle_feature_and_label, make_context, skip_frame
# Classes to read and write Kaldi features. They are used when PDNN passes Kaldi features
# through trained models and saves network activation into Kaldi features. Currently we
# are using them during decoding of convolutional networks.

# Class to read Kaldi features. Each time, it reads one line of the .scp file
# and reads in the corresponding features into a numpy matrix. It only supports
# binary-formatted .ark files. Text and compressed .ark files are not supported.
class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1,"UNK": 2}
        self.word2count = {}
        #self.sent2words = {}
        self.SOS_token = 0
        self.EOS_token = 1
        self.index2word = {0: "SOS", 1: "EOS",2: "UNK"}
        self.n_words = 3  # Count SOS and EOS
        self.readLang(self.name)

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
    def readLang(self, name):
        print("Reading languages from %s " % name)
        lines = open(name).read().strip().split('\n')
        for l in lines:
            ll=l.strip(' ').split(' ')
            if len(ll)>1:
                sentence = ' '.join(ll[1:])
            else:
                sentence = ll[0]
            self.addSentence(sentence)

        print("Read %s sentences" % len(lines))
        print("Counting %d characters " % self.n_words)

# A transfer function for LAS label
# We need transfer label and make them onehot encoded
# each sequence should end with an <eos> (index = 1)
# Input y: list of np array with shape ()
# Output tuple: (indices, values, shape)
def OneHotEncode(Y,max_len,max_idx=100):
    new_y = numpy.zeros((len(Y),max_len,max_idx))
    for idx,label_seq in enumerate(Y):
        last_value = -1
        cnt = 0
        for label in label_seq:
            #if last_value != label:
            new_y[idx,cnt,label] = 1.0
            cnt += 1
            #last_value = label
        #new_y[idx,cnt,1] = 1.0 # <eos>
    return new_y

class KaldiReadIn(object):
    def __init__(self, language, scp_path, lab_path, max_timestep, max_label_len, batch_size, left_cxt, right_cxt, kaldi_feat_dim, n_skip_frame, n_downsample, **kwargs):
        self.language=language
        self.scp_path = scp_path
        self.scp_file_read = smart_open(self.scp_path,"r")
        self.lab_path = lab_path
        self.sent2words={}
        self.max_timestep = max_timestep
        self.max_lab_len = max_label_len
        self.batch_size = batch_size
        self.left_cxt = left_cxt
        self.right_cxt = right_cxt
        self.feat_dim = kaldi_feat_dim
        self.n_skip_frame = n_skip_frame
        self.n_downsample = n_downsample
        self.read_label()

    def read_next_utt(self):
        next_scp_line = self.scp_file_read.readline()
        if next_scp_line == '' or next_scp_line == None:
            return '', None
        utt_id, path_pos = next_scp_line.replace('\n','').split(' ')
        path, pos = path_pos.split(':')

        ark_read_buffer = smart_open(path, 'rb')
        ark_read_buffer.seek(int(pos),0)
        header = struct.unpack('<xcccc', ark_read_buffer.read(5))
        #if header[0] != "B":
        #    print("Input .ark file is not binary"); exit(1)

        rows = 0; cols= 0
        m, rows = struct.unpack('<bi', ark_read_buffer.read(5))
        n, cols = struct.unpack('<bi', ark_read_buffer.read(5))

        tmp_mat = numpy.frombuffer(ark_read_buffer.read(rows * cols * 4), dtype=numpy.float32)
        utt_mat = numpy.reshape(tmp_mat, (rows, cols))

        ark_read_buffer.close()

        return utt_id, utt_mat

    def read_label(self):
        print("Reading lines from %s " % self.lab_path)
        lines = open(self.lab_path).read().strip().split('\n')
        for l in lines:
            ll=l.strip(' ').split(' ')
            sentence = ' '.join(ll[1:])
            if ll[0] not in self.sent2words:
                self.sent2words[ll[0]] = sentence
        print("Read %d lines from %s " %(len(self.sent2words), self.lab_path))


    def indexesFromSentence(self,sentname):
        indexes = [self.language.word2index[word] if word in self.language.word2index 
                else self.language.word2index['UNK'] for word in self.sent2words[sentname].split(' ')]
        indexes.append(self.language.word2index['EOS'])
        return indexes
    
    
    # load num_streams features and labels
    def load_next_nstreams(self):
        length = []
        feat_mat = []
        lab_len =[]
        label = []
        nstreams = 0
        max_frame_num = 0
        max_lab_len = 0
        utt_lists =[]
        while True:
            utt_id,utt_mat = self.read_next_utt()
            if utt_mat is None:
                self.scp_file_read.seek(0,0)
                break;
            if utt_id not in self.sent2words:
                continue
            if len(utt_mat)>self.max_timestep:
                continue
            lab_index = self.indexesFromSentence(utt_id)
            if len(lab_index)>self.max_lab_len:
                continue

            if True: #(len(ali_utt) * 2 + 1) < len(utt_mat):
                label.append(lab_index)
                utt_lists.append(utt_id)
                lab_len.append(len(lab_index))
                '''if self.read_opts['lcxt'] != 0 or self.read_opts['rcxt'] != 0:
                    feat_mat.append(make_context(utt_mat, self.read_opts['lcxt'], self.read_opts['rcxt']))
                else:
                    feat_mat.append(utt_mat)'''
                feat_mat.append(skip_frame(make_context(utt_mat, self.left_cxt, self.right_cxt),self.n_skip_frame))
                length.append(len(feat_mat[nstreams]))
            else:
                continue
            if max_frame_num < length[nstreams]:
                max_frame_num = length[nstreams]
            if max_lab_len < lab_len[nstreams]:
                max_lab_len =lab_len[nstreams]
            nstreams += 1


            if nstreams == self.batch_size:
                res=max_frame_num % self.n_downsample
                max_frame_num = (max_frame_num+self.n_downsample-res if res else max_frame_num)
            #sort in decrease order
            #feat_mat.sort(key=lambda x:x.size,reverse=True)
            #length.sort(reverse=True)
            #label.sort(key=lambda x:x.size,reverse=True)
            # zero fill
                i = 0
                while i < nstreams:
                    if max_frame_num != length[i]:
                        feat_mat[i] = numpy.vstack((feat_mat[i], numpy.zeros((max_frame_num-length[i], feat_mat[i].shape[1]),dtype=numpy.float32)))
                    label[i] = numpy.hstack((label[i], numpy.zeros((max_lab_len-len(label[i])),dtype=numpy.int32)) )
                    i += 1

                if feat_mat.__len__():
                    label_nstream = numpy.vstack(label)
                    feat_mat_nstream = numpy.vstack(feat_mat).reshape(nstreams, -1, self.feat_dim)
                    np_length = numpy.vstack(length).reshape(-1)
                    yield feat_mat_nstream,label_nstream,np_length,lab_len,utt_lists
                    length,feat_mat,label,lab_len,utt_lists = [],[],[],[],[]
                    nstreams = 0
                    max_frame_num =0
                    max_lab_len =0
                    #return feat_mat_nstream , label_nstream , np_length
                else:
                    break
                    #return None,None,None

    def next(self):
        return self.__iter__()

    def __iter__(self):
        return self.load_next_nstreams()

    def __call__(self):
        return self.__iter__()

# Class to write numpy matrix into Kaldi .ark file. It only supports binary-formatted .ark files.
# Text and compressed .ark files are not supported.
class KaldiWriteOut(object):

    def __init__(self, ark_path):

        self.ark_path = ark_path
        self.ark_file_write = smart_open(ark_path,"wb")

    def write_kaldi_mat(self, utt_id, utt_mat):
        utt_mat = numpy.asarray(utt_mat, dtype=numpy.float32)
        rows, cols = utt_mat.shape
        self.ark_file_write.write(struct.pack('<%ds'%(len(utt_id)), utt_id))
        self.ark_file_write.write(struct.pack('<cxcccc', ' ','B','F','M',' '))
        self.ark_file_write.write(struct.pack('<bi', 4, rows))
        self.ark_file_write.write(struct.pack('<bi', 4, cols))
        self.ark_file_write.write(utt_mat)

    def close(self):
        self.ark_file_write.close()
