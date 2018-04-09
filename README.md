# End2End LAS -a Pytorch Implementation to train over Chinese characters


## Description

This is a pytorch implementation of attention based end to end model, LAS, for speech recognition.


## Requirements

##### Execution Environment

- python 3
- GPU computing is recommanded for training efficiency


##### Packages for running LAS model

- [pytorch](http://pytorch.org/) (0.3.0 or later version)

    Please use pytorch after version 0.3.0.


- [editdistance](https://github.com/aflc/editdistance)

    Package for calculating edit distance (Levenshtein distance).

    

## Setup
- LAS Model
        
        mkdir -p checkpoint
        mkdir -p log
        python3 run_exp.py <config file path>
    
    Training log will be stored at `log/` while model checkpoint at ` checkpoint/`
    
    For a customized experiment, please read and modify [`config/las_bank_config.yaml`](config/las_fbank_config.yaml)

## ToDo
- Add attention mechanism
- Beam search decoding
- Rescore or decode with Language model 
- Chinese initial finals modeling units
- Attention Visualize
-

## References
- [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211v2) (LAS)  published in ICASSP 2016.
- 
