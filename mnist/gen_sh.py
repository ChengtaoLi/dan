
methods = ['dan_s']
script = 'main.py'
script_id = 1
dataset = 'mnist'
checkpoint_dir = 'ckpt_' + str(script_id)
train_epoch = 50
num_rep = 5

log_path = 'log_' + dataset + '_'
eval_path = 'eval_' + dataset + '_'

fid = open('run_' + dataset + '_' + str(script_id) + '.sh', 'w')

fid.write('#!/bin/bash\n')
fid.write('export CUDA_VISIBLE_DEVICES=' + str(script_id) + '\n')

for rep in xrange(num_rep):
    for i in xrange(len(methods)):
        fid.write('python ' + script + \
                  ' --flag_train True' + \
                  ' --dataset ' + dataset + \
                  ' --ckpt_dir ' + checkpoint_dir + '_' + str(rep) + \
                  ' --model_mode ' + methods[i] + \
                  ' --savepath ' + eval_path + methods[i] + '_' + str(script_id) + '_' + str(rep) + '.pkl' + \
                  ' > ' + log_path + methods[i] + '_' + str(script_id) + '_' + str(rep) + '.txt\n')

