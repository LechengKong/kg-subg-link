import os
import os.path as osp
import numpy as np


base_dir = '/project/tantra/jerry.kong/ogb_project/ogb-grail-mod/data'

file_path = "fb237_v1"

output_path = file_path+"_resample"

filenames = ['test.txt', 'train.txt','valid.txt']

graph_paths = [osp.join(base_dir,file_path,filename) for filename in filenames]

arr = []

for g in graph_paths:
    f = open(g, 'r')
    for x in f:
        arr.append(x)

all_edges = np.array(arr)

edge_c = len(all_edges)
perm = np.random.permutation(edge_c)

test = 0
valid = int(edge_c/10)*0

test_edge = all_edges[perm[:test]]
valid_edge = all_edges[perm[test:valid]]
train_edge = all_edges[perm[valid:]]

if not osp.isdir(osp.join(base_dir, output_path)):
    os.mkdir(osp.join(base_dir, output_path))

valid_file = osp.join(base_dir,output_path,'valid.txt')
f = open(valid_file,'w')
for s in valid_edge:
    f.write(s)

test_file = osp.join(base_dir,output_path,'test.txt')
f = open(test_file,'w')
for s in test_edge:
    f.write(s)


train_file = osp.join(base_dir,output_path,'train.txt')
f = open(train_file,'w')
for s in train_edge:
    f.write(s)