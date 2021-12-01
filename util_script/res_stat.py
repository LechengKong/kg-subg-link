import numpy as np

h10_name = 'ark.npy'

dist_name = 'dist.npy'

max_dist = 50

h10 = np.load(h10_name)
h10 = h10<=10
nh10 = np.logical_not(h10)
dist = np.load(dist_name)
dist = dist[:int(len(dist)/10)]
c_sample = len(h10)

num_max_dist = np.sum(dist==max_dist)
h10_c = np.sum(h10)
nh10_c = np.sum(nh10)

h10_max_dist = h10*(dist==max_dist)
nh10_max_dist = nh10*(dist==max_dist)

print('num max dist', num_max_dist, num_max_dist/c_sample)
print('num true max dist', np.sum(h10_max_dist), np.sum(h10_max_dist)/num_max_dist)
print('num false non max', 1-np.sum(nh10_max_dist)/nh10_c,(1-np.sum(nh10_max_dist)/nh10_c)* nh10_c/c_sample)
print('num false max dist', np.sum(nh10_max_dist), np.sum(nh10_max_dist)/num_max_dist)

print('dist true count', np.unique(h10*dist,return_counts=True)[0])
print('dist true count', np.unique(h10*dist,return_counts=True)[1])
print('dist false count', np.unique(nh10*dist,return_counts=True)[0])
print('dist false count', np.unique(nh10*dist,return_counts=True)[1])
