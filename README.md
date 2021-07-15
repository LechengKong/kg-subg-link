# kg-subg-link

Code repo for subgraph link prediction for knowledge graph

To run the code, download WN18RR dataset and put them in a folder of name "WN18RR" under a directory ./data

Modify train_benchmark.py: make Mem.root_path your ./data directory, make Mem.data_path your desired path to save and load models. Some parameters are irrelevant, but might be useful in the future.

Type
python train_benchmark.py

You can start training on the dataset specified

train.py is a file for another project, ignore it for now.
