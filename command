python train_graph.py --data_set citeseer --ind_data_set citeseer --reptition 1 --num_workers 8 --transductive True --homogeneous True --eval_rep 1 --rel_emb_dim 16 --emb_dim 16 --attn_rel_emb_dim 16 --num_gcn_layers 3 --dropout 0.7 --edge_dropout 0 --l2 0.001

python train_graph.py --transductive True --homogeneous True --data_set celegan --reptition 1 --eval_rep 1 --ind_data_set celegan --edge_dropout 0 --l2 0.00001 --lr 0.005 --num_gcn_layers 3 --batch_size 64 --eval True

python train_graph.py --transductive True --homogeneous True --data_set pb --reptition 1 --eval_rep 1 --ind_data_set pb --edge_dropout 0 --l2 0.000001 --lr 0.001 --num_gcn_layers 5 --batch_size 128 --use_deep_set True