import numpy as np
from scipy.sparse import csr_matrix

def process_files(files, relation2id=None):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    if relation2id is None:
        relation2id = {}

    converted_triplets = {}
    rel_list = [[] for i in range(len(relation2id))]

    ent = 0
    rel = len(relation2id)

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1
                rel_list.append([])

            data.append([entity2id[triplet[0]], relation2id[triplet[1]], entity2id[triplet[2]]])
        
        if file_type == "train":
            for trip in data:
                rel_list[trip[1]].append([trip[0], trip[2]])

        converted_triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    adj_list = []
    for rel_mat in rel_list:
        rel_array = np.array(rel_mat)
        if len(rel_array)==0:
            adj_list.append(csr_matrix((len(entity2id),len(entity2id))))
        else:
            adj_list.append(csr_matrix((np.ones(len(rel_mat)),(rel_array[:,0],rel_array[:,1])), shape=(len(entity2id),len(entity2id))))

    return adj_list, converted_triplets, entity2id, relation2id, id2entity, id2relation
