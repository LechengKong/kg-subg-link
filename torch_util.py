import torch
import dgl
import numpy as np
import torch
import random
import time


def collate_dgl(samples):
    # The input `samples` is a list of pairs
    graphs_pos, g_labels_pos, r_labels_pos, graphs_negs, g_labels_negs, r_labels_negs = map(list, zip(*samples))
    batched_graph_pos = dgl.batch(graphs_pos)

    graphs_neg = [item for sublist in graphs_negs for item in sublist]
    g_labels_neg = [item for sublist in g_labels_negs for item in sublist]
    r_labels_neg = [item for sublist in r_labels_negs for item in sublist]

    batched_graph_neg = dgl.batch(graphs_neg)
    return (batched_graph_pos, r_labels_pos), g_labels_pos, (batched_graph_neg, r_labels_neg), g_labels_neg


def collate_dgl_val(samples):
    graphs, rel, t_label = map(list, zip(*samples))
    graphs = [item for sublist in graphs for item in sublist]
    rels = [item for sublist in rel for item in sublist]
    batched_graph_pos = dgl.batch(graphs)
    return (batched_graph_pos, rels), t_label



def move_batch_to_device_dgl(batch, device):
    ((g_dgl_pos, r_labels_pos), targets_pos, (g_dgl_neg, r_labels_neg), targets_neg) = batch

    targets_pos = torch.LongTensor(targets_pos).to(device=device)
    r_labels_pos = torch.LongTensor(r_labels_pos).to(device=device)

    targets_neg = torch.LongTensor(targets_neg).to(device=device)
    r_labels_neg = torch.LongTensor(r_labels_neg).to(device=device)

    g_dgl_pos = g_dgl_pos.to(device)
    g_dgl_neg = g_dgl_neg.to(device)

    return ((g_dgl_pos, r_labels_pos), targets_pos, (g_dgl_neg, r_labels_neg), targets_neg)


def move_batch_to_device_dgl_val(batch, device):
    (g_dgl_pos, r_labels_pos), targets_pos = batch

    targets_pos = torch.LongTensor(targets_pos).to(device=device)
    r_labels_pos = torch.LongTensor(r_labels_pos).to(device=device)

    g_dgl_pos = g_dgl_pos.to(device)

    return (g_dgl_pos, r_labels_pos), targets_pos


def collate_dgl_spe(samples):
    # The input `samples` is a list of pairs
    graphs_pos, g_labels_pos, r_labels_pos, spe_pos, graphs_negs, g_labels_negs, r_labels_negs, spe_negs = map(list, zip(*samples))
    batched_graph_pos = dgl.batch(graphs_pos)

    graphs_neg = [item for sublist in graphs_negs for item in sublist]
    g_labels_neg = [item for sublist in g_labels_negs for item in sublist]
    r_labels_neg = [item for sublist in r_labels_negs for item in sublist]
    spe_neg = [item for sublist in spe_negs for item in sublist]

    batched_graph_neg = dgl.batch(graphs_neg)
    return (batched_graph_pos, r_labels_pos, spe_pos), g_labels_pos, (batched_graph_neg, r_labels_neg, spe_neg), g_labels_neg


def collate_dgl_val_spe(samples):
    graphs, rel, t_label, spes = map(list, zip(*samples))
    graphs = [item for sublist in graphs for item in sublist]
    rels = [item for sublist in rel for item in sublist]
    spe = [item for sublist in spes for item in sublist]
    batched_graph_pos = dgl.batch(graphs)
    return (batched_graph_pos, rels, spe), t_label



def move_batch_to_device_dgl_spe(batch, device):
    ((g_dgl_pos, r_labels_pos, spe_pos), targets_pos, (g_dgl_neg, r_labels_neg, spe_neg), targets_neg) = batch

    targets_pos = torch.LongTensor(targets_pos).to(device=device)
    r_labels_pos = torch.LongTensor(r_labels_pos).to(device=device)
    spe_pos = torch.FloatTensor(spe_pos).to(device=device)

    targets_neg = torch.LongTensor(targets_neg).to(device=device)
    r_labels_neg = torch.LongTensor(r_labels_neg).to(device=device)
    spe_neg = torch.FloatTensor(spe_neg).to(device=device)

    g_dgl_pos = g_dgl_pos.to(device)
    g_dgl_neg = g_dgl_neg.to(device)

    return ((g_dgl_pos, r_labels_pos, spe_pos), targets_pos, (g_dgl_neg, r_labels_neg, spe_neg), targets_neg)


def move_batch_to_device_dgl_val_spe(batch, device):
    (g_dgl_pos, r_labels_pos, spe_neg), targets_pos = batch

    targets_pos = torch.LongTensor(targets_pos).to(device=device)
    r_labels_pos = torch.LongTensor(r_labels_pos).to(device=device)
    spe_neg = torch.FloatTensor(spe_neg).to(device=device)

    g_dgl_pos = g_dgl_pos.to(device)

    return (g_dgl_pos, r_labels_pos, spe_neg), targets_pos


def collate_dgl_full(samples):
    # The input `samples` is a list of pairs
    graphs, links, dist = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    ct = 0
    lks = []
    for i , sublist in enumerate(links):
        for item in sublist:
            lks.append([item[0]+ct, item[1], item[2]+ct])
        ct+=graphs[i].num_nodes()
    dists = [item for sublist in dist for item in sublist]
    return batched_graph, lks, dists


def collate_dgl_onlylink(samples):
    # The input `samples` is a list of pairs
    d = map(list, zip(*samples))
    ls = []
    for d1 in d:
        ls.append([item for sublist in d1 for item in sublist])
    b_l = [len(l) for l in d1]
    ls.append(b_l)

    return ls


def move_batch_to_device_dgl_full(batch, device):
    g, links, dists = batch

    g_device = g.to(device)
    links_device = torch.LongTensor(links).to(device=device)
    dists_device = torch.LongTensor(dists).to(device=device)

    return (g_device, (links_device, dists_device))


def move_batch_to_device_dgl_onlylink(batch,device):
    d = []
    # links, dists, mid_inds= batch
    for g in batch:
        d.append(torch.LongTensor(g).to(device=device))
    # links_device = torch.LongTensor(links).to(device=device)
    # dists_device = torch.LongTensor(dists).to(device=device)

    # return (links_device, dists_device)
    return d


def send_graph_to_device(g, device):
    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device)

    # edges
    labels = g.edge_attr_schemes()
    for l in labels.keys():
        g.edata[l] = g.edata.pop(l).to(device)
    return g


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_data_passing(params):
    if params.use_spe:
        params.collate_fn = collate_dgl_spe
        params.collate_fn_val = collate_dgl_val_spe
        params.move_batch_to_device = move_batch_to_device_dgl_spe
        params.move_batch_to_device_val = move_batch_to_device_dgl_val_spe
    elif params.whole_graph:
        params.collate_fn = collate_dgl_full
        params.move_batch_to_device = move_batch_to_device_dgl_full
    elif params.only_link_sample:
        params.collate_fn = collate_dgl_onlylink
        params.move_batch_to_device = move_batch_to_device_dgl_onlylink
    else:
        params.collate_fn = collate_dgl
        params.collate_fn_val = collate_dgl_val
        params.move_batch_to_device = move_batch_to_device_dgl
        params.move_batch_to_device_val = move_batch_to_device_dgl_val