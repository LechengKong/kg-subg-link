import numpy as np
import scipy.sparse as ssp
import torch
import dgl


def sample_neg_link(adj, head, tail, num_nodes, sample_size):
    arr = np.arange(num_nodes)
    neg_head_neighbor = adj.col[adj.row==head]
    neg_tail_neighbor = adj.row[adj.col==tail]
    cans = set(arr)
    tail_cans = cans.difference(neg_head_neighbor)
    head_cans = cans.difference(neg_tail_neighbor)
    tail_can_arr = np.array(list(tail_cans))
    head_can_arr = np.array(list(head_cans))
    # print(type(tail_can_arr))
    # print(tail_cans)
    tail_sample = np.random.choice(tail_can_arr, sample_size, replace = False)
    head_sample = np.random.choice(head_can_arr, sample_size, replace = False)
    return tail_sample, head_sample


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[:, idxs_wo_nodes][idxs_wo_nodes, :]


def construct_graph_from_edges(edges, n_entities):
    g = dgl.graph((edges[0], edges[2]), num_nodes=n_entities)
    g.edata['type'] = torch.tensor(edges[1], dtype=torch.int32)
    return g


def extract_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None, median_mult=2, inc_size=0.1):
    cur_nodes = roots
    visited = set()
    in_hop_neighbor = []
    # st = time.time()
    sim_nodes=np.zeros(adj.shape[0])
    visited.update(cur_nodes)
    for i in range(h):
        # print("sparse hop",i)
        # st = time.time()
        neighb = []
        small_nodes = np.array(list(cur_nodes))
        if len(small_nodes)==0:
            break
        # print("candidate", time.time()-st)
        neighbor_count = adj.indptr[small_nodes+1] - adj.indptr[small_nodes]
        neighbor_count_median = np.median(neighbor_count)
        # print("median", time.time()-st)
        for j, cur in enumerate(small_nodes):
            if i>0 and neighbor_count[j]>neighbor_count_median*median_mult:
                continue
            neighbors = adj.indices[adj.indptr[cur]: adj.indptr[cur+1]]
            n_set = sim_nodes[neighbors]
            n_num = len(n_set)
            n_same_num = np.sum(n_set)
            if i>0 and (n_same_num/n_num)<(i)*inc_size:
                continue
            neighb.append(neighbors)
        if len(neighb)==0:
            break
        # print("filter", time.time()-st)
        neighbor_nodes = np.concatenate(neighb)
        sz = len(neighbor_nodes)
        neighbor_nodes, counts = np.unique(neighbor_nodes, return_counts=True)
        sim_nodes = np.zeros(adj.shape[0])
        sim_nodes[neighbor_nodes] = 1
        # print("sim dict",time.time()-st)
        if max_nodes_per_hop and max_nodes_per_hop<len(neighbor_nodes):
            next_nodes = np.random.choice(neighbor_nodes, max_nodes_per_hop, p=counts/sz)
            next_nodes = set(next_nodes)
        else:
            next_nodes = set(neighbor_nodes)
        next_nodes.difference_update(visited)
        visited.update(next_nodes)
        in_hop_neighbor.append(next_nodes)
        cur_nodes = next_nodes
        # print("update",time.time()-st)
    return set().union(*in_hop_neighbor)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    cur_nodes = roots
    visited = set()
    in_hop_neighbor = []
    inc_size = 0.3
    # st = time.time()
    if isinstance(adj, np.ndarray):
        visited.update(cur_nodes)
        for i in range(h):
            # print("dense hop:",i)
            # st = time.time()
            small_nodes = np.array(list(cur_nodes))
            # print("create candidiate", time.time()-st)
            if len(small_nodes)==0:
                break
            if i>0:
                neighbor_sim = np.sum(np.logical_and(adj[small_nodes], sim_dict),axis=-1)
                neighbor_count = np.sum(adj[small_nodes], axis=-1)
                neighbor_count_median = np.median(neighbor_count)
                small_nodes = small_nodes[np.logical_and(neighbor_count<neighbor_count_median*500000,neighbor_sim/neighbor_count>=inc_size*(i-1))]
                # small_nodes = small_nodes[neighbor_count<neighbor_count_median*1.5]
            if len(small_nodes)==0:
                break
            # print("filter", time.time()-st)
            neighbor_nodes = adj[small_nodes, :].nonzero()[1]
            sz = len(neighbor_nodes)
            neighbor_nodes, counts = np.unique(neighbor_nodes, return_counts=True)
            sim_dict = np.zeros(len(adj))
            sim_dict[neighbor_nodes] = 1
            # print("create dict", time.time()-st)
            if max_nodes_per_hop and max_nodes_per_hop<len(neighbor_nodes):
                next_nodes = np.random.choice(neighbor_nodes, max_nodes_per_hop, p=counts/sz)
                next_nodes = set(next_nodes)
            else:
                next_nodes = set(neighbor_nodes)
            next_nodes.difference_update(visited)
            visited.update(next_nodes)
            in_hop_neighbor.append(next_nodes)
            cur_nodes = next_nodes
            # print("update", time.time()-st)
    else:
        sim_nodes=np.zeros(adj.shape[0])
        visited.update(cur_nodes)
        for i in range(h):
            # print("sparse hop",i)
            # st = time.time()
            neighb = []
            small_nodes = np.array(list(cur_nodes))
            if len(small_nodes)==0:
                break
            # print("candidate", time.time()-st)
            neighbor_count = adj.indptr[small_nodes+1] - adj.indptr[small_nodes]
            neighbor_count_median = np.median(neighbor_count)
            # print("median", time.time()-st)
            for j, cur in enumerate(small_nodes):
                if i>0 and neighbor_count[j]>neighbor_count_median*50000:
                    continue
                neighbors = adj.indices[adj.indptr[cur]: adj.indptr[cur+1]]
                n_set = sim_nodes[neighbors]
                n_num = len(n_set)
                n_same_num = np.sum(n_set)
                if i>0 and (n_same_num/n_num)<(i-1)*inc_size:
                    continue
                neighb.append(neighbors)
            if len(neighb)==0:
                break
            # print("filter", time.time()-st)
            neighbor_nodes = np.concatenate(neighb)
            sz = len(neighbor_nodes)
            neighbor_nodes, counts = np.unique(neighbor_nodes, return_counts=True)
            sim_nodes = np.zeros(adj.shape[0])
            sim_nodes[neighbor_nodes] = 1
            # print("sim dict",time.time()-st)
            if max_nodes_per_hop and max_nodes_per_hop<len(neighbor_nodes):
                next_nodes = np.random.choice(neighbor_nodes, max_nodes_per_hop, p=counts/sz)
                next_nodes = set(next_nodes)
            else:
                next_nodes = set(neighbor_nodes)
            next_nodes.difference_update(visited)
            visited.update(next_nodes)
            in_hop_neighbor.append(next_nodes)
            cur_nodes = next_nodes
            # print("update",time.time()-st)
    return set().union(*in_hop_neighbor)


def node_label(subgraph, max_distance=1):
    # implementation of the node labeling scheme described in the paper
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)
    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes


def subgraph_extraction_labeling_wiki(ind, rel, A_incidence, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    labels, enclosing_subgraph_nodes = node_label(A_incidence[:, subgraph_nodes][subgraph_nodes, :], max_distance=h)
    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]
    # pruned_subgraph_nodes = subgraph_nodes
    # pruned_labels = labels

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)

    return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes

