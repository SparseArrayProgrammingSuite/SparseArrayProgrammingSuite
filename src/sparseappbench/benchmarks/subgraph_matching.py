"""
Name: Subgraph Matching
Author: Taishan Chen
Email: utallow@bu.edu

Motivation:
Subgraph matching is a classic problem and widely used in query evaluations in databases. 

Role of sparsity:
The graph structures from databases are usually much sparse, 
and naive approaches with adjacent matrices suffers from O(V^2) space requirements.

Implementation:
This code is hand written by me with help and discussion with @kylebd99.

Dataset:
The G-CARE dataset is provided by https://github.com/yspark-dblab/gcare.
This code will download the data files from the Google Drive links they provide.

Generative AI:
No generative AI was used to implement benchmark functions.
"""

import gdown
from pathlib import Path
import os
import tarfile
import numpy as np
from sparseappbench.binsparse_format import BinsparseFormat


def read_gcare_query(p: Path, continous_label=True):
    with p.open('r', encoding='utf-8') as f:
        exprs = []
        qvs = []
        sp_mats_name = set()
        for line in f.readlines():
            if line.startswith('t'):
                query_id = int(line.strip().split(' ')[-1])
            elif line.startswith('v'):
                vals = line.strip().split(' ')
                qv_id = int(vals[1])
                v_label = int(vals[2])
                v_id = int(vals[3])
                qvs.append(f'v_{qv_id}')
                if v_label == -1:
                    # Since all vertices have at least one label then it's unnecessary
                    # exprs.append(f'VA[v_{qv_id}]')
                    # sp_mats_name.add(f'VA')
                    pass
                else:
                    # Q[qv_id] must have label of v_label
                    exprs.append(f'V{v_label}[v_{qv_id}]')
                    sp_mats_name.add(f'V{v_label}')

                if v_id == -1:
                    # C should be a vector that for all v_id existed, C[v_id] = 1 and all else 0
                    # i.e C = union of all U_{v_id}
                    # If v_id is continous: [0, 1, ..., max_vid] then it's unnecessary
                    if not continous_label:
                        exprs.append(f'C[v_{qv_id}]')
                        sp_mats_name.add(f'C')
                else:
                    # QV[qv_id] must have index of v_id
                    exprs.append(f'P{v_id}[v_{qv_id}]')
                    sp_mats_name.add(f'P{v_id}')
                    # P_{v_id} is a unit (one-hot vector) where P[v_id] = 1
            elif line.startswith('e'):
                vals = line.strip().split(' ')
                qv_id1 = int(vals[1])
                qv_id2 = int(vals[2])
                e_label = int(vals[3])
                exprs.append(f'E{e_label}[v_{qv_id1},v_{qv_id2}]')
                sp_mats_name.add(f'E{e_label}')

        final_expr = 'S[] += ' + ' * '.join(exprs)
        return final_expr, qvs, sp_mats_name


def read_gcare_data(p: Path):
    with p.open('r', encoding='utf-8') as f:
        max_vid = 0
        num_nodes = 0
        num_edges = 0
        all_verts = []

        V_dict = dict()
        E_dict = dict()

        for line in f.readlines():
            if line.startswith('t'):
                data_id = int(line.strip().split(' ')[-1])
            elif line.startswith('v'):
                vals = line.strip().split(' ')
                num_nodes += 1

                # Read vertex id
                v_id = int(vals[1])
                if v_id > max_vid:
                    max_vid = v_id
                all_verts.append(v_id)

                # Read vertex labels
                v_labels = [int(x) for x in vals[2:]]
                if len(v_labels) == 0:
                    v_labels = [0]

                # Update V_l
                for label in v_labels:
                    if label in V_dict:
                        V_dict[label].append(v_id)
                    else:
                        V_dict[label] = [v_id]
            elif line.startswith('e'):
                vals = line.strip().split(' ')
                num_edges += 1

                # Read edge start and end point
                v_id1 = int(vals[1])
                v_id2 = int(vals[2])

                # Read edge labels
                e_labels = [int(x) for x in vals[3:]]
                if len(e_labels) == 0:
                    e_labels = [0]

                # Update E_l
                for label in e_labels:
                    if label in E_dict:
                        E_dict[label][0].append(v_id1)
                        E_dict[label][1].append(v_id2)
                    else:
                        E_dict[label] = ([v_id1], [v_id2])
            else:
                pass

        # V[label] is a vector of all vertices with this label
        V = dict()
        for (label, verts) in V_dict.items():
            V_l = {}
            V_l['V'] = np.ones((len(verts), ), dtype=np.int64)
            V_l['I_tuple'] = (np.array(verts), )
            V_l['shape'] = (max_vid+1, )
            # V_l = sp.coo_array((np.ones((len(verts), ), dtype=np.int64), (verts, )), shape=(max_vid+1, ))
            V[f'V{label}'] = V_l

        # E[label] is a sparse adjacency matrix of all edges with this label
        E = dict()
        for (label, edges) in E_dict.items():
            assert len(edges[0]) == len(edges[1])
            l_num_edges = len(edges[0])
            E_l = {}
            E_l['V'] = np.ones((l_num_edges,), dtype=np.int64)
            E_l['I_tuple'] = (np.array(edges[0]), np.array(edges[1]))
            E_l['shape'] = (max_vid+1, max_vid+1)
            # E_l = sp.coo_array((np.ones((l_num_edges,), dtype=np.int64), (edges[0], edges[1])),
            #                     shape=(max_vid+1, max_vid+1))
            E[f'E{label}'] = E_l

        sp_mats = V | E

        if max_vid + 1 == num_nodes:
            continous_label = True
        else:
            continous_label = False
            C = {}
            C['V'] = np.ones((num_nodes, ), dtype=np.int64)
            C['I_tuple'] = (all_verts, )
            C['shape'] = (max_vid+1, )
            sp_mats['C'] = C

        return max_vid, continous_label, sp_mats


def run_one_match(xp, sp_mats: dict, expr: str):
    for key, val in sp_mats.items():
        sp_mats[key] = xp.from_benchmark(val)
    count = xp.einsum(expr, **sp_mats)
    return xp.to_benchmark(np.array(count))


def benchmark_subgraph_matching(xp, queries):
    counts = np.zeros((len(queries), ), dtype=np.int64)
    for i, (sp_mats, expr) in enumerate(queries):
        for key, val in sp_mats.items():
            sp_mats[key] = xp.from_benchmark(val)
        counts[i] = xp.einsum(expr, **sp_mats)
    return xp.to_benchmark(counts)


def download_gcare_data():
    root_dir = Path('./data/gcare')
    dataset_dir = root_dir / 'dataset'
    queryset_dir = root_dir / 'queryset'
    ground_truth_dir = root_dir / 'ground_truth'

    dataset_link = 'https://drive.google.com/file/d/1HAgSVE-24NOap6_Q1_twH56Dkb2kPvGU/view?usp=sharing'
    queryset_link = 'https://drive.google.com/file/d/1Dlj43rBAOVPAsfzKlYxIbZ9RsqeGM_MN/view?usp=sharing'
    grount_truth_link = 'https://drive.google.com/file/d/1Bc6Q2RZQTcIB8IfOw5KafNYwPhq2BO94/view?usp=sharing'

    # Download dataset
    if not dataset_dir.exists():
        os.makedirs(dataset_dir, exist_ok=True)

    gdown.cached_download(dataset_link,
                          str(dataset_dir / 'dataset.tar.gz'),
                          hash='sha256:78B86CDA06115C4554CDFCFB93A7FBC8ECB759DF39927510DD02CED4228A95E4'.lower(),
                          fuzzy=True)

    with tarfile.open(dataset_dir / 'dataset.tar.gz', 'r:gz') as tar:
        tar.extractall(path=dataset_dir, filter='data')

    # Download queryset
    if not queryset_dir.exists():
        os.makedirs(dataset_dir, exist_ok=True)

    gdown.cached_download(queryset_link,
                          str(queryset_dir / 'queryset.tar.gz'),
                          hash='sha256:C8DC9F978296559E9E55335A989CE16E7B5BCBA7AA9D43E25FBD9E588D00EBC7'.lower(),
                          fuzzy=True)

    with tarfile.open(queryset_dir / 'queryset.tar.gz', 'r:gz') as tar:
        tar.extractall(path=queryset_dir, filter='data')

    # Download true cardinality
    if not ground_truth_dir.exists():
        os.makedirs(ground_truth_dir, exist_ok=True)

    gdown.cached_download(grount_truth_link,
                          str(ground_truth_dir / 'ground_truth.tar.gz'),
                          hash='sha256:22E59F4FC06FFB79711D582513C6422CA555422C9947FDE64A52F0A9292D382C'.lower(),
                          fuzzy=True)

    with tarfile.open(ground_truth_dir / 'ground_truth.tar.gz', 'r:gz') as tar:
        tar.extractall(path=ground_truth_dir, filter='data')

    return (dataset_dir, queryset_dir)


def process_one_query(query_path, all_sp_mats, max_vid, continous_label):
    (expr, qvs, sp_mats_name) = read_gcare_query(
        query_path, continous_label=continous_label)

    sp_mats_needed = dict()
    for sp_name in sp_mats_name:
        if sp_name not in all_sp_mats:
            if sp_name.startswith('P'):  # Node id
                new_P = {}
                new_P['V'] = np.array([1])
                new_P['I_tuple'] = (
                    np.array([0]), np.array([int(sp_name[1:])]))
                new_P['shape'] = (max_vid+1, )
                sp_mats_needed[sp_name] = BinsparseFormat.from_coo(
                    new_P['I_tuple'], new_P['V'], new_P['shape'])
            else:
                # Some queried node / edge labels do not existed in the data graph. The output must be 0.
                # In reality we can just skip it, but for benchmark let's create an all zero matrix
                if sp_name.startswith('V'):
                    zero_V = {}
                    # for compatiblity with Binsparse
                    zero_V['V'] = np.array([0])
                    zero_V['I_tuple'] = (np.array([0]), )
                    zero_V['shape'] = (max_vid+1, )
                    sp_mats_needed[sp_name] = BinsparseFormat.from_coo(
                        zero_V['I_tuple'], zero_V['V'], zero_V['shape'])
                elif sp_name.startswith('E'):
                    zero_E = {}
                    zero_E['V'] = np.array([0])
                    zero_E['I_tuple'] = (np.array([0]), np.array([0]))
                    zero_E['shape'] = (max_vid+1, max_vid+1)
                    sp_mats_needed[sp_name] = BinsparseFormat.from_coo(
                        zero_E['I_tuple'], zero_E['V'], zero_E['shape'])
        else:
            sp_mat = all_sp_mats[sp_name]
            sp_mats_needed[sp_name] = BinsparseFormat.from_coo(
                sp_mat['I_tuple'], sp_mat['V'], sp_mat['shape'])

    return (sp_mats_needed, expr)


def process_queries(queryset_dir, all_sp_mats, max_vid, continous_label):
    ret_val = []
    for query_path in queryset_dir.rglob('*.txt'):
        ret_val.append(process_one_query(
            query_path, all_sp_mats, max_vid, continous_label))

    return ret_val


def gcare_human_all():
    (dataset_dir, queryset_dir) = download_gcare_data()
    (max_vid, continous_label, all_sp_mats) = read_gcare_data(
        dataset_dir / 'human' / 'human.txt')
    return process_queries(queryset_dir / 'human', all_sp_mats, max_vid, continous_label)


def gcare_human_one(query_name: str):
    (dataset_dir, queryset_dir) = download_gcare_data()
    (max_vid, continous_label, all_sp_mats) = read_gcare_data(
        dataset_dir / 'human' / 'human.txt')
    
    query_path = queryset_dir / query_name
    if not query_path.exists():
        raise Exception(f'Query {query_name} does not exist! Path {query_path}')

    return process_one_query(query_path, all_sp_mats, max_vid, continous_label)


# Commented because current framework supported cannot handle this large

# def gcare_aids():
#     (dataset_dir, queryset_dir) = download_gcare_data()
#     (max_vid, continous_label, all_sp_mats) = read_gcare_data(
#         dataset_dir / 'aids' / 'aids.txt')
#     return process_queries(queryset_dir / 'aids', all_sp_mats, max_vid, continous_label)
