from pathlib import Path
import scipy.sparse as sp
import numpy as np
import sys

from sparseappbench.benchmarks.subgraph_matching import benchmark_subgraph_matching
from sparseappbench.frameworks.numpy_framework import NumpyFramework
from sparseappbench.frameworks.sparse_framework import PyDataSparseFramework
from sparseappbench.binsparse_format import BinsparseFormat


def read_gcare_data(p: Path):
    if not p.exists():
        print('File not existed!')
        exit(1)

    with p.open('r', encoding='utf-8') as f:
        max_vid = 0
        # max_vlabel = 0
        num_nodes = 0
        num_edges = 0

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
            V_l['I_tuple'] = (verts, )
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
            E_l['I_tuple'] = (edges[0], edges[1])
            E_l['shape'] = (max_vid+1, max_vid+1)
            # E_l = sp.coo_array((np.ones((l_num_edges,), dtype=np.int64), (edges[0], edges[1])),
            #                     shape=(max_vid+1, max_vid+1))
            E[f'E{label}'] = E_l

        if max_vid + 1 == num_nodes:
            continous_label = True
        else:
            continous_label = False

        # print(f'V: {V}')
        # print(f'E: {E}')
        return max_vid, continous_label, V | E


def read_gcare_query(p: Path, continous_label=True):
    if not p.exists():
        print('File not existed!')
        exit(1)
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
                # If e_label not existed. we should put an all-zero matrix Z here..?

        final_expr = 'S[] += ' + ' * '.join(exprs)
        print(final_expr)
        print(f'QVs: {qvs}')
        print(f'Sparse matrices: {sp_mats_name}')
        return final_expr, qvs, sp_mats_name

def run_test():
    dataset_name = 'aids'
    data_path = Path.cwd() / 'gcare' / 'data' / dataset_name / f'{dataset_name}.txt'
    
    query_type = 'Graph_3'
    query_name = 'uf_Q_0_1.txt'
    query_path = Path.cwd() / 'gcare' / 'query' / dataset_name / query_type / query_name
    gt_path = Path.cwd() / 'gcare' / 'ground_truth' / dataset_name / query_type / query_name

    with gt_path.open('r') as f:
        gt = int(f.readline())
    
    res = _run_test(data_path, query_path)
    print(f'Our result: {res}, expected: {gt}, is_equal: {res == gt}')
    assert res == gt

def run_test_given_path(raw_data_path: str, raw_query_path: str, raw_gt_path: str):
    data_path = Path(raw_data_path)
    query_path = Path(raw_query_path)
    gt_path = Path(raw_gt_path)
    with gt_path.open('r') as f:
        gt = int(f.readline())
    
    res = _run_test(data_path, query_path)
    print(f'Our result: {res}, expected: {gt}, is_equal: {res == gt}')
    assert res == gt

def _run_test(data_path, query_path):
    max_vid, continous_label, sp_mats = read_gcare_data(data_path)
    expr, qvs, sp_mats_name = read_gcare_query(query_path, continous_label)
    sp_mats_needed = dict()
    for sp_name in sp_mats_name:
        if sp_name not in sp_mats:
            if sp_name.startswith('P'):  # Node id
                new_P = {}
                new_P['V'] = np.array([1])
                new_P['I_tuple'] = (
                    np.array([0]), np.array([int(sp_name[1:])]))
                new_P['shape'] = (max_vid+1, )
                # new_P = sp.coo_array(([1], ([0], [int(sp_name[1:])])), shape=(max_vid+1, ))
                print(len(new_P['I_tuple']), new_P['shape'])
                sp_mats_needed[sp_name] = BinsparseFormat.from_coo(new_P['I_tuple'], new_P['V'], new_P['shape'])
            else:
                # Some queried node / edge labels not existed in the data graph. The output must be 0.
                return 0
        else:
            sp_mat = sp_mats[sp_name]
            sp_mats_needed[sp_name] = BinsparseFormat.from_coo(sp_mat['I_tuple'], sp_mat['V'], sp_mat['shape'])
    
    # xp = NumpyFramework()
    xp = PyDataSparseFramework()
    return benchmark_subgraph_matching(xp, expr, sp_mats_needed)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: test_subgraph.py [data_path] [query_path] [gt_path]')
        run_test()
    else:
        run_test_given_path(sys.argv[1], sys.argv[2], sys.argv[3])
