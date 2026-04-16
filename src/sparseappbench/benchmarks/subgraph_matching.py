import numpy as np
from pathlib import Path
import tarfile
import gdown

import sparseappbench
from sparseappbench.benchmark import (
    Author,
    Benchmark,
    BinsparseFormat,
    Contributor,
    Dataset,
    Generator,
    Ref,
)

xp = sparseappbench.xp


def download_gcare_data(root_dir: Path):
    dataset_dir = root_dir / 'dataset'
    queryset_dir = root_dir / 'queryset'
    ground_truth_dir = root_dir / 'ground_truth'

    dataset_link = 'https://drive.google.com/file/d/1HAgSVE-24NOap6_Q1_twH56Dkb2kPvGU/view?usp=sharing'
    queryset_link = 'https://drive.google.com/file/d/1Dlj43rBAOVPAsfzKlYxIbZ9RsqeGM_MN/view?usp=sharing'
    ground_truth_link = 'https://drive.google.com/file/d/1Bc6Q2RZQTcIB8IfOw5KafNYwPhq2BO94/view?usp=sharing'

    # Download dataset
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)

    gdown.cached_download(dataset_link,
                          str(dataset_dir / 'dataset.tar.gz'),
                          hash='sha256:78B86CDA06115C4554CDFCFB93A7FBC8ECB759DF39927510DD02CED4228A95E4'.lower())

    with tarfile.open(dataset_dir / 'dataset.tar.gz', 'r:gz') as tar:
        tar.extractall(path=dataset_dir, filter='data')

    # Download queryset
    if not queryset_dir.exists():
        queryset_dir.mkdir(parents=True, exist_ok=True)

    gdown.cached_download(queryset_link,
                          str(queryset_dir / 'queryset.tar.gz'),
                          hash='sha256:C8DC9F978296559E9E55335A989CE16E7B5BCBA7AA9D43E25FBD9E588D00EBC7'.lower())

    with tarfile.open(queryset_dir / 'queryset.tar.gz', 'r:gz') as tar:
        tar.extractall(path=queryset_dir, filter='data')

    # Download true cardinality
    if not ground_truth_dir.exists():
        ground_truth_dir.mkdir(parents=True, exist_ok=True)

    gdown.cached_download(ground_truth_link,
                          str(ground_truth_dir / 'ground_truth.tar.gz'),
                          hash='sha256:22E59F4FC06FFB79711D582513C6422CA555422C9947FDE64A52F0A9292D382C'.lower())

    with tarfile.open(ground_truth_dir / 'ground_truth.tar.gz', 'r:gz') as tar:
        tar.extractall(path=ground_truth_dir, filter='data')

    return (dataset_dir, queryset_dir, ground_truth_dir)


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


class SubgraphGCareDataset(Dataset):
    def __init__(
        self,
        name,
        pretty_name,
        description,
        tags,
    ):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._tags = tags

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def tags(self) -> list[str]:
        return self._tags


class SubgraphGCareGenerator(Generator[SubgraphGCareDataset]):
    @property
    def name(self) -> str:
        return "subgraph_gcare_inputs"

    @property
    def pretty_name(self) -> str:
        return "Subgraph G-CARE Input Generator"

    @property
    def description(self) -> str:
        return (
            "Transforms the G-CARE dataset to the input of subgraph matching algorithms. "
        )

    @property
    def tags(self) -> list[str]:
        return ["subgraph matching", "sparse"]

    @property
    def authors(self) -> list[Contributor]:
        return [
            Contributor("Taishan Chen", "utallow@bu.edu"),
            Contributor("Kyle Deeds", "kdeeds@bu.edu"),
        ]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "G-CARE: "
                    "A Framework for Performance Benchmarking of Cardinality Estimation Techniques "
                    "for Subgraph Matching "
                ),
                authors=[
                    Author("Yeonsu Park"),
                    Author("Seongyun Ko"),
                    Author("Sourav S Bhowmick"),
                    Author("Kyoungmin Kim"),
                    Author("Kijae Hong"),
                    Author("Wook-Shin Han"),
                ],
                year=2020,
                url="https://dl.acm.org/doi/10.1145/3318464.3389702",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the algorithms for the benchmark function. "
            "Generative AI might have been used to construct the definition of the framework."
        )

    @property
    def motivation(self) -> str:
        return (
            "Subgraph matching and counting are classic problems and widely used in query evaluations in database systems. "
        )

    @property
    def datasets(self) -> list[SubgraphGCareDataset]:
        return [
            SubgraphGCareDataset(
                name="human",
                pretty_name="G-CARE Human Subset (Small)",
                description=(
                    "G-CARE Human Subset (Small)"
                ),
                tags=["small", "sparse"],
            ),
            SubgraphGCareDataset(
                name="aids",
                pretty_name="G-CARE AIDS Subset (Medium)",
                description=(
                    "G-CARE AIDS Subset (Medium)"
                ),
                tags=["medium", "sparse"],
            ),
            SubgraphGCareDataset(
                name="lubm80",
                pretty_name="G-CARE LUBM80 Subset (Large)",
                description=(
                    "G-CARE LUBM80 Subset (Large)"
                ),
                tags=["large", "sparse"],
            ),
            SubgraphGCareDataset(
                name="yago",
                pretty_name="G-CARE YAGO Subset (Huge)",
                description=(
                    "G-CARE YAGO Subset (Huge)"
                ),
                tags=["huge", "sparse"],
            ),
        ]

    def generate(self, dataset: SubgraphGCareDataset):
        root_dir = Path('./data/gcare')
        (dataset_dir, queryset_dir, ground_truth_dir) = download_gcare_data(root_dir)
        (max_vid, continous_label, all_sp_mats) = read_gcare_data(
            dataset_dir / dataset.name / f'{dataset.name}.txt')

        queries = {}

        for query_path in (queryset_dir / dataset.name).rglob('*.txt'):
            (sp_mats_needed, expr) = process_one_query(
                query_path, all_sp_mats, max_vid, continous_label)
            queries[query_path.stem] = {
                "matrices": sp_mats_needed, "expr": expr}

        for gt_path in (ground_truth_dir / dataset.name).rglob('*.txt'):
            with open(gt_path, 'r') as f:
                queries[gt_path.stem]["ground_truth"] = int(f.readline())

        matrices = []
        meta = {'exprs': [], 'gts': [], 'names': []}
        for query_name, query_data in queries.items():
            matrices.append(query_data["matrices"])
            meta['exprs'].append(query_data["expr"])
            meta['gts'].append(query_data["ground_truth"])
            meta['names'].append(query_name)

        return matrices, meta


class SubgraphMatching(Benchmark):
    @property
    def tag(self):
        return "subgraph_matching"

    @property
    def name(self):
        return "Subgraph Matching Algorithm using einsum"

    @property
    def pretty_name(self):
        return "Subgraph Matching Algorithm using einsum"

    @property
    def description(self):
        return (
            "Benchmarks subgraph matching algorithms using einsum operations."
        )

    @property
    def tags(self):
        return ["subgraph-matching", "sparse"]

    @property
    def authors(self):
        return [Contributor("Taishan Chen", "utallow@bu.edu"),
                Contributor("Kyle Deeds", "kdeeds@bu.edu")]

    @property
    def references(self):
        return []

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used to write the algorithms for the benchmark function. "
            "Generative AI might have been used to construct the definition of the framework."
        )

    @property
    def motivation(self):
        return (
            "Subgraph matching and counting are classic problems and widely used in query evaluations in database systems. "
        )

    @property
    def generators(self):
        return [SubgraphGCareGenerator()]

    def benchmark(self, data, meta):
        exprs = meta['exprs']
        assert len(data) == len(exprs)
        counts = np.zeros((len(data), ), dtype=np.int64)
        for i in range(len(data)):
            sp_mats = data[i]
            expr = exprs[i]
            for key, val in sp_mats.items():
                sp_mats[key] = xp.from_binsparse(val)
            counts[i] = xp.einsum(expr, **sp_mats)
        return xp.to_binsparse(counts)
