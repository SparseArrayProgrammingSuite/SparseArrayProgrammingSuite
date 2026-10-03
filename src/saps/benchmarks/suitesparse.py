from typing import Any

import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy

from saps.benchmark import (
    Author,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
    ShellBenchmark,
)
from saps.downloaders.suitesparse import (
    load_lpnetlib_problem,
    load_suitesparse_matrix,
    random_rhs_for_matrix,
)


class SuiteSparseDataset(Dataset):
    """Base Dataset for benchmarks backed by a SuiteSparse Matrix Collection matrix."""

    def __init__(
        self,
        name: str,
        *,
        source_name: str | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        nnz: int | None = None,
        rhs_index: int | None = None,
    ):
        self._name = name
        self.source_name = source_name if source_name is not None else name
        self._pretty_name = pretty_name
        self._description = description
        self._suites = suites or []
        self.nnz = nnz
        self.rhs_index = rhs_index

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name or self._name

    @property
    def description(self) -> str:
        return self._description or f"SuiteSparse matrix {self.source_name}."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        if self.nnz is not None:
            data["nnz"] = self.nnz
        if self.rhs_index is not None:
            data["rhs_index"] = self.rhs_index
        return data


# The Netlib linear programs in the LPnetlib group. Each one ships its objective
# and bound vectors beside the matrix, which the LP-aware branch of
# SuiteSparseMatrixGenerator.generate reads along with A and b.
_LPNETLIB_PROBLEMS: list[str] = [
    "lp_25fv47",
    "lp_80bau3b",
    "lp_adlittle",
    "lp_afiro",
    "lp_agg",
    "lp_agg2",
    "lp_agg3",
    "lp_bandm",
    "lp_beaconfd",
    "lp_blend",
    "lp_bnl1",
    "lp_bnl2",
    "lp_bore3d",
    "lp_brandy",
    "lp_capri",
    "lp_cre_a",
    "lp_cre_b",
    "lp_cre_c",
    "lp_cre_d",
    "lp_cycle",
    "lp_czprob",
    "lp_d2q06c",
    "lp_d6cube",
    "lp_degen2",
    "lp_degen3",
    "lp_dfl001",
    "lp_e226",
    "lp_etamacro",
    "lp_fffff800",
    "lp_finnis",
    "lp_fit1d",
    "lp_fit1p",
    "lp_fit2d",
    "lp_fit2p",
    "lp_ganges",
    "lp_gfrd_pnc",
    "lp_greenbea",
    "lp_greenbeb",
    "lp_grow15",
    "lp_grow22",
    "lp_grow7",
    "lp_israel",
    "lp_kb2",
    "lp_ken_07",
    "lp_ken_11",
    "lp_ken_13",
    "lp_ken_18",
    "lp_lotfi",
    "lp_maros",
    "lp_maros_r7",
    "lp_modszk1",
    "lp_osa_07",
    "lp_osa_14",
    "lp_osa_30",
    "lp_osa_60",
    "lp_pds_02",
    "lp_pds_06",
    "lp_pds_10",
    "lp_pds_20",
    "lp_perold",
    "lp_pilot",
    "lp_pilot4",
    "lp_pilot87",
    "lp_pilot_ja",
    "lp_pilot_we",
    "lp_pilotnov",
    "lp_qap12",
    "lp_qap15",
    "lp_qap8",
    "lp_recipe",
    "lp_sc105",
    "lp_sc205",
    "lp_sc50a",
    "lp_sc50b",
    "lp_scagr25",
    "lp_scagr7",
    "lp_scfxm1",
    "lp_scfxm2",
    "lp_scfxm3",
    "lp_scorpion",
    "lp_scrs8",
    "lp_scsd1",
    "lp_scsd6",
    "lp_scsd8",
    "lp_sctap1",
    "lp_sctap2",
    "lp_sctap3",
    "lp_share1b",
    "lp_share2b",
    "lp_shell",
    "lp_ship04l",
    "lp_ship04s",
    "lp_ship08l",
    "lp_ship08s",
    "lp_ship12l",
    "lp_ship12s",
    "lp_sierra",
    "lp_stair",
    "lp_standata",
    "lp_standgub",
    "lp_standmps",
    "lp_stocfor1",
    "lp_stocfor2",
    "lp_stocfor3",
    "lp_truss",
    "lp_tuff",
    "lp_vtp_base",
    "lp_wood1p",
    "lp_woodw",
    "lpi_bgdbg1",
    "lpi_bgetam",
    "lpi_bgindy",
    "lpi_bgprtr",
    "lpi_box1",
    "lpi_ceria3d",
    "lpi_chemcom",
    "lpi_cplex1",
    "lpi_cplex2",
    "lpi_ex72a",
    "lpi_ex73a",
    "lpi_forest6",
    "lpi_galenet",
    "lpi_gosh",
    "lpi_gran",
    "lpi_greenbea",
    "lpi_itest2",
    "lpi_itest6",
    "lpi_klein1",
    "lpi_klein2",
    "lpi_klein3",
    "lpi_mondou2",
    "lpi_pang",
    "lpi_pilot4i",
    "lpi_qual",
    "lpi_reactor",
    "lpi_refinery",
    "lpi_vol1",
    "lpi_woodinfe",
]

_MATRICES: list[SuiteSparseDataset] = [
    SuiteSparseDataset(name)
    for name in [
        "Pothen/mesh3em5",
        "HB/bcsstm02",
        "Norris/fv1",
        "MathWorks/Muu",
        "JGD_Trefethen/Trefethen_200",
        "Norris/fv2",
        "HB/ash958",
        "Bai/mhdb416",
        "HB/lund_b",
        "HB/bcsstm12",
        "Pothen/mesh1em1",
        "HB/bcsstk05",
        "HB/nos1",
        "HB/nos2",
        "HB/nos3",
        "HB/dwt_59",
        "HB/bcspwr01",
        "HB/bcspwr02",
        "HB/bcspwr03",
        "DIMACS10/chesapeake",
        "HB/ash85",
        "HB/arc130",
        "HB/bcspwr04",
        "HB/ash292",
        "Newman/karate",
        "Newman/dolphins",
        "SNAP/ca-GrQc",
        "Arenas/email",
        "Muite/Chebyshev3",
        "SNAP/ca-HepPh",
        "HB/bcsstk01",
        "GAP/GAP-road",
        "GAP/GAP-twitter",
        "GAP/GAP-web",
        "GAP/GAP-kron",
        "GAP/GAP-urand",
        "ANSYS/Delor338K",
        "Andrews/Andrews",
        "Andrianov/ins2",
        "Andrianov/net100",
        "Andrianov/net125",
        "Andrianov/net150",
        "Andrianov/net25",
        "Andrianov/net50",
        "Andrianov/net75",
        "Bai/bfwa62",
        "Bai/bfwb398",
        "Bai/bfwb62",
        "Bai/bfwb782",
        "Bai/cdde2",
        "Bai/cdde4",
        "Bai/cdde6",
        "Bai/ck104",
        "Bai/dw256B",
        "Bai/dwb512",
        "Bai/mhd3200b",
        "Bai/mhd4800b",
        "Bai/odepb400",
        "Bai/pde225",
        "Bai/pde900",
        "Bai/rdb200",
        "Bai/rdb200l",
        "Bindel/ted_B",
        "Bindel/ted_B_unscaled",
        "Boeing/bcsstk34",
        "Boeing/bcsstm39",
        "Boeing/crystm01",
        "Boeing/crystm02",
        "Boeing/crystm03",
        "Boeing/msc00726",
        "Botonakis/FEM_3D_thermal1",
        "Botonakis/FEM_3D_thermal2",
        "Botonakis/thermomech_TC",
        "Botonakis/thermomech_dM",
        "Brunetiere/thermal",
        "CPM/cz148",
        "Cunningham/m3plates",
        "Cunningham/qa8fk",
        "Cunningham/qa8fm",
        "FEMLAB/problem1",
        "FIDAP/ex29",
        "FIDAP/ex37",
        "FIDAP/ex5",
        "FIDAP/ex7",
        "Freescale/circuit5M_dc",
        "GHS_indef/blockqp1",
        "GHS_indef/laser",
        "GHS_indef/qpband",
        "GHS_psdef/jnlbrng1",
        "GHS_psdef/minsurfo",
        "GHS_psdef/obstclae",
        "GHS_psdef/wathen100",
        "GHS_psdef/wathen120",
        "Grund/meg4",
        "Grund/poli3",
        "Grund/poli4",
        "Grund/poli_large",
        "Guettel/TEM27623",
        "HB/ash219",
        "HB/ash331",
        "HB/ash608",
        "HB/bcsstk02",
        "HB/bcsstk03",
        "HB/bcsstk04",
        "HB/bcsstk08",
        "HB/bcsstk20",
        "HB/bcsstk22",
        "HB/bcsstm01",
        "HB/bcsstm03",
        "HB/bcsstm04",
        "HB/bcsstm05",
        "HB/bcsstm06",
        "HB/bcsstm07",
        "HB/bcsstm08",
        "HB/bcsstm09",
        "HB/bcsstm11",
        "HB/bcsstm19",
        "HB/bcsstm20",
        "HB/bcsstm21",
        "HB/bcsstm22",
        "HB/bcsstm23",
        "HB/bcsstm24",
        "HB/bcsstm25",
        "HB/bcsstm26",
        "HB/can_144",
        "HB/can_24",
        "HB/can_61",
        "HB/can_62",
        "HB/can_73",
        "HB/can_96",
        "HB/curtis54",
        "HB/dwt_66",
        "HB/dwt_72",
        "HB/fs_183_1",
        "HB/fs_183_3",
        "HB/fs_183_4",
        "HB/fs_183_6",
        "HB/fs_541_1",
        "HB/fs_680_1",
        "HB/fs_680_2",
        "HB/fs_760_1",
        "HB/fs_760_2",
        "HB/fs_760_3",
        "HB/gr_30_30",
        "HB/jpwh_991",
        "HB/lap_25",
        "HB/lund_a",
        "HB/nos4",
        "HB/nos6",
        "HB/nos7",
        "HB/pores_1",
        "HB/psmigr_3",
        "HB/steam2",
        "HB/steam3",
        "HB/watt_1",
        "HB/watt_2",
        "Lourakis/bundle1",
        "MKS/fp",
        "MathWorks/tomography",
        "MaxPlanck/shallow_water1",
        "MaxPlanck/shallow_water2",
        "Morandini/rotor1",
        "Mulvey/finan512",
        "Nemeth/nemeth02",
        "Nemeth/nemeth03",
        "Nemeth/nemeth04",
        "Nemeth/nemeth05",
        "Nemeth/nemeth06",
        "Nemeth/nemeth07",
        "Nemeth/nemeth08",
        "Nemeth/nemeth09",
        "Nemeth/nemeth10",
        "Nemeth/nemeth11",
        "Nemeth/nemeth12",
        "Nemeth/nemeth13",
        "Nemeth/nemeth16",
        "Nemeth/nemeth17",
        "Nemeth/nemeth18",
        "Nemeth/nemeth19",
        "Nemeth/nemeth20",
        "Nemeth/nemeth21",
        "Nemeth/nemeth22",
        "Nemeth/nemeth23",
        "Nemeth/nemeth24",
        "Nemeth/nemeth25",
        "Nemeth/nemeth26",
        "Norris/torso2",
        "Oberwolfach/LF10",
        "Oberwolfach/LFAT5",
        "PARSEC/Si2",
        "Pothen/bodyy4",
        "Pothen/mesh1e1",
        "Pothen/mesh1em6",
        "Pothen/mesh2e1",
        "Pothen/mesh2em5",
        "Pothen/mesh3e1",
        "Pothen/sphere2",
        "Precima/analytics",
        "QLi/majorbasis",
        "Rajat/rajat13",
        "Rommes/bips98_1450",
        "Rommes/bips98_606",
        "Rommes/ww_36_pmec_36",
        "Sandia/ASIC_100k",
        "Sandia/ASIC_100ks",
        "Sandia/ASIC_320ks",
        "Sandia/ASIC_680k",
        "Sandia/ASIC_680ks",
        "Sandia/adder_dcop_31",
        "Sandia/adder_dcop_32",
        "Sandia/adder_dcop_33",
        "Sandia/adder_dcop_34",
        "Sandia/adder_dcop_35",
        "Sandia/adder_dcop_36",
        "Sandia/adder_dcop_37",
        "Sandia/adder_dcop_38",
        "Sandia/adder_dcop_40",
        "Sandia/adder_dcop_41",
        "Sandia/adder_dcop_42",
        "Sandia/adder_dcop_43",
        "Sandia/adder_dcop_44",
        "Sandia/adder_dcop_45",
        "Sandia/adder_dcop_46",
        "Sandia/adder_dcop_47",
        "Sandia/adder_dcop_48",
        "Sandia/adder_dcop_49",
        "Sandia/adder_dcop_50",
        "Sandia/adder_dcop_51",
        "Sandia/adder_dcop_52",
        "Sandia/adder_dcop_53",
        "Sandia/adder_dcop_54",
        "Sandia/adder_dcop_55",
        "Sandia/adder_dcop_57",
        "Sandia/adder_dcop_58",
        "Sandia/adder_dcop_59",
        "Sandia/adder_dcop_60",
        "Sandia/adder_dcop_61",
        "Sandia/adder_dcop_62",
        "Sandia/adder_dcop_63",
        "Sandia/adder_dcop_64",
        "Sandia/adder_dcop_65",
        "Sandia/adder_dcop_66",
        "Sandia/adder_dcop_67",
        "Sandia/adder_dcop_68",
        "Sandia/adder_dcop_69",
        "Sandia/adder_trans_01",
        "Sandia/adder_trans_02",
        "VLSI/ss1",
        "Wang/swang1",
        "Wang/swang2",
        "Zhao/Zhao1",
        "ATandT/onetone2",
        "Boeing/bcsstk35",
        "Boeing/crystk02",
        "Boeing/crystk03",
        "Boeing/ct20stif",
        "Brethour/coater2",
        "Cote/vibrobox",
        "FIDAP/ex11",
        "Goodwin/goodwin",
        "Goodwin/rim",
        "Grund/bayer02",
        "Grund/bayer10",
        "HB/bcsstk09",
        "HB/gemat11",
        "HB/orani678",
        "Hamm/memplus",
        "Mallya/lhr10",
        "Nasa/nasasrb",
        "Nasa/pwt",
        "Rothberg/3dtube",
        "SNAP/CollegeMsg",
        "SNAP/email-Eu-core",
        "SNAP/wiki-Vote",
        "SNAP/Oregon-1",
        "SNAP/Oregon-2",
        "SNAP/amazon0302",
        "SNAP/amazon0312",
        "SNAP/amazon0505",
        "SNAP/amazon0601",
        "SNAP/as-735",
        "SNAP/as-Skitter",
        "SNAP/as-caida",
        "SNAP/ca-AstroPh",
        "SNAP/ca-CondMat",
        "SNAP/ca-HepTh",
        "SNAP/cit-HepPh",
        "SNAP/cit-HepTh",
        "SNAP/cit-Patents",
        "SNAP/com-Amazon",
        "SNAP/com-DBLP",
        "SNAP/com-Friendster",
        "SNAP/com-LiveJournal",
        "SNAP/com-Orkut",
        "SNAP/com-Youtube",
        "SNAP/email-Enron",
        "SNAP/email-Eu-core-temporal",
        "SNAP/email-EuAll",
        "SNAP/higgs-twitter",
        "SNAP/loc-Brightkite",
        "SNAP/loc-Gowalla",
        "SNAP/p2p-Gnutella04",
        "SNAP/p2p-Gnutella05",
        "SNAP/p2p-Gnutella06",
        "SNAP/p2p-Gnutella08",
        "SNAP/p2p-Gnutella09",
        "SNAP/p2p-Gnutella24",
        "SNAP/p2p-Gnutella25",
        "SNAP/p2p-Gnutella30",
        "SNAP/p2p-Gnutella31",
        "SNAP/roadNet-CA",
        "SNAP/roadNet-PA",
        "SNAP/roadNet-TX",
        "SNAP/soc-Epinions1",
        "SNAP/soc-LiveJournal1",
        "SNAP/soc-Pokec",
        "SNAP/soc-Slashdot0811",
        "SNAP/soc-Slashdot0902",
        "SNAP/soc-sign-Slashdot081106",
        "SNAP/soc-sign-Slashdot090216",
        "SNAP/soc-sign-Slashdot090221",
        "SNAP/soc-sign-bitcoin-alpha",
        "SNAP/soc-sign-bitcoin-otc",
        "SNAP/soc-sign-epinions",
        "SNAP/sx-askubuntu",
        "SNAP/sx-mathoverflow",
        "SNAP/sx-stackoverflow",
        "SNAP/sx-superuser",
        "SNAP/twitter7",
        "SNAP/web-BerkStan",
        "SNAP/web-Google",
        "SNAP/web-NotreDame",
        "SNAP/web-Stanford",
        "SNAP/wiki-RfA",
        "SNAP/wiki-Talk",
        "SNAP/wiki-talk-temporal",
        "SNAP/wiki-topcats",
        "Simon/olafu",
        "Simon/raefsky3",
        "Simon/raefsky4",
        "Simon/venkat01",
        "UTEP/Dubcova1",
        "Wang/wang4",
        "Zitney/rdist1",
        "Bomhof/circuit_1",
        "Bourchtein/atmosmodd",
        "Bourchtein/atmosmodj",
        "Bourchtein/atmosmodl",
        "Bourchtein/atmosmodm",
        "FEMLAB/poisson2D",
        "GHS_indef/boyd1",
        "GHS_indef/boyd2",
        "Grund/b1_ss",
        "Grund/poli",
        "Hamm/add32",
        "Hamrle/Hamrle1",
        "NYPA/Maragal_1",
        "NYPA/Maragal_2",
        "NYPA/Maragal_3",
        "NYPA/Maragal_4",
        "NYPA/Maragal_5",
        "NYPA/Maragal_6",
        "Nasa/nasa2146",
        "Sandia/mult_dcop_02",
        "Schenk_AFE/af_shell3",
        "Schenk_AFE/af_shell4",
        "Schenk_AFE/af_shell7",
        "Schenk_AFE/af_shell8",
        "Simon/raefsky5",
        "Simon/raefsky6",
        "TOKAMAK/utm1700b",
        "TOKAMAK/utm3060",
        "Um/2cubes_sphere",
        "VDOL/hangGlider_1",
        "VDOL/tumorAntiAngiogenesis_1",
        "VDOL/tumorAntiAngiogenesis_2",
    ]
]
_MATRICES += [SuiteSparseDataset(f"LPnetlib/{name}") for name in _LPNETLIB_PROBLEMS]


class SuiteSparseMatrixGenerator(Generator[SuiteSparseDataset]):
    """Downloads and caches raw SuiteSparse matrices, shared across every benchmark."""

    @property
    def name(self) -> str:
        return "suitesparse_matrix"

    @property
    def pretty_name(self) -> str:
        return "SuiteSparse Matrices"

    @property
    def description(self) -> str:
        return (
            "Downloads and caches raw matrices from the SuiteSparse Matrix Collection."
            " Benchmark-specific generators compose this generator instead of"
            " downloading matrices themselves, so a matrix used by multiple benchmarks"
            " is only downloaded, cached, and uploaded once."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="The university of Florida sparse matrix collection",
                authors=[
                    Author("Timothy A. Davis"),
                    Author("Yifan Hu"),
                ],
                journal="ACM Transactions on Mathematical Software",
                publisher="Association for Computing Machinery (ACM)",
                volume="38",
                number="1",
                pages="1-25",
                year=2011,
                url="https://doi.org/10.1145/2049662.2049663",
                doi="10.1145/2049662.2049663",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the algorithms for the benchmark"
            " function. Generative AI might have been used to construct the framework,"
            " comments and helper functions."
        )

    @property
    def motivation(self) -> str:
        return (
            "Many benchmarks reuse the same SuiteSparse matrices. Sharing a single"
            " cacheable generator for the raw download avoids redundant downloads and"
            " redundant cached copies of the same matrix."
        )

    @property
    def datasets(self) -> list[SuiteSparseDataset]:
        return _MATRICES

    def generate(self, dataset: SuiteSparseDataset) -> DataInstance:
        if dataset.source_name.startswith("LPnetlib/"):
            A, rhs, c, lo, hi, meta = load_lpnetlib_problem(dataset.source_name)
            inputs = [from_scipy(A)]
            inputs.extend(from_numpy(vector) for vector in (rhs, c, lo, hi))
            return DataInstance(inputs=inputs, meta=meta)
        A, b, meta = load_suitesparse_matrix(dataset.source_name)
        inputs = [from_scipy(A)]
        if b is not None:
            inputs.append(from_numpy(b))
        return DataInstance(inputs=inputs, meta=meta)


class SuiteSparseMatrixBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return SuiteSparseMatrixGenerator()


def fetch_suitesparse_matrix(source_name: str) -> DataInstance:
    """Fetch a listed raw matrix via the shared `SuiteSparseMatrixGenerator`.

    `.inputs[0]` is the matrix; `.inputs[1]` contains all compatible real RHSs
    when available (see `.meta["has_b_file"]`): a vector for one RHS, or a matrix
    with one RHS per column. Consumers select RHSs after fetching this shared data.
    `.meta["shape"]` and `.meta["nnz"]` give the matrix shape/nnz. LPnetlib
    entries always carry `b`, followed by the objective `c` and the bounds `lo`
    and `hi` as `.inputs[2:5]`, with the objective offset in `.meta["z0"]`.
    """
    raw_generator = SuiteSparseMatrixGenerator()
    raw_dataset = next(
        (d for d in raw_generator.datasets if d.source_name == source_name),
        None,
    )
    if raw_dataset is None:
        raise ValueError(
            f"Dataset {source_name!r} "
            "is not listed in SuiteSparseMatrixGenerator.datasets. "
            "Add it to the shell dataset list before using it."
        )
    return raw_generator.cached_generate(raw_dataset)


def fetch_suitesparse_linear_system(
    source_name: str,
    *,
    rhs_index: int | None = None,
) -> tuple[BinsparseTensor, np.ndarray, bool]:
    """Fetch a matrix paired with a right-hand-side vector `b` to solve against.

    Returns `(A, b, has_real_rhs)`. This helper runs in the uncached solver
    generators: it fetches the shared matrix and all RHS vectors, then selects
    *rhs_index*. With no index, it uses a sole real RHS if available; otherwise
    it synthesizes `b = A @ x` using `random_rhs_for_matrix`'s defaults. The
    boolean reports whether the returned vector came from the real RHS data.
    """
    if rhs_index is not None and rhs_index < 0:
        raise ValueError(f"rhs_index must be nonnegative, got {rhs_index}")
    raw = fetch_suitesparse_matrix(source_name)
    A_bin = raw.inputs[0]
    if len(raw.inputs) > 1:
        rhs = to_numpy(raw.inputs[1])
        rhs_count = 1 if rhs.ndim == 1 else rhs.shape[1]
        if rhs_index is not None and rhs_index >= rhs_count:
            raise ValueError(
                f"SuiteSparse matrix '{source_name}' contains {rhs_count} RHS "
                f"vectors, got rhs_index={rhs_index}"
            )
        if rhs_index is not None or rhs_count == 1:
            index = 0 if rhs_index is None else rhs_index
            b = rhs if rhs.ndim == 1 else rhs[:, index]
            return A_bin, b, True
    elif rhs_index is not None:
        raise ValueError(
            f"SuiteSparse matrix '{source_name}' has no compatible RHS file"
        )
    b = random_rhs_for_matrix(to_scipy(A_bin).tocoo())
    return A_bin, b, False
