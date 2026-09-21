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


def suite_sparse_rhs_dataset_name(source_name: str, rhs_index: int | None) -> str:
    if rhs_index is None:
        return source_name
    return f"{source_name}__rhs{rhs_index}"


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

_GAP_ROAD_SOURCES: list[int] = [
    4795720,
    21003853,
    417968,
    6496511,
    6648699,
    9811073,
    22247478,
    5720252,
    12366459,
    20413729,
    4217374,
    2674749,
    22085557,
    19445040,
    2360788,
    19115968,
    7758767,
    13468234,
    30367,
    18599547,
    7526108,
    16836280,
    12742067,
    7697995,
    5876443,
    9616340,
    2497673,
    10052290,
    12493057,
    1670855,
    2760679,
    2460941,
    8489650,
    5005225,
    8744645,
    8512023,
    21912165,
    1105390,
    15432163,
    1600177,
    19079469,
    16516637,
    20202566,
    21372803,
    2898009,
    8491277,
    18798317,
    23757560,
    17161819,
    23180739,
    10997085,
    3730630,
    1079068,
    15426822,
    12190925,
    1155218,
    10693488,
    14434835,
    19963339,
    3486185,
    18383269,
    20269908,
    12370764,
    7843140,
]

_GAP_TWITTER_SOURCES: list[int] = [
    12441072,
    54488257,
    25451915,
    57714473,
    14839494,
    32081104,
    52957357,
    50444380,
    49590701,
    20127816,
    34939333,
    48251001,
    19524253,
    43676726,
    33055508,
    15244687,
    24946738,
    6479472,
    26077682,
    22023875,
    22081915,
    40034162,
    49496014,
    42847507,
    52409557,
    55445388,
    22028097,
    48766648,
    44521241,
    60135542,
    28528671,
    9678012,
    40020306,
    31625735,
    37446892,
    51788952,
    52584255,
    20346696,
    48387909,
    37337427,
    50501084,
    30130061,
    41185893,
    56495703,
    45663305,
    33359460,
    48143058,
    33291513,
    53461445,
    29340610,
    34148498,
    49171806,
    35550696,
    14521507,
    51633218,
    46823382,
    19396273,
    19871750,
    36862677,
    49539126,
    34016452,
    36567395,
    55487793,
    14391370,
]

_GAP_WEB_SOURCES: list[int] = [
    10219452,
    44758211,
    890671,
    13843756,
    14168062,
    20906930,
    12189584,
    26352335,
    43500686,
    8987024,
    5699762,
    41436455,
    5030727,
    40735218,
    16533563,
    28700166,
    64711,
    39634750,
    16037779,
    27152739,
    16404061,
    20491963,
    5322423,
    21420953,
    26622109,
    5882875,
    18091040,
    10665896,
    18634422,
    18138715,
    2355535,
    32885205,
    40657440,
    35196167,
    45544426,
    6175519,
    40058318,
    50626230,
    36571019,
    49397052,
    23434265,
    2299444,
    32873823,
    25978282,
    2461715,
    22787314,
    30759947,
    7428894,
    39173870,
    43194209,
    26361509,
    39747211,
    30670029,
    41483033,
    9358666,
    9945008,
    3355244,
    33831269,
    45124744,
    16137877,
    11235448,
    37509144,
    27402414,
    39546083,
]

_GAP_KRON_SOURCES: list[int] = [
    2338012,
    31997659,
    23590940,
    43400604,
    75337937,
    169867,
    104041220,
    94177942,
    32871357,
    56230002,
    69883037,
    9346345,
    48915358,
    122571173,
    6183279,
    86323663,
    106725780,
    92389938,
    16210738,
    59816700,
    111669929,
    102831411,
    113384800,
    43872564,
    80508827,
    26105648,
    8807516,
    118452455,
    121818859,
    42361928,
    29493053,
    98461503,
    71931337,
    103808468,
    4092345,
    115276241,
    4649343,
    76656189,
    31312001,
    111334127,
    100962918,
    41823215,
    22631240,
    42848461,
    79485148,
    106818742,
    73347974,
    78848445,
    109920510,
    121492133,
    101037296,
    15438600,
    4584784,
    124503845,
    87241743,
    108297008,
    33955082,
    79934823,
    8608481,
    82435063,
    46579271,
    515421,
    121530467,
    127978736,
]

_GAP_URAND_SOURCES: list[int] = [
    27691419,
    121280314,
    2413431,
    37512113,
    38390877,
    56651037,
    128461248,
    33029842,
    71406328,
    117872827,
    24351938,
    15444519,
    127526281,
    112279428,
    13631649,
    110379302,
    44800623,
    77768193,
    175347,
    107397389,
    43457209,
    97215940,
    73575165,
    44449715,
    33931724,
    55526610,
    14422051,
    58043873,
    72137329,
    9647840,
    15940695,
    14209952,
    49020883,
    28901138,
    50493273,
    49150069,
    126525082,
    6382740,
    89108297,
    9239735,
    110168548,
    95370259,
    116653530,
    123410703,
    16733665,
    49030282,
    108545121,
    99095665,
    133850077,
    63499301,
    21541382,
    6230751,
    89077456,
    70392765,
    6670455,
    61746271,
    83349535,
    115272184,
    20129908,
    106148553,
    117042375,
    71431187,
    45287808,
    107702120,
]


class SuiteSparseMatrixGenerator(Generator[SuiteSparseDataset]):
    """Downloads and caches raw SuiteSparse matrices, shared across every benchmark."""

    @property
    def name(self) -> str:
        return "suitesparse_matrix"

    @property
    def pretty_name(self) -> str:
        return "SuiteSparse Matrix Collection"

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
