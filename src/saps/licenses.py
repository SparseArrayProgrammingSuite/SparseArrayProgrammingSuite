"""Licenses of the data that SAPS generators synthesize or download.

Each generator's ``license`` property returns one of these constants. A
`License` records the SPDX expression, where it was found, and the day it was
read there. ``None`` means the source was checked and no license for its data
could be found; the comment on each such constant says where it was checked.
A license file covering a repository that merely redistributes someone else's
data is not counted as a license for that data.
"""

from __future__ import annotations

import datetime

from saps.benchmark import Author, License, Ref

# All licenses below were checked on this day.
_ACCESSED_2026_11_18 = datetime.date(2026, 11, 18)

# Data synthesized by SAPS's own generator code (random matrices, test cases,
# discretized PDEs, and the like).
SAPS = License(
    spdx="MIT",
    ref=Ref(
        title="SparseArrayProgrammingSuite LICENSE",
        authors=[Author("Willow Ahrens")],
        year=2025,
        url="https://github.com/willow-ahrens/SparseArrayProgrammingSuite/blob/main/LICENSE",
    ),
    date_accessed=_ACCESSED_2026_11_18,
)

# The SuiteSparse Matrix Collection, including the matrices SAPS takes from its
# SNAP, GAP, and LPnetlib groups. The About page states: "The matrices
# themselves are under the CC-BY 4.0 License."
SUITESPARSE = License(
    spdx="CC-BY-4.0",
    ref=Ref(
        title="About the SuiteSparse Matrix Collection",
        authors=[Author("Timothy A. Davis")],
        institution="Texas A&M University",
        url="https://sparse.tamu.edu/about",
    ),
    date_accessed=_ACCESSED_2026_11_18,
)

# The OGB node property prediction datasets that SAPS uses, each licensed
# separately on the same page: ogbn-arxiv under ODC-BY, ogbn-products under the
# "Amazon license" (https://s3.amazonaws.com/amazon-reviews-pds/license.txt,
# which no longer resolves), and ogbn-proteins under CC-0.
OGB_NODEPROP = License(
    spdx="ODC-By-1.0 AND LicenseRef-Amazon-Reviews-PDS AND CC0-1.0",
    ref=Ref(
        title="Node Property Prediction",
        authors=[Author("Open Graph Benchmark")],
        url="https://ogb.stanford.edu/docs/nodeprop/",
    ),
    date_accessed=_ACCESSED_2026_11_18,
)

# QASMBench circuits, under a BSD-style license from Battelle Memorial
# Institute that has no SPDX identifier. The license file is unchanged between
# the commit the downloader pins and the default branch.
QASMBENCH = License(
    spdx="LicenseRef-QASMBench",
    ref=Ref(
        title="QASMBench LICENSE",
        authors=[Author("Ang Li")],
        institution="Pacific Northwest National Laboratory",
        year=2020,
        url="https://github.com/pnnl/QASMBench/blob/master/LICENSE",
    ),
    date_accessed=_ACCESSED_2026_11_18,
)

# The pruned Conv-2 network for the lottery-ticket benchmark.
LTH_CONV2 = License(
    spdx="CC-BY-4.0",
    ref=Ref(
        title="CIFAR Conv-2 Lottery Ticket Hypothesis Benchmark Artifacts",
        authors=[Author("Ramya Polaki")],
        publisher="Zenodo",
        year=2026,
        url="https://zenodo.org/records/22650920",
    ),
    date_accessed=_ACCESSED_2026_11_18,
)

# SLICOT's "Benchmark Examples for Model Reduction". Checked 2026-11-18: no
# license on https://www.slicot.org/20-site/126-benchmark-examples-for-model-reduction,
# in its archives, in the SLICOT/Benchmark-ModelReduction GitHub repository, in
# SLICOT Working Note 2002-2, or on MORwiki ("License: NA").
SLICOT_MODEL_REDUCTION: License | None = None

# FROSTT tensors. Checked 2026-11-18: the tensor pages of
# https://github.com/frostt-tensor/frostt-tensor.github.io state a license only
# for lanl2 and uber_pickups, which have their own constants below; the other
# tensors SAPS uses state none. The repository's MIT license covers the website.
FROSTT: License | None = None

# FROSTT's lanl2 tensor. Its page states "Public Domain" followed by CC0's
# waiver wording: "To the extent possible under law, Los Alamos National
# Laboratory has waived all copyright and related or neighboring rights to
# Comprehensive, Multi-Source Cyber-Security Events."
FROSTT_LANL2 = License(
    spdx="CC0-1.0",
    ref=Ref(
        title="LANL2 Network Traffic",
        authors=[Author("FROSTT")],
        url="http://frostt.io/tensors/lanl2/",
    ),
    date_accessed=_ACCESSED_2026_11_18,
)

# FROSTT's uber_pickups tensor. Its page states "CC0: Public Domain".
FROSTT_UBER_PICKUPS = License(
    spdx="CC0-1.0",
    ref=Ref(
        title="Uber Pickups",
        authors=[Author("FROSTT")],
        url="http://frostt.io/tensors/uber-pickups/",
    ),
    date_accessed=_ACCESSED_2026_11_18,
)

# OpenML datasets 554 (mnist_784) and 40927 (CIFAR_10). Checked 2026-11-18:
# both report licence "Public" through https://www.openml.org/api/v1/json/data/,
# which is OpenML's default value for that field rather than a license.
OPENML: License | None = None

# Netflix Prize data from Kaggle (netflix-inc/netflix-prize-data). Checked
# 2026-11-18: Kaggle lists "Other (specified in description)", but the
# description specifies no license.
NETFLIX_PRIZE: License | None = None

# Brain MRI images from Kaggle
# (navoneel/brain-mri-images-for-brain-tumor-detection). Checked 2026-11-18:
# Kaggle lists only "Data files © Original Authors".
KAGGLE_BRAIN_MRI: License | None = None

# Model counting competition instances from
# https://github.com/arijitsh/mccomp-test-instances. Checked 2026-11-18: the
# repository has no license file or README.
MCCOMP_TEST_INSTANCES: License | None = None

# NEMO archive snapshots (Plummer models and Dubinski's Milky Way/Andromeda
# initial conditions). Checked 2026-11-18: https://carma.astro.umd.edu/nemo/archive/
# states no license; NEMO's GPL-2.0 covers its software, not the archive.
NEMO_ARCHIVE: License | None = None

# The ETH Walking Pedestrians (EWAP) dataset, fetched from OpenTraj. Checked
# 2026-11-18: OpenTraj's README for the dataset states no license, and its
# repository-wide MIT license does not cover ETH Zurich's data.
EWAP: License | None = None

# G-CARE subgraph matching datasets and query sets. Checked 2026-11-18, with no
# license found in any of: https://github.com/yspark-dblab/gcare, which links the
# Google Drive archives (README, all files, history); the lab's own fork
# https://github.com/postechdblab/gcare and the repository's nine other forks;
# the dataset, queryset, and true cardinality archives, which hold only data
# files; the project page
# https://sites.google.com/dblab.postech.ac.kr/kmkim/%ED%99%88/g-care; and the
# SIGMOD 2020 paper, whose only notice is ACM's copyright on the article.
# Each of G-CARE's four graphs is licensed by its upstream source below, and the
# graph datasets carry those licenses themselves.
GCARE: License | None = None

# G-CARE's yago graph: the YAGO 2009-w10-5 Notation 3 export (yago.n3, dated
# 2009-03-06). 65 of the graph's 91 relation sizes equal that export's
# predicate counts exactly (e.g. hasFamilyName 569,410, hasGivenName 568,852,
# bornOnDate 441,274), and the rest are a few triples smaller; RDF-3X's 2008
# dump matches 5 and the 2009-10-31 release is larger. The download page states
# "The Notation 3 version of YAGO is licensed under the GNU Free Documentation
# License" with no version, which lets the recipient pick any published one.
GCARE_YAGO = License(
    spdx="GFDL-1.1-or-later",
    ref=Ref(
        title="YAGO Downloads",
        authors=[
            Author("Fabian M. Suchanek"),
            Author("Gjergji Kasneci"),
            Author("Gerhard Weikum"),
        ],
        institution="Max Planck Institute for Informatics",
        year=2009,
        url=(
            "https://web.archive.org/web/20090417053715/"
            "http://www.mpi-inf.mpg.de/yago-naga/yago/downloads.html"
        ),
    ),
    date_accessed=_ACCESSED_2026_11_18,
)

# G-CARE's lubm80 graph: synthetic LUBM data for 80 universities, made by
# Lehigh's UBA 1.7 generator, whose source files are under GPL version 2 "or
# (at your option) any later version".
GCARE_LUBM80 = License(
    spdx="GPL-2.0-or-later",
    ref=Ref(
        title="UBA (Univ-Bench Artificial Data Generator) 1.7",
        authors=[Author("Yuanbo Guo"), Author("Zhengxiang Pan"), Author("Jeff Heflin")],
        institution="Lehigh University",
        year=2004,
        url="https://swat.cse.lehigh.edu/projects/lubm/uba1.7.zip",
    ),
    date_accessed=_ACCESSED_2026_11_18,
)

# G-CARE's aids graph: the 10,000 AIDS antiviral screen compounds used by
# QuickSI, from the National Cancer Institute's Developmental Therapeutics
# Program. A US government work, so in the public domain; SPDX has no
# identifier for that.
GCARE_AIDS = License(
    spdx="LicenseRef-Public-Domain",
    ref=Ref(
        title="Copyright and Reuse",
        authors=[Author("National Cancer Institute")],
        institution="National Institutes of Health",
        url="https://www.cancer.gov/policies/copyright-reuse",
    ),
    date_accessed=_ACCESSED_2026_11_18,
)

# G-CARE's human graph: Han's group's re-creation (Lee et al., PVLDB 6(2), 2012)
# of GADDI's human protein interaction network, labeled with Gene Ontology
# terms, which are CC BY 4.0. Neither paper names the interaction database.
GCARE_HUMAN = License(
    spdx="CC-BY-4.0",
    ref=Ref(
        title="GO Citation Policy and License",
        authors=[Author("Gene Ontology Consortium")],
        url="https://geneontology.org/docs/go-citation-policy/",
    ),
    date_accessed=_ACCESSED_2026_11_18,
)
