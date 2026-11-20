"""Licenses of the data that SAPS generators synthesize or download.

Each generator's ``license`` property returns one of these constants. A
`License` records the SPDX expression, where it was found, the day it was read
there, whether it allows redistribution, and notes on the evidence. ``None``
means the source was checked and no license for its data could be found; the
comment on each such constant says where it was checked.
A license file covering a repository that merely redistributes someone else's
data is not counted as a license for that data.
"""

from __future__ import annotations

import datetime

from saps.benchmark import Author, License, Ref

# Each license below was checked on one of these days.
_ACCESSED_2026_11_18 = datetime.date(2026, 11, 18)
_ACCESSED_2026_11_19 = datetime.date(2026, 11, 19)

SAPS = License(
    spdx="MIT",
    ref=Ref(
        title="SparseArrayProgrammingSuite LICENSE",
        authors=[Author("Willow Ahrens")],
        year=2025,
        url="https://github.com/willow-ahrens/SparseArrayProgrammingSuite/blob/main/LICENSE",
    ),
    date_accessed=_ACCESSED_2026_11_18,
    redistributable=True,
    notes=(
        "Data synthesized by SAPS's own generator code (random matrices, test cases, "
        "discretized PDEs, and the like)."
    ),
)

SUITESPARSE = License(
    spdx="CC-BY-4.0",
    ref=Ref(
        title="About the SuiteSparse Matrix Collection",
        authors=[Author("Timothy A. Davis")],
        institution="Texas A&M University",
        url="https://sparse.tamu.edu/about",
    ),
    date_accessed=_ACCESSED_2026_11_18,
    redistributable=True,
    notes=(
        "The SuiteSparse Matrix Collection, including the matrices SAPS takes from "
        "its SNAP, GAP, and LPnetlib groups, and the model reduction problems of "
        "its Oberwolfach, Rommes and Puri groups, which replace SLICOT's (see "
        'SLICOT_MODEL_REDUCTION). The About page states: "The matrices '
        'themselves are under the CC-BY 4.0 License."'
    ),
)

OGB_NODEPROP = License(
    spdx="ODC-By-1.0 AND LicenseRef-Amazon-Reviews-PDS AND CC0-1.0",
    ref=Ref(
        title="Node Property Prediction",
        authors=[Author("Open Graph Benchmark")],
        url="https://ogb.stanford.edu/docs/nodeprop/",
    ),
    date_accessed=_ACCESSED_2026_11_18,
    redistributable=False,
    notes=(
        "The OGB node property prediction datasets that SAPS uses, taken together. "
        "OGB licenses each separately on the same page, and each SAPS dataset "
        "carries its own license: OGBN_ARXIV, OGBN_PRODUCTS, or OGBN_PROTEINS. "
        "Not redistributable because ogbn-products is not."
    ),
)

OGBN_ARXIV = License(
    spdx="ODC-By-1.0",
    ref=Ref(
        title="Node Property Prediction",
        authors=[Author("Open Graph Benchmark")],
        url="https://ogb.stanford.edu/docs/nodeprop/",
    ),
    date_accessed=_ACCESSED_2026_11_18,
    redistributable=True,
    notes='OGB\'s ogbn-arxiv dataset, listed with "License: ODC-BY".',
)

OGBN_PRODUCTS = License(
    spdx="LicenseRef-Amazon-Reviews-PDS",
    ref=Ref(
        title="Node Property Prediction",
        authors=[Author("Open Graph Benchmark")],
        url="https://ogb.stanford.edu/docs/nodeprop/",
    ),
    date_accessed=_ACCESSED_2026_11_18,
    redistributable=False,
    notes=(
        'OGB\'s ogbn-products dataset, listed with "License: Amazon license", '
        "linking https://s3.amazonaws.com/amazon-reviews-pds/license.txt, which no "
        "longer resolves. The archived copy "
        "(https://web.archive.org/web/2020/https://s3.amazonaws.com/amazon-reviews-pds/license.txt)"
        ' grants a revocable license "for purposes of academic research" and states '
        '"You may not resell, republish, or make any commercial use of the Reviews '
        'Library or its contents".'
    ),
)

OGBN_PROTEINS = License(
    spdx="CC0-1.0",
    ref=Ref(
        title="Node Property Prediction",
        authors=[Author("Open Graph Benchmark")],
        url="https://ogb.stanford.edu/docs/nodeprop/",
    ),
    date_accessed=_ACCESSED_2026_11_18,
    redistributable=True,
    notes='OGB\'s ogbn-proteins dataset, listed with "License: CC-0".',
)

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
    redistributable=True,
    notes=(
        "QASMBench circuits, under a BSD-style license from Battelle Memorial "
        "Institute that has no SPDX identifier. The license file is unchanged between"
        " the commit the downloader pins and the default branch."
    ),
)

LTH_CONV2 = License(
    spdx="CC-BY-4.0 AND CC-BY-2.0",
    ref=Ref(
        title="CIFAR Conv-2 Lottery Ticket Hypothesis Benchmark Artifacts",
        authors=[Author("Ramya Polaki")],
        publisher="Zenodo",
        year=2026,
        url="https://zenodo.org/records/22650920",
    ),
    date_accessed=_ACCESSED_2026_11_18,
    redistributable=True,
    notes=(
        "The pruned Conv-2 network for the lottery-ticket benchmark, under CC-BY-4.0 "
        "in its Zenodo record, together with the PASS images it classifies, which "
        "are under PASS's license: CC-BY-2.0 photos in a CC-BY-4.0 dataset."
    ),
)

PASS = License(
    spdx="CC-BY-4.0 AND CC-BY-2.0",
    ref=Ref(
        title=(
            "PASS: An ImageNet replacement for self-supervised pretraining without "
            "humans"
        ),
        authors=[
            Author("Yuki M. Asano"),
            Author("Christian Rupprecht"),
            Author("Andrew Zisserman"),
            Author("Andrea Vedaldi"),
        ],
        institution="University of Oxford",
        url="https://www.robots.ox.ac.uk/~vgg/data/pass/",
    ),
    date_accessed=_ACCESSED_2026_11_19,
    redistributable=True,
    notes=(
        'PASS photos, from Zenodo record 6615455. The project page states "The PASS '
        "dataset is available to download for commercial/research purposes under a "
        'Creative Commons Attribution 4.0 International License" and "The whole '
        "dataset only contains CC-BY licensed images with full attribution "
        'information." Each photo keeps its own Flickr license: all 1,439,588 rows '
        'of the record\'s pass_metadata.csv give "Attribution License", taken here '
        "to be Flickr's license 4, which "
        "https://www.flickr.com/services/api/flickr.photos.licenses.getInfo.html "
        'now lists as "CC BY 2.0". That page no longer uses the name, so this is '
        "unconfirmed; Flickr's only other attribution-only license is CC BY 4.0. "
        "pass_metadata.csv names each photo's creator by the hash that names its "
        "file. The pass_images generator records the hashes of the photos it uses, "
        "which it center-crops and downsamples to 32x32."
    ),
)

# SLICOT's "Benchmark Examples for Model Reduction". Checked 2026-11-18: no
# license on https://www.slicot.org/20-site/126-benchmark-examples-for-model-reduction,
# in its archives, in the SLICOT/Benchmark-ModelReduction GitHub repository, in
# SLICOT Working Note 2002-2, or on MORwiki ("License: NA"). Its generators
# are de-listed in favor of SuiteSparse's model reduction problems.
SLICOT_MODEL_REDUCTION: License | None = None

# FROSTT tensors. Checked 2026-11-18 and 2026-11-20: FROSTT adds no data
# license of its own (the MIT licenses of
# https://github.com/frostt-tensor/frostt-tensor.github.io and of its parsing
# scripts, https://github.com/frostt-tensor/parse-tools, cover that code, and
# FROSTT issue #19 asking about tensor licenses is unresolved), so each tensor
# is licensed by its upstream source. The tensors with a license have their own
# constants below. The rest have none upstream:
# - darpa: MIT Lincoln Laboratory's 1998 DARPA intrusion detection data, via
#   HaTen2. No license for the dataset; the site-wide terms of
#   https://www.ll.mit.edu/ forbid copying or distributing "content", which may
#   not have been meant to cover datasets.
# - delicious_*, flickr_*: Uni Koblenz PINTS crawls; no terms on the PINTS or
#   TAGora pages.
# - enron: released by FERC and distributed by CMU and ISI as public record,
#   with no license or public domain dedication. FROSTT's parser reads Kaggle's
#   copy (wcukierski/enron-email-dataset), which lists only "Data files ©
#   Original Authors".
# - nell_*: CMU Read the Web; the knowledge base downloads state no license,
#   and the tensors may come from internal CMU data (unconfirmed).
# - nips: Chechik's extension of Roweis's NIPS data, which only asks for a
#   citation; the papers' copyright stays with their authors or publishers.
# - reddit_2015: Pushshift dumps, which have no license; under Reddit's terms
#   users keep copyright in their comments.
# - lbnl_network: LBNL/ICSI Enterprise Tracing, "© 2005 LBNL and ICSI" with no
#   license; the README that came with the traces is no longer retrievable.
# - vast_2015_mc1_*: VAST Challenge 2015 synthetic data; no license on the
#   archived pages, and the original hosts are gone.
# - toy: FROSTT's small example tensor, with no license.
FROSTT: License | None = None

FROSTT_MATMUL = License(
    spdx="LicenseRef-Public-Domain",
    ref=Ref(
        title="Matrix Multiplication",
        authors=[Author("FROSTT")],
        url="http://frostt.io/tensors/matmul/",
    ),
    date_accessed=datetime.date(2026, 11, 20),
    redistributable=True,
    notes=(
        "FROSTT's matmul_* tensors. No license is stated, but there is nothing to "
        "license: each is the matrix multiplication tensor for its dimensions, "
        "whose nonzeros are fixed by the definition of matrix multiplication, so "
        "it is a mathematical fact with no creative expression. FROSTT distributes "
        "the generator alongside the tensors "
        "(https://s3.us-east-2.amazonaws.com/frostt/frostt_data/matrix-multiplication/matmul_generator.m),"
        " a loop that sets T(N*(k-1)+n, K*(m-1)+k, M*(n-1)+m) = 1."
    ),
)

FROSTT_LANL2 = License(
    spdx="CC0-1.0",
    ref=Ref(
        title="Comprehensive, Multi-Source Cyber-Security Events",
        authors=[Author("Alexander D. Kent")],
        institution="Los Alamos National Laboratory",
        year=2015,
        url="https://csr.lanl.gov/data/cyber1/",
        doi="10.17021/1179829",
    ),
    date_accessed=_ACCESSED_2026_11_18,
    redistributable=True,
    notes=(
        "FROSTT's lanl2 tensor, from LANL's \"Comprehensive, Multi-Source "
        'Cyber-Security Events" (Kent 2015), whose page states: "To the extent '
        "possible under law, Los Alamos National Laboratory has waived all copyright "
        "and related or neighboring rights to Comprehensive, Multi-Source "
        'Cyber-Security Events", linking CC0 1.0.'
    ),
)

FROSTT_FB_M = License(
    spdx="CC-BY-2.5",
    ref=Ref(
        title="Freebase Data Dumps",
        authors=[Author("Google")],
        url="https://developers.google.com/freebase",
    ),
    date_accessed=_ACCESSED_2026_11_18,
    redistributable=True,
    notes=(
        "FROSTT's fb_m (Freebase Music) tensor, built by HaTen2 from the Freebase "
        'data dumps, which are "provided under the CC-BY license" linking CC BY 2.5. '
        "HaTen2 adds no license, only a request to cite it. Freebase's API terms "
        "(https://developers.google.com/freebase/terms) also apply."
    ),
)

FROSTT_PATENTS = License(
    spdx="LicenseRef-Public-Domain",
    ref=Ref(
        title="Terms of Use for USPTO Websites",
        authors=[Author("United States Patent and Trademark Office")],
        url="https://www.uspto.gov/terms-use-uspto-websites",
    ),
    date_accessed=_ACCESSED_2026_11_18,
    redistributable=True,
    notes=(
        "FROSTT's patents tensor. FROSTT gives no source, but it is almost certainly "
        "US patent text processed in Karypis's lab at the University of Minnesota "
        "(unconfirmed). The USPTO states that, subject to the exceptions in 37 CFR "
        '1.71(d) & (e) and 1.84(s), "the text and drawings of a patent are typically '
        'not subject to copyright restrictions".'
    ),
)

FROSTT_CHICAGO_CRIME = License(
    spdx="LicenseRef-Chicago-Data-Terms-of-Use",
    ref=Ref(
        title="Data Terms of Use",
        authors=[Author("City of Chicago")],
        url="https://www.chicago.gov/content/city/en/narr/foia/data_disclaimer.html",
    ),
    date_accessed=_ACCESSED_2026_11_18,
    redistributable=True,
    notes=(
        "FROSTT's chicago_crime_* tensors, from the Chicago Data Portal, whose "
        'license field says "See Terms of Use". Those terms are not an open license: '
        "they require a specific disclaimer, let the City revoke use, include an "
        "indemnity, and forbid trying to identify specific addresses. The terms "
        "contemplate distributing derivative applications, provided they carry the "
        "City's disclaimer, but the City \"may require a user of this data to "
        'terminate any and all display, distribution or other use" at any time.'
    ),
)

FROSTT_AMAZON_REVIEWS = License(
    spdx="LicenseRef-SNAP-Amazon-Research-Only",
    ref=Ref(
        title="Amazon review data README",
        authors=[Author("Julian McAuley"), Author("Jure Leskovec")],
        institution="Stanford University",
        url="https://snap.stanford.edu/data/amazon/amazon_readme.txt",
    ),
    date_accessed=_ACCESSED_2026_11_18,
    redistributable=False,
    notes=(
        "FROSTT's amazon_reviews tensor, from SNAP's web-Amazon data (McAuley and "
        'Leskovec), whose README states: "Amazon review data will be made available '
        '(for research purposes only) on request." Nothing grants permission to '
        "redistribute it."
    ),
)

FROSTT_UBER_PICKUPS = License(
    spdx="CC0-1.0",
    ref=Ref(
        title="Uber Pickups",
        authors=[Author("FROSTT")],
        url="http://frostt.io/tensors/uber-pickups/",
    ),
    date_accessed=_ACCESSED_2026_11_18,
    redistributable=True,
    notes=('FROSTT\'s uber_pickups tensor. Its page states "CC0: Public Domain".'),
)

# OpenML datasets 554 (mnist_784) and 40996 (Fashion-MNIST). Checked 2026-11-18
# and 2026-11-19: both report licence "Public" through
# https://www.openml.org/api/v1/json/data/, which is OpenML's default value for
# that field rather than a license. Each OpenML dataset carries its source's
# license instead: MNIST or FASHION_MNIST, below. SAPS de-listed OpenML's
# CIFAR-10 (40927), which has no license: checked 2026-11-19,
# https://www.cs.toronto.edu/~kriz/cifar.html states none and asks only for a
# citation, and Armory's "MIT" for it
# (https://armory.readthedocs.io/en/develop/dataset_licensing/) is under the
# copyright of Valay Shah, who is not a CIFAR-10 author.
OPENML: License | None = None

FASHION_MNIST = License(
    spdx="MIT",
    ref=Ref(
        title="Fashion-MNIST LICENSE",
        authors=[Author("Zalando SE")],
        year=2017,
        url="https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE",
    ),
    date_accessed=_ACCESSED_2026_11_19,
    redistributable=True,
    notes=(
        "Fashion-MNIST, from OpenML's copy (dataset 40996), which names Zalando "
        "Research as its source. Zalando's repository, which publishes the dataset, "
        'is under the MIT license, "Copyright © 2017 Zalando SE". Its README states '
        '"Fashion-MNIST is a dataset of Zalando\'s article images" and applies the '
        "license to the repository without saying separately that it covers the "
        "data files."
    ),
)

MNIST = License(
    spdx="CC-BY-SA-3.0",
    ref=Ref(
        title="MNIST digits classification dataset",
        authors=[Author("Keras")],
        url="https://keras.io/api/datasets/mnist/",
    ),
    date_accessed=_ACCESSED_2026_11_19,
    redistributable=True,
    notes=(
        "MNIST, from OpenML's copy (dataset 554, mnist_784). The Keras page states "
        '"Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset, which '
        "is a derivative work from original NIST datasets. MNIST dataset is made "
        "available under the terms of the Creative Commons Attribution-Share Alike "
        '3.0 license." PyMVPA (http://www.pymvpa.org/datadb/mnist.html) states the '
        "same in the same words. Neither gives a source, and MNIST's own page "
        "(http://yann.lecun.com/exdb/mnist/) could not be reached to confirm it "
        "(unconfirmed). This license covers only LeCun and Cortes's rights. No NIST "
        "license was found for Special Databases 1 and 3, which MNIST is built from: "
        "their nist.gov pages are gone, and Special Database 19's states none. If "
        "NIST holds them as Standard Reference Data, "
        'https://www.nist.gov/srd/public-law states "None of our SRD may be '
        'reproduced, stored in a retrieval system or transmitted" without prior '
        "permission."
    ),
)

GOODBOOKS = License(
    spdx="CC-BY-SA-4.0",
    ref=Ref(
        title="goodbooks-10k LICENSE",
        authors=[Author("Zygmunt Zając")],
        year=2017,
        url="https://github.com/zygmuntz/goodbooks-10k/blob/master/LICENSE",
    ),
    date_accessed=_ACCESSED_2026_11_19,
    redistributable=True,
    notes=(
        "goodbooks-10k's ratings.csv: 5,976,479 one-to-five star ratings of the "
        "10,000 most-rated Goodreads books by 53,424 users. The repository's "
        'LICENSE states "This work is licensed under the Creative Commons '
        'Attribution-ShareAlike 4.0 International License." The README does not '
        "say how the ratings were collected from Goodreads. goodbooks-10k replaces "
        "the Netflix Prize data, which has no license (see NETFLIX_PRIZE)."
    ),
)

# Netflix Prize data from Kaggle (netflix-inc/netflix-prize-data). Checked
# 2026-11-18: Kaggle lists "Other (specified in description)", but the
# description specifies no license. Its generators are de-listed.
NETFLIX_PRIZE: License | None = None

# Brain MRI images from Kaggle
# (navoneel/brain-mri-images-for-brain-tumor-detection). Checked 2026-11-18:
# Kaggle lists only "Data files © Original Authors".
KAGGLE_BRAIN_MRI: License | None = None

# Model counting competition instances from
# https://github.com/arijitsh/mccomp-test-instances. Checked 2026-11-18 and
# 2026-11-20: the repository has no license file or README, only the
# random_*.cnf files (10 variables, 20 clauses each) that MC 2025 co-chair
# Arijit Shaw committed on 2025-05-29. The MC 2025 page
# (https://mccompetition.org/2025/mc_description.html) links it as "A few test
# instances" and states no license; the MIT license of the site's repository
# (https://github.com/daajoe/mccompetition.org) covers the website, and MC 2025
# has no instance set on Zenodo. The MC 2020-2024 instance sets on Zenodo are
# CC-BY-4.0 (e.g. https://doi.org/10.5281/zenodo.14249068), but they do not
# include these test instances.
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
    redistributable=True,
    notes=(
        "G-CARE's yago graph: the YAGO 2009-w10-5 Notation 3 export (yago.n3, dated "
        "2009-03-06). 65 of the graph's 91 relation sizes equal that export's "
        "predicate counts exactly (e.g. hasFamilyName 569,410, hasGivenName 568,852, "
        "bornOnDate 441,274), and the rest are a few triples smaller; RDF-3X's 2008 "
        "dump matches 5 and the 2009-10-31 release is larger. The download page "
        'states "The Notation 3 version of YAGO is licensed under the GNU Free '
        'Documentation License" with no version, which lets the recipient pick any '
        "published one."
    ),
)

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
    redistributable=True,
    notes=(
        "G-CARE's lubm80 graph: synthetic LUBM data for 80 universities, made by "
        "Lehigh's UBA 1.7 generator, whose source files are under GPL version 2 \"or "
        '(at your option) any later version".'
    ),
)

GCARE_AIDS = License(
    spdx="LicenseRef-Public-Domain",
    ref=Ref(
        title="Copyright and Reuse",
        authors=[Author("National Cancer Institute")],
        institution="National Institutes of Health",
        url="https://www.cancer.gov/policies/copyright-reuse",
    ),
    date_accessed=_ACCESSED_2026_11_18,
    redistributable=True,
    notes=(
        "G-CARE's aids graph: the 10,000 AIDS antiviral screen compounds used by "
        "QuickSI, from the National Cancer Institute's Developmental Therapeutics "
        "Program. A US government work, so in the public domain; SPDX has no "
        "identifier for that."
    ),
)

GCARE_HUMAN = License(
    spdx="CC-BY-4.0",
    ref=Ref(
        title="GO Citation Policy and License",
        authors=[Author("Gene Ontology Consortium")],
        url="https://geneontology.org/docs/go-citation-policy/",
    ),
    date_accessed=_ACCESSED_2026_11_18,
    redistributable=True,
    notes=(
        "G-CARE's human graph: Han's group's re-creation (Lee et al., PVLDB 6(2), "
        "2012) of GADDI's human protein interaction network, labeled with Gene "
        "Ontology terms, which are CC BY 4.0. Neither paper names the interaction "
        "database."
    ),
)
