"""Shared SNAP source catalog and prepared graphs for graph benchmarks."""

# ruff: noqa: E501

from typing import Any

import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import to_scipy

from saps.benchmark import (
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
    ShellBenchmark,
)
from saps.downloaders.snap import download_snap_dataset

# Domain classifications selected from ACM CCS 2012:
# https://dl.acm.org/pb-assets/dl_ccs/acm_ccs2012-1626988337597.xml
# SNAP group names are dataset categories, not literal CCS taxonomy entries.
_GROUP_CONCEPTS: dict[str, str] = {
    "Social networks": """<concept>
<concept_id>10003120.10003130.10003131.10003292</concept_id>
<concept_desc>Human-centered computing~Social networks</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Networks with ground-truth communities": """<concept>
<concept_id>10010147.10010257.10010258.10010260.10003697</concept_id>
<concept_desc>Computing methodologies~Cluster analysis</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Communication networks": """<concept>
<concept_id>10003120.10003130</concept_id>
<concept_desc>Human-centered computing~Collaborative and social computing</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Citation networks": """<concept>
<concept_id>10002951.10003317.10003365.10010851</concept_id>
<concept_desc>Information systems~Link and co-citation analysis</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Collaboration networks": """<concept>
<concept_id>10003120.10003130.10003131.10003292</concept_id>
<concept_desc>Human-centered computing~Social networks</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Web graphs": """<concept>
<concept_id>10003033.10003106.10003114.10003116</concept_id>
<concept_desc>Networks~World Wide Web (network structure)</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Product co-purchasing networks": """<concept>
<concept_id>10010405.10003550</concept_id>
<concept_desc>Applied computing~Electronic commerce</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Internet peer-to-peer networks": """<concept>
<concept_id>10003033.10003106.10003114.10003115</concept_id>
<concept_desc>Networks~Peer-to-peer networks</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Road networks": """<concept>
<concept_id>10002951.10003227.10003236</concept_id>
<concept_desc>Information systems~Spatial-temporal systems</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Autonomous systems graphs": """<concept>
<concept_id>10003033.10003083.10003090</concept_id>
<concept_desc>Networks~Network structure</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Signed networks": """<concept>
<concept_id>10002951.10003227.10003233.10003449</concept_id>
<concept_desc>Information systems~Reputation systems</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Location-based online social networks": """<concept>
<concept_id>10003033.10003106.10003114.10011730</concept_id>
<concept_desc>Networks~Online social networks</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Wikipedia networks, articles, and metadata": """<concept>
<concept_id>10003120.10003130.10003131.10003235</concept_id>
<concept_desc>Human-centered computing~Collaborative content creation</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Temporal networks": """<concept>
<concept_id>10002951.10002952.10002953.10010820.10010518</concept_id>
<concept_desc>Information systems~Temporal data</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "User actions": """<concept>
<concept_id>10003120.10003130.10011762</concept_id>
<concept_desc>Human-centered computing~Empirical studies in collaborative and social computing</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Memetracker and Twitter": """<concept>
<concept_id>10003033.10003106.10003114.10003118</concept_id>
<concept_desc>Networks~Social media networks</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Online communities": """<concept>
<concept_id>10003120.10003130.10003131.10003292</concept_id>
<concept_desc>Human-centered computing~Social networks</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Online reviews": """<concept>
<concept_id>10002951.10003227.10003233.10003449</concept_id>
<concept_desc>Information systems~Reputation systems</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Face-to-face communication networks": """<concept>
<concept_id>10003120.10003130</concept_id>
<concept_desc>Human-centered computing~Collaborative and social computing</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Graph classification datasets": """<concept>
<concept_id>10010147.10010257.10010258.10010259.10010263</concept_id>
<concept_desc>Computing methodologies~Supervised learning by classification</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Computer communication networks": """<concept>
<concept_id>10003033.10003083.10003090</concept_id>
<concept_desc>Networks~Network structure</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Cryptocurrency transactions": """<concept>
<concept_id>10010405.10003550</concept_id>
<concept_desc>Applied computing~Electronic commerce</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
    "Telecom networks": """<concept>
<concept_id>10010405.10010432.10010988</concept_id>
<concept_desc>Applied computing~Telecommunications</concept_desc>
<concept_significance>500</concept_significance>
</concept>""",
}


class SNAPDataset(Dataset):
    def __init__(
        self,
        name: str,
        groups: list[str],
        types: list[str],
        nodes: int | str | None,
        edges: int | str | None,
        description: str,
        *,
        pretty_name: str | None = None,
        communities: int | str | None = None,
        static_edges: int | str | None = None,
        items: int | str | None = None,
        graphs: int | str | None = None,
    ):
        self.groups = list(dict.fromkeys(groups))
        for category in self.groups:
            if category not in _GROUP_CONCEPTS:
                raise ValueError(f"Unknown SNAP group: {category!r}")
        self._name = name
        self.types = list(types)
        self._description = description
        self.nodes = nodes
        self.edges = edges
        self._pretty_name = pretty_name or name
        self.communities = communities
        self.static_edges = static_edges
        self.items = items
        self.graphs = graphs

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
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        concepts = dict.fromkeys(_GROUP_CONCEPTS[group] for group in self.groups)
        return "<ccs2012>" + "".join(concepts) + "</ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata,
            "types": self.types,
            "nodes": self.nodes,
            "edges": self.edges,
            "groups": self.groups,
            "communities": self.communities,
            "static_edges": self.static_edges,
            "items": self.items,
            "graphs": self.graphs,
        }


class SNAPSourceDataset(Dataset):
    """A shared SNAP graph paired with a reproducible source-selection seed."""

    def __init__(self, graph: SNAPDataset, seed: int):
        self.graph = graph
        self.seed = seed

    @property
    def name(self) -> str:
        return f"{self.graph.name}_seed{self.seed}"

    @property
    def pretty_name(self) -> str:
        return f"{self.graph.pretty_name} (source seed {self.seed})"

    @property
    def description(self) -> str:
        return self.graph.description

    @property
    def suites(self) -> list[str]:
        return self.graph.suites

    @property
    def concepts(self) -> str:
        return self.graph.concepts

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self.graph.metadata,
            **super().metadata,
            "graph": self.graph.name,
            "seed": self.seed,
        }


# One explicit entry per source. Additional group memberships and metadata
# from repeated catalog listings are merged into that source entry.
# Counts describe the published source, not the remapped prepared adjacency.
# fmt: off
# Columns: name, groups, types, nodes, edges, description.
_GRAPHS = [
    # Social networks
    SNAPDataset("facebook_combined", ["Social networks"], ["Undirected"], 4039, 88234, "Social circles from Facebook (anonymized)", pretty_name="ego-Facebook"),
    SNAPDataset("gplus_combined", ["Social networks"], ["Directed"], 107614, 13673453, "Social circles from Google+", pretty_name="ego-Gplus"),
    SNAPDataset("twitter_combined", ["Social networks"], ["Directed"], 81306, 1768149, "Social circles from Twitter", pretty_name="ego-Twitter"),
    SNAPDataset("soc-Epinions1", ["Social networks"], ["Directed"], 75879, 508837, "Who-trusts-whom network of Epinions.com"),
    SNAPDataset("soc-LiveJournal1", ["Social networks"], ["Directed"], 4847571, 68993773, "LiveJournal online social network"),
    SNAPDataset("soc-Pokec", ["Social networks"], ["Directed"], 1632803, 30622564, "Pokec online social network"),
    SNAPDataset("soc-Slashdot0811", ["Social networks"], ["Directed"], 77360, 905468, "Slashdot social network from November 2008"),
    SNAPDataset("soc-Slashdot0922", ["Social networks"], ["Directed"], 82168, 948464, "Slashdot social network from February 2009"),
    SNAPDataset("wiki-Vote", ["Social networks", "Wikipedia networks, articles, and metadata"], ["Directed"], 7115, 103689, "Wikipedia who-votes-on-whom network"),
    SNAPDataset("wiki-RfA", ["Social networks", "Signed networks", "Wikipedia networks, articles, and metadata"], ["Directed", "Signed"], 10835, 159388, "Wikipedia Requests for Adminship (with text)"),
    SNAPDataset("gemsec-Deezer", ["Social networks"], ["Undirected"], 143884, 846915, "Gemsec Deezer dataset"),
    SNAPDataset("gemsec-Facebook", ["Social networks"], ["Undirected"], 134833, 1380293, "Gemsec Facebook dataset"),
    SNAPDataset("soc-RedditHyperlinks", ["Social networks", "Signed networks", "Temporal networks", "Online communities"], ["Directed", "Signed", "Temporal", "Attributed", "Subreddit hyperlinks"], 55863, 858490, "Hyperlinks between subreddits on Reddit", static_edges=858490, items="858,490 links between 55,863 subreddits"),
    SNAPDataset("soc-sign-bitcoin-otc", ["Social networks", "Signed networks", "Temporal networks"], ["Weighted", "Signed", "Directed", "Temporal"], 5881, 35592, "Bitcoin OTC web of trust network"),
    SNAPDataset("soc-sign-bitcoin-alpha", ["Social networks", "Signed networks", "Temporal networks"], ["Weighted", "Signed", "Directed", "Temporal"], 3783, 24186, "Bitcoin Alpha web of trust network"),
    SNAPDataset("comm-f2f-Resistance", ["Social networks", "Communication networks", "Temporal networks", "Face-to-face communication networks"], ["Weighted", "Directed", "Temporal"], 451, 3126993, "Dynamic face-to-face interaction network between group of people"),
    SNAPDataset("musae-twitch", ["Social networks"], ["Undirected"], 34118, 429113, "Social networks of Twitch users."),
    SNAPDataset("musae-facebook", ["Social networks"], ["Undirected"], 22470, 171002, "Facebook page-page network with page names."),
    SNAPDataset("act-mooc", ["Social networks", "Temporal networks", "User actions"], ["Bipartite", "Directed", "Attributed", "Temporal"], 7143, 411749, "Student actions on a MOOC platform, with student drop-out binary labels."),
    SNAPDataset("musae-github", ["Social networks"], ["Undirected"], 37700, 289003, "Social network of Github developers."),
    SNAPDataset("feather-deezer-social", ["Social networks"], ["Undirected"], 28281, 92752, "Social network of Deezer users from Europe."),
    SNAPDataset("feather-lastfm-social", ["Social networks"], ["Undirected"], 7624, 27806, "Social network of LastFM users from Asia."),
    SNAPDataset("twitch-gamers", ["Social networks"], ["Undirected"], 168114, 6797557, "Social network of Twitch users."),
    SNAPDataset("congress-Twitter", ["Social networks"], ["Directed"], 475, 13289, "Twitter interaction network for the US Congress"),
    # Networks with ground-truth communities
    SNAPDataset("com-LiveJournal", ["Networks with ground-truth communities"], ["Undirected", "Communities"], 3997962, 34681189, "LiveJournal online social network", communities=287512),
    SNAPDataset("com-Friendster", ["Networks with ground-truth communities"], ["Undirected", "Communities"], 65608366, 1806067135, "Friendster online social network", communities=957154),
    SNAPDataset("com-Orkut", ["Networks with ground-truth communities"], ["Undirected", "Communities"], 3072441, 117185083, "Orkut online social network", communities=6288363),
    SNAPDataset("com-Youtube", ["Networks with ground-truth communities"], ["Undirected", "Communities"], 1134890, 2987624, "Youtube online social network", communities=8385),
    SNAPDataset("com-DBLP", ["Networks with ground-truth communities"], ["Undirected", "Communities"], 317080, 1049866, "DBLP collaboration network", communities=13477),
    SNAPDataset("com-Amazon", ["Networks with ground-truth communities"], ["Undirected", "Communities"], 334863, 925872, "Amazon product network", communities=75149),
    SNAPDataset("email-Eu-core", ["Networks with ground-truth communities"], ["Directed", "Communities"], 1005, 25571, "E-mail network", communities=42),
    SNAPDataset("wiki-topcats", ["Networks with ground-truth communities"], ["Directed", "Communities"], 1791489, 28511807, "Wikipedia hyperlinks", communities=17364),
    SNAPDataset("cisco-secure-workload", ["Networks with ground-truth communities", "Computer communication networks"], ["Directed", "Communities", "Attributed"], "between 86 and 278,739", "between 155 and 2,158,346", "22 graphs of communications between computers, 2 have ground truth groupings", communities="10 and 23", graphs=22),
    # Communication networks
    SNAPDataset("email-EuAll", ["Communication networks"], ["Directed"], 265214, 420045, "Email network from a EU research institution"),
    SNAPDataset("email-Enron", ["Communication networks"], ["Undirected"], 36692, 183831, "Email communication network from Enron"),
    SNAPDataset("wiki-Talk", ["Communication networks", "Wikipedia networks, articles, and metadata"], ["Directed"], 2394385, 5021410, "Wikipedia talk (communication) network"),
    # Citation networks
    SNAPDataset("cit-HepPh", ["Citation networks"], ["Directed", "Temporal", "Labeled"], 34546, 421578, "Arxiv High Energy Physics paper citation network"),
    SNAPDataset("cit-HepTh", ["Citation networks"], ["Directed", "Temporal", "Labeled"], 27770, 352807, "Arxiv High Energy Physics paper citation network"),
    SNAPDataset("cit-Patents", ["Citation networks"], ["Directed", "Temporal", "Labeled"], 3774768, 16518948, "Citation network among US Patents"),
    # Collaboration networks
    SNAPDataset("ca-AstroPh", ["Collaboration networks"], ["Undirected"], 18772, 198110, "Collaboration network of Arxiv Astro Physics"),
    SNAPDataset("ca-CondMat", ["Collaboration networks"], ["Undirected"], 23133, 93497, "Collaboration network of Arxiv Condensed Matter"),
    SNAPDataset("ca-GrQc", ["Collaboration networks"], ["Undirected"], 5242, 14496, "Collaboration network of Arxiv General Relativity"),
    SNAPDataset("ca-HepPh", ["Collaboration networks"], ["Undirected"], 12008, 118521, "Collaboration network of Arxiv High Energy Physics"),
    SNAPDataset("ca-HepTh", ["Collaboration networks"], ["Undirected"], 9877, 25998, "Collaboration network of Arxiv High Energy Physics Theory"),
    # Web graphs
    SNAPDataset("web-BerkStan", ["Web graphs"], ["Directed"], 685230, 7600595, "Web graph of Berkeley and Stanford"),
    SNAPDataset("web-Google", ["Web graphs"], ["Directed"], 875713, 5105039, "Web graph from Google"),
    SNAPDataset("web-NotreDame", ["Web graphs"], ["Directed"], 325729, 1497134, "Web graph of Notre Dame"),
    SNAPDataset("web-Stanford", ["Web graphs"], ["Directed"], 281903, 2312497, "Web graph of Stanford.edu"),
    # Product co-purchasing networks
    SNAPDataset("amazon0302", ["Product co-purchasing networks"], ["Directed"], 262111, 1234877, "Amazon product co-purchasing network from March 2 2003"),
    SNAPDataset("amazon0312", ["Product co-purchasing networks"], ["Directed"], 400727, 3200440, "Amazon product co-purchasing network from March 12 2003"),
    SNAPDataset("amazon0505", ["Product co-purchasing networks"], ["Directed"], 410236, 3356824, "Amazon product co-purchasing network from May 5 2003"),
    SNAPDataset("amazon0601", ["Product co-purchasing networks"], ["Directed"], 403394, 3387388, "Amazon product co-purchasing network from June 1 2003"),
    SNAPDataset("amazon-meta", ["Product co-purchasing networks"], ["Metadata"], 548552, 1788725, "Amazon product metadata: product info and all reviews on around 548,552 products."),
    # Internet peer-to-peer networks
    SNAPDataset("p2p-Gnutella04", ["Internet peer-to-peer networks"], ["Directed"], 10876, 39994, "Gnutella peer to peer network from August 4 2002"),
    SNAPDataset("p2p-Gnutella05", ["Internet peer-to-peer networks"], ["Directed"], 8846, 31839, "Gnutella peer to peer network from August 5 2002"),
    SNAPDataset("p2p-Gnutella06", ["Internet peer-to-peer networks"], ["Directed"], 8717, 31525, "Gnutella peer to peer network from August 6 2002"),
    SNAPDataset("p2p-Gnutella08", ["Internet peer-to-peer networks"], ["Directed"], 6301, 20777, "Gnutella peer to peer network from August 8 2002"),
    SNAPDataset("p2p-Gnutella09", ["Internet peer-to-peer networks"], ["Directed"], 8114, 26013, "Gnutella peer to peer network from August 9 2002"),
    SNAPDataset("p2p-Gnutella24", ["Internet peer-to-peer networks"], ["Directed"], 26518, 65369, "Gnutella peer to peer network from August 24 2002"),
    SNAPDataset("p2p-Gnutella25", ["Internet peer-to-peer networks"], ["Directed"], 22687, 54705, "Gnutella peer to peer network from August 25 2002"),
    SNAPDataset("p2p-Gnutella30", ["Internet peer-to-peer networks"], ["Directed"], 36682, 88328, "Gnutella peer to peer network from August 30 2002"),
    SNAPDataset("p2p-Gnutella31", ["Internet peer-to-peer networks"], ["Directed"], 62586, 147892, "Gnutella peer to peer network from August 31 2002"),
    # Road networks
    SNAPDataset("roadNet-CA", ["Road networks"], ["Undirected"], 1965206, 2766607, "Road network of California"),
    SNAPDataset("roadNet-PA", ["Road networks"], ["Undirected"], 1088092, 1541898, "Road network of Pennsylvania"),
    SNAPDataset("roadNet-TX", ["Road networks"], ["Undirected"], 1379917, 1921660, "Road network of Texas"),
    # Autonomous systems graphs
    SNAPDataset("as-733", ["Autonomous systems graphs"], ["Undirected"], "103-6,474", "243-13,233", "733 daily instances(graphs) from November 8 1997 to January 2 2000"),
    SNAPDataset("as-Skitter", ["Autonomous systems graphs"], ["Undirected"], 1696415, 11095298, "Internet topology graph, from traceroutes run daily in 2005"),
    SNAPDataset("as-Caida", ["Autonomous systems graphs"], ["Directed"], "8,020-26,475", "36,406-106,762", "The CAIDA AS Relationships Datasets, from January 2004 to November 2007"),
    SNAPDataset("Oregon-1", ["Autonomous systems graphs"], ["Undirected"], "10,670-11,174", "22,002-23,409", "AS peering information inferred from Oregon route-views between March 31 and May 26 2001"),
    SNAPDataset("Oregon-2", ["Autonomous systems graphs"], ["Undirected"], "10,900-11,461", "31,180-32,730", "AS peering information inferred from Oregon route-views between March 31 and May 26 2001"),
    # Signed networks
    SNAPDataset("soc-sign-epinions", ["Signed networks"], ["Directed"], 131828, 841372, "Epinions signed social network"),
    SNAPDataset("wiki-Elec", ["Signed networks", "Wikipedia networks, articles, and metadata"], ["Directed", "Bipartite"], "~7,000", "~100,000", "Wikipedia adminship election data"),
    SNAPDataset("soc-sign-Slashdot081106", ["Signed networks"], ["Directed"], 77357, 516575, "Slashdot Zoo signed social network from November 6 2008"),
    SNAPDataset("soc-sign-Slashdot090216", ["Signed networks"], ["Directed"], 81871, 545671, "Slashdot Zoo signed social network from February 16 2009"),
    SNAPDataset("soc-sign-Slashdot090221", ["Signed networks"], ["Directed"], 82144, 549202, "Slashdot Zoo signed social network from February 21 2009"),
    # Location-based online social networks
    SNAPDataset("loc-Gowalla", ["Location-based online social networks"], ["Undirected", "Geo-Location"], 196591, 950327, "Gowalla location based online social network"),
    SNAPDataset("loc-Brightkite", ["Location-based online social networks"], ["Undirected", "Geo-Location"], 58228, 214078, "Brightkite location based online social network"),
    # Wikipedia networks, articles, and metadata
    SNAPDataset("wikispeedia", ["Wikipedia networks, articles, and metadata"], ["Navigation paths"], 4604, 119882, "Navigation paths on the Wikipedia hyperlink network, collected by the human-computation game Wikispeedia"),
    SNAPDataset("wiki-meta", ["Wikipedia networks, articles, and metadata"], ["Edits"], "2.3M users, 3.5M pages", "250M edits", "Complete Wikipedia edit history (who edited what page)"),
    SNAPDataset("wiki-hoaxes", ["Wikipedia networks, articles, and metadata"], ["Wiki markup"], 64, None, "Public Wikipedia hoaxes"),
    SNAPDataset("musae-wiki", ["Wikipedia networks, articles, and metadata"], ["Undirected"], 19109, 400832, "Wikipedia page network with traffic information."),
    # Temporal networks
    SNAPDataset("sx-stackoverflow", ["Temporal networks"], ["Directed", "Temporal"], 2601977, 63497050, "Comments, questions, and answers on Stack Overflow", static_edges=36233450),
    SNAPDataset("sx-mathoverflow", ["Temporal networks"], ["Directed", "Temporal"], 24818, 506550, "Comments, questions, and answers on Math Overflow", static_edges=239978),
    SNAPDataset("sx-superuser", ["Temporal networks"], ["Directed", "Temporal"], 194085, 1443339, "Comments, questions, and answers on Super User", static_edges=924886),
    SNAPDataset("sx-askubuntu", ["Temporal networks"], ["Directed", "Temporal"], 159316, 964437, "Comments, questions, and answers on Ask Ubuntu", static_edges=596933),
    SNAPDataset("wiki-talk-temporal", ["Temporal networks"], ["Directed", "Temporal"], 1140149, 7833140, "Users editing talk pages on Wikipedia", static_edges=3309592),
    SNAPDataset("email-Eu-core-temporal", ["Temporal networks"], ["Directed", "Temporal"], 986, 332334, "E-mails between users at a research institution", static_edges=24929),
    SNAPDataset("CollegeMsg", ["Temporal networks"], ["Directed", "Temporal"], 1899, 20296, "Messages on a Facebook-like platform at UC-Irvine", static_edges=59835),
    # Memetracker and Twitter
    SNAPDataset("twitter7", ["Memetracker and Twitter"], ["Tweets"], "17,069,982 users", "476,553,560 tweets", "A collection of 476 million tweets collected between June-Dec 2009"),
    SNAPDataset("memetracker9", ["Memetracker and Twitter"], ["Memes"], "96 million", "418 million links", "Memetracker phrases and hyperlinks between 96 million blog posts from Aug 2008 to Apr 2009"),
    SNAPDataset("ksc-time-series", ["Memetracker and Twitter"], ["Time Series"], 2000, None, "Time series of volume of 1,000 most popular Memetracker phrases and 1,000 most popular Twitter hashtags"),
    SNAPDataset("higgs-twitter", ["Memetracker and Twitter"], ["Tweets"], 456631, 14855875, "Spreading processes of the announcement of the discovery of a new particle with the features of the Higgs boson on 4th July 2012."),
    # Online communities
    SNAPDataset("web-RedditEmbeddings", ["Online communities"], ["Reddit Embeddings"], None, None, "Embeddings of users and subreddits", items="118,381 users and 51,278 subreddits"),
    SNAPDataset("web-RedditPizzaRequests", ["Online communities"], ["Reddit requests"], None, None, "Textual requests for pizza with outcome labels", items="5,671 submissions"),
    SNAPDataset("web-Reddit", ["Online communities"], ["Reddit submissions"], None, None, "Resubmitted content on reddit.com", items="132,308 submissions"),
    SNAPDataset("web-flickr", ["Online communities"], ["Images"], None, None, "Images sharing common metadata on Flickr", items="2,316,948 related images"),
    # Online reviews
    SNAPDataset("web-BeerAdvocate", ["Online reviews"], ["Beer reviews"], None, None, "Beer reviews from BeerAdvocate", items="1,586,259 beer reviews"),
    SNAPDataset("web-RateBeer", ["Online reviews"], ["Beer reviews"], None, None, "Beer reviews from RateBeer", items="2,924,127 beer reviews"),
    SNAPDataset("web-CellarTracker", ["Online reviews"], ["Wine reviews"], None, None, "Wine reviews from CellarTracker", items="2,025,995 wine reviews"),
    SNAPDataset("web-Amazon", ["Online reviews"], ["Amazon reviews (all categories)"], None, None, "Reviews from Amazon", items="34,686,770 product reviews"),
    SNAPDataset("web-FineFoods", ["Online reviews"], ["Food reviews"], None, None, "Food reviews from Amazon", items="568,454 food reviews"),
    SNAPDataset("web-Movies", ["Online reviews"], ["Movie reviews"], None, None, "Movie reviews from Amazon", items="7,911,684 movie reviews"),
    # Graph classification datasets
    SNAPDataset("Deezer Ego-nets", ["Graph classification datasets"], ["Undirected", "Unattributed"], None, None, "Ego-nets of European Deezer users", graphs=9629),
    SNAPDataset("GitHub Stargazers", ["Graph classification datasets"], ["Undirected", "Unattributed"], None, None, "Communities of developers who starred repositories", graphs=12725),
    SNAPDataset("Reddit Threads", ["Graph classification datasets"], ["Undirected", "Unattributed"], None, None, "Reddit discussion and non-discussion based threads", graphs=203088),
    SNAPDataset("Ego-Nets", ["Graph classification datasets"], ["Undirected", "Unattributed"], None, None, "Ego-Nets of Twitch users in the partnership program", graphs=127094),
    # Cryptocurrency transactions
    SNAPDataset("ERC20-stablecoins", ["Cryptocurrency transactions"], ["Directed", "Attributed"], None, None, "Transaction data of the top five stablecoins in market cap (USDT, USDC, DAI, UST, PAX) and WLUNA", graphs=6),
    SNAPDataset("ethereum-exchanges", ["Cryptocurrency transactions"], ["Directed", "Attributed"], None, None, "Token (asset) networks from the Ethereum blockchain", graphs=28),
    # Telecom networks
    SNAPDataset("telecom-graph", ["Telecom networks"], ["Directed", "Attributed"], None, None, "Relationships between users, packages, apps, and cells in a telecom network", graphs=1),
    # Temporal networks
    SNAPDataset("email-Eu-core-temporal-Dept1", ["Temporal networks"], ["Directed", "Temporal"], 309, 61046, "E-mails between members of Department 1 at a European research institution.", static_edges=3031),
    SNAPDataset("email-Eu-core-temporal-Dept2", ["Temporal networks"], ["Directed", "Temporal"], 162, 46772, "E-mails between members of Department 2 at a European research institution.", static_edges=1772),
    SNAPDataset("email-Eu-core-temporal-Dept3", ["Temporal networks"], ["Directed", "Temporal"], 89, 12216, "E-mails between members of Department 3 at a European research institution.", static_edges=1506),
    SNAPDataset("email-Eu-core-temporal-Dept4", ["Temporal networks"], ["Directed", "Temporal"], 142, 48141, "E-mails between members of Department 4 at a European research institution.", static_edges=1375),
]
# fmt: on


class SNAPGraphGenerator(Generator[SNAPDataset]):
    @property
    def name(self) -> str:
        return "snap_graph"

    @property
    def pretty_name(self) -> str:
        return "Stanford Network Analysis Project Graphs"

    @property
    def description(self) -> str:
        return "Shared sparse SNAP adjacency matrices and original node IDs."

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
        return []

    @property
    def ai_disclosure(self) -> str:
        return "This shell generator was written with assistance from OpenAI Codex."

    @property
    def motivation(self) -> str:
        return "Prepare each SNAP graph once for reuse across graph benchmarks."

    @property
    def datasets(self) -> list[SNAPDataset]:
        return _GRAPHS

    def generate(self, dataset: SNAPDataset) -> DataInstance:
        inputs, meta = download_snap_dataset(
            dataset.name, data_dir=self.backend.cache_dir / "snap"
        )
        return DataInstance(inputs=inputs, meta=meta)


class SNAPGraphBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return SNAPGraphGenerator()


def fetch_snap_graph(name: str) -> DataInstance:
    """Read a declared graph's adjacency and original node IDs from prepared storage."""
    generator = SNAPGraphGenerator()
    dataset = next((d for d in generator.datasets if d.name == name), None)
    if dataset is None:
        raise ValueError(
            f"Dataset {name!r} is not listed in SNAPGraphGenerator.datasets. "
            "Add it to the shell dataset list before using it."
        )
    return generator.cached_generate(dataset)


def select_source_vertices(
    graph: BinsparseTensor, count: int = 1, *, seed: int = 0
) -> np.ndarray:
    """Sample source vertices from nonzero adjacency edges, with replacement.

    Each directed edge (u, v) contributes u as a candidate, so vertices are
    sampled in proportion to their outgoing edge counts. Self-loops count as
    edges. Returned IDs index the adjacency matrix, not the original SNAP IDs.
    The seed is local and does not change NumPy's global random state.
    """
    if count < 1:
        raise ValueError("Source vertex count must be positive.")
    edges = to_scipy(graph).tocoo(copy=True)
    if edges.shape[0] != edges.shape[1]:
        raise ValueError("Source selection requires a square adjacency matrix.")
    edges.sum_duplicates()
    rows = edges.row[edges.data != 0]
    if rows.size == 0:
        raise ValueError(
            "Cannot select source vertices from a graph without nonzero edges."
        )
    rng = np.random.default_rng(seed)
    return rows[rng.integers(rows.size, size=count)].astype(np.int64, copy=False)


def with_source_vertex(raw: DataInstance, *, seed: int) -> DataInstance:
    """Attach a seeded source without changing the shared shell metadata."""
    src = int(select_source_vertices(raw.inputs[0], seed=seed)[0])
    return DataInstance(inputs=raw.inputs, meta={**raw.meta, "src": src, "seed": seed})


def fetch_snap_source_graph(dataset: SNAPSourceDataset) -> DataInstance:
    return with_source_vertex(fetch_snap_graph(dataset.graph.name), seed=dataset.seed)
