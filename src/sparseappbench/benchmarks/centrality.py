"""
Name: Betweenness Centrality

Author: Aadharsh Rajkumar

Email: arajkumar34@gatech.edu

What does this code do:
This code is based on the Brandes betweennness centrality algorithm. The
current code for the benchmark takes a two step approach. The first step
involves going layer by layer from each potential starting node to find
the total amount of shortest paths that lead to a node. So for example
4 -> 6 could have 3 diff shortest paths and 4 -> 2 could have only 1
shortest path. The second step is for tracing backwards to see how many
times a node appears in other shortest paths. The number of times this
node is in one of the shortest path divided by total shortest paths between
the two edge nodes gets added to the intermediate nodes bc score. This code
performs lazy calculations before computing at the end of iteration blocks.

Citation for reference implementation:
https://github.com/SparseApplicationBenchmark/SparseApplicationBenchmark/pull/48/files

Citation for importance of the problem:
Matta, J., Ercal, G. & Sinha, K. Comparing the speed and accuracy of
approaches to betweenness centrality approximation.
Comput Soc Netw 6, 2 (2019). https://doi.org/10.1186/s40649-019-0062-5

Statement on the use of Generative AI: No generative AI was used to construct
the benchmark function. This statement is written by hand.
"""


def betweenness_centrality(xp, A_binsparse):
    G = xp.lazy(xp.from_benchmark(A_binsparse))
    n = G.shape[0]
    bc_scores = xp.zeros((n,), dtype=float)

    for v in range(n):
        number_of_paths = xp.zeros((n,), dtype=float)
        self_dist = xp.zeros((n,), dtype=float)
        self_dist = self_dist + xp.array([1.0 if i == v else 0.0 for i in range(n)])
        number_of_paths = number_of_paths + self_dist

        neighbors = xp.array(G[v], dtype=float)
        layer_traversal = []
        depth = 0

        neighbors = xp.lazy(neighbors)
        node_count = xp.compute(xp.sum(neighbors))

        while node_count != 0:
            depth += 1

            layer_traversal.append(neighbors != 0)

            number_of_paths, neighbors = xp.lazy((number_of_paths, neighbors))

            number_of_paths = number_of_paths + neighbors

            not_neighbors = xp.equal(number_of_paths, 0)
            next_neighbors = xp.matmul(neighbors, G) * not_neighbors

            number_of_paths, next_neighbors, node_count = xp.compute(
                (number_of_paths, next_neighbors, xp.sum(next_neighbors))
            )

            neighbors = next_neighbors

        score_update = xp.zeros((n,), dtype=float)

        while depth >= 2:
            neighbors_layer = layer_traversal[depth - 1].astype(float)

            prev_layer = layer_traversal[depth - 2].astype(float)

            score_update, neighbors_layer, prev_layer, number_of_paths = xp.lazy(
                (score_update, neighbors_layer, prev_layer, number_of_paths)
            )

            denom = xp.maximum(number_of_paths, 1e-10)
            update_val = neighbors_layer * (1.0 + score_update) / denom

            update_val = xp.matmul(G, update_val)

            update_val = update_val * prev_layer * number_of_paths

            score_update, update_val = xp.compute((score_update, update_val))
            score_update = score_update + update_val

            depth -= 1

        bc_scores = bc_scores + score_update

    return xp.to_benchmark(bc_scores)
