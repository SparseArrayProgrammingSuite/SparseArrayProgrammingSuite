"""
Name: Weighted Model Counting Algorithm
Author: Richard Wan
Email: rwan41@gatech.edu

Motivation:
Weighted Model Counting extends model counting by assigning
specific weights to the variables. This problem is
fundamental in probabilistic reasoning, combinatorial
design, etc.

Role of sparsity:
The relationship between variables and clauses can be sparse.
Each clause can contain only a tiny fraction of the
variables.

Implementation Reference:
Implementation for model counting provided by Willow Ahrens.
Weighted model counting extends this by factoring in weights
for clauses that make the formula true.

Data Generation:
SAT formulas are provided using the modified DIMACS CNF format.
found here:
https://mccompetition.org/assets/files/mccomp_format_25.pdf
Input strings can be provided manually.

Statement on the use of Generative AI: No generative AI was
used for the benchmark function itself. Generative AI might
have been used to construct tests. This statement was written
by hand.
"""


def benchmark_weighted_model_counting(xp, format):
    num_vars, clauses, weights = parse_format(format)
    einsum_input = clauses_to_einsum(clauses, num_vars)

    if einsum_input is None:
        total = 1.0
        for i in range(1, num_vars + 1):
            total *= weights[i] + weights[-i]
        return total

    args = {}
    args["B"] = xp.array([0, 1])
    for i in range(1, num_vars + 1):
        args[f"W{i}"] = xp.array([weights[-i], weights[i]])

    return xp.einsum(einsum_input, **args)


def parse_format(text):
    lines = [line.strip() for line in text.strip().split("\n")]

    weights = {}
    weight_lines = [line for line in lines if line.startswith("c p weight")]
    for line in weight_lines:
        parts = line.split()
        literal = int(parts[3])
        weights[literal] = parse_weight(parts[4])

    cleaned = [line for line in lines if not line.startswith("c") and line]
    num_vars = 0
    num_clauses = 0
    rest = []

    for i, line in enumerate(cleaned):
        if line.startswith("p cnf"):
            parts = line.split()
            num_vars = int(parts[2])
            num_clauses = int(parts[3])

            rest = " ".join(cleaned[i + 1 :]).split()
            break

    clauses = []
    current_clause = []
    idx = 0

    while len(clauses) < num_clauses and idx < len(rest):
        val = int(rest[idx])

        if val == 0:
            clauses.append(current_clause)
            current_clause = []
        else:
            current_clause.append(val)

        idx += 1

    # infer complementary weight if not given
    for lit in list(weights.keys()):
        if -lit not in weights:
            w = weights[lit]
            if 0 < w < 1:
                weights[-lit] = 1 - w

    # default weight to one if not given
    for lit in range(1, num_vars + 1):
        if lit not in weights:
            weights[lit] = 0.5
            weights[-lit] = 0.5

    return num_vars, clauses, weights


def parse_weight(s):
    s = s.strip()
    if "/" in s:
        num, den = s.split("/")
        return float(num) / float(den)
    return float(s)


def clauses_to_einsum(clauses, num_vars):
    if len(clauses) == 0:
        return None

    clause_strings = []
    for clause in clauses:
        literal_strings = []
        for val in clause:
            var_idx = abs(val)
            var_name = f"B[v{var_idx}]"

            if val < 0:
                literal_strings.append(f"not {var_name}")
            else:
                literal_strings.append(var_name)

        clause_str = "(" + " or ".join(literal_strings) + ")"
        clause_strings.append(clause_str)

    mask_str = "(" + " and ".join(clause_strings) + ")"

    # use bool expr as mask for the weights
    weight_strings = [f"W{i}[v{i}]" for i in range(1, num_vars + 1)]
    weights_str = " * ".join(weight_strings)

    return f"s[] += {mask_str} * {weights_str}"
