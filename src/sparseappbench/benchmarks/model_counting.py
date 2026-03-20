"""
Name: Model Counting Algorithm
Author: Richard Wan
Email: rwan41@gatech.edu

Motivation:
Model Counting is used to determine the total number of
satisfying assignments for a SAT problem. This problem is
fundamental in probabilistic reasoning, combinatorial
design, etc.

Role of sparsity:
The relationship between variables and clauses can be sparse.
Each clause can contain only a tiny fraction of the
variables.

Implementation Reference:
Implementation provided by Willow Ahrens

Data Generation:
SAT formulas are provided using the DIMACS CNF format. Input
strings can be provided manually.

Statement on the use of Generative AI: No generative AI was
used for the benchmark function itself. Generative AI might
have been used to construct tests. This statement was written
by hand.
"""


def benchmark_model_counting(xp, dimacs_text):
    num_vars, clauses = parse_dimacs(dimacs_text)
    einsum_input = clauses_to_einsum(clauses)

    if einsum_input is None:
        return 2**num_vars

    B = xp.array([0, 1])
    return xp.einsum(einsum_input, B=B)


def parse_dimacs(text):
    lines = [line.strip() for line in text.strip().split("\n")]
    cleaned = [line for line in lines if not line.startswith("c") and line]

    clauses = []

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

    return num_vars, clauses


def clauses_to_einsum(clauses):
    if len(clauses) == 0:
        print("IS  NON sd as dsa dasdsaE")
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

    full_str = " and ".join(clause_strings)

    return f"s[] += {full_str}"
