def benchmark_model_counting(xp, dimacs_text):
    clauses = parse_dimacs(dimacs_text)
    einsum_input = clauses_to_einsum(clauses)

    B = xp.array([0, 1])
    return xp.einsum(einsum_input, B=B)


def parse_dimacs(text):
    lines = [line.strip() for line in text.strip().split("\n")]
    cleaned = [line for line in lines if not line.startswith("c") and line]

    clauses = []

    num_clauses = 0
    rest = []

    for i, line in enumerate(cleaned):
        if line.startswith("p cnf"):
            parts = line.split()
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

    return clauses


def clauses_to_einsum(clauses):
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
