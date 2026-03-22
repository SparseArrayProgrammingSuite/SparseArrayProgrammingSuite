import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Search for matrices with b files")
    parser.add_argument(
        "--filepath",
        type=str,
        help="Path to the MM folder containing matrix files",
    )
    args = parser.parse_args()
    pairs = find_matrices_with_b(args.filepath)
    for folder, name in pairs:
        print(f"Matrix with b found: {name} in folder {folder}")


def find_matrices_with_b(folder):
    entries = os.listdir(folder)
    files = [f for f in entries if f.endswith(".mtx")]
    base_names = set()

    for f in files:
        if f.endswith("_b.mtx"):
            base_names.add(f.replace("_b.mtx", ""))

    matching = []
    for name in base_names:
        a_file = os.path.join(folder, f"{name}.mtx")
        b_file = os.path.join(folder, f"{name}_b.mtx")
        if os.path.exists(a_file) and os.path.exists(b_file):
            matching.append((folder, name))

    for entry in entries:
        subpath = os.path.join(folder, entry)
        if os.path.isdir(subpath):
            sub_matches = find_matrices_with_b(subpath)
            matching.extend(sub_matches)
    return matching


if __name__ == "__main__":
    main()
