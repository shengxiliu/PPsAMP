#!/usr/bin/env python3
"""
Sequence deduplication by greedy clustering at 40% similarity.

Custom Python implementation, NOT the official CD-HIT tool.
- Input  : dataset/new_data/all_combined.fasta   (train1.csv + test.csv + val1.csv -> 17,238)
- Output : dataset/new_data/all_dedup.fasta  (7,522)
           dataset/new_data/all_dedup.clstr
           dataset/new_data/{train,test,val}_dedup.{fasta,csv}
"""

from collections import defaultdict
from pathlib import Path

BASE_DIR = Path("/home/wangjingyuan/lsx/交接材料/模型/dataset/new_data")


def calc_identity(seq1, seq2):
    """Position-wise identity normalised by the shorter sequence (no gap alignment)."""
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return 0.0
    identical = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return identical / min_len


def read_fasta(path):
    seqs, cur_id, cur_seq = [], None, ""
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if cur_id and cur_seq:
                seqs.append((cur_id, cur_seq))
            cur_id, cur_seq = line[1:], ""
        else:
            cur_seq += line
    if cur_id and cur_seq:
        seqs.append((cur_id, cur_seq))
    return seqs


def run_dedup(fasta_in, fasta_out, cluster_out, threshold=0.40):
    sequences = read_fasta(fasta_in)
    print(f"Read {len(sequences)} sequences")

    # Sort by decreasing length; the longest sequence becomes the first representative
    sequences.sort(key=lambda x: len(x[1]), reverse=True)

    clusters = []   # [(rep_id, rep_seq, [(member_id, member_seq), ...]), ...]
    reps = []

    for seq_id, seq in sequences:
        found = False
        for rep_idx, (rep_id, rep_seq) in enumerate(reps):
            if calc_identity(seq, rep_seq) >= threshold:
                clusters[rep_idx][2].append((seq_id, seq))
                found = True
                break
        if not found:
            clusters.append((seq_id, seq, []))
            reps.append((seq_id, seq))

    print(f"Clustering complete. {len(clusters)} clusters.")

    with open(fasta_out, "w") as f:
        for rep_id, rep_seq, _ in clusters:
            f.write(f">{rep_id}\n{rep_seq}\n")

    with open(cluster_out, "w") as f:
        for i, (rep_id, rep_seq, members) in enumerate(clusters):
            f.write(f">Cluster {i}\n")
            f.write(f"0\t{len(rep_seq)}aa, >{rep_id}... *\n")
            for mem_id, mem_seq in members:
                f.write(f"1\t{len(mem_seq)}aa, >{mem_id}... at +/{calc_identity(mem_seq, rep_seq)*100:.1f}%\n")

    label_counts = defaultdict(lambda: defaultdict(int))
    for rep_id, rep_seq, members in clusters:
        label_counts[rep_id.split("|label=")[-1]]["rep"] += 1
        for mem_id, _ in members:
            label_counts[rep_id.split("|label=")[-1]]["mem"] += 1

    print("\n=== Representative label distribution ===")
    for lbl in sorted(label_counts.keys()):
        print(f"  label={lbl}: reps={label_counts[lbl]['rep']}, non-rep members={label_counts[lbl]['mem']}")

    return clusters


def split_by_source(clusters):
    splits = {"train": [], "test": [], "val": []}
    for rep_id, rep_seq, _ in clusters:
        source = rep_id.split("_")[0]
        label = rep_id.split("|label=")[-1]
        splits[source].append((rep_id, rep_seq, label))

    for src, items in splits.items():
        with open(BASE_DIR / f"{src}_dedup.fasta", "w") as f:
            for rep_id, rep_seq, _ in items:
                f.write(f">{rep_id}\n{rep_seq}\n")
        with open(BASE_DIR / f"{src}_dedup.csv", "w") as f:
            f.write("Sequence,Label\n")
            for _, rep_seq, label in items:
                f.write(f"{rep_seq},{label}\n")
        print(f"{src}: {len(items)} representatives")


def main():
    clusters = run_dedup(
        BASE_DIR / "all_combined.fasta",
        BASE_DIR / "all_dedup.fasta",
        BASE_DIR / "all_dedup.clstr",
        threshold=0.40,
    )
    split_by_source(clusters)


if __name__ == "__main__":
    main()
