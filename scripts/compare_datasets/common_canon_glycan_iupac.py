import re, pandas as pd, sys
from pathlib import Path

import argparse

sys.setrecursionlimit(100000)
TOK = re.compile(r"\[|\]|\([^()]*\)|[^()\[\]]+")



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_custom", type=str, default="data/pretraining_atom_full_glyles_1_2_2_parsable.csv")
    parser.add_argument("--data_glycanAA", type=str, default="data/glycan_pretrain.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/compare_datasets/common_canon_glycan_iupac")
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def parse_chain(toks, i):
    """Retourne (noeud_racine, lien_final, i) ; noeud=(nom, [(lien,e nfant)...])"""
    prev = None
    branches = []
    link = None

    while i < len(toks):
        t = toks[i]
        if t == "[":
            b, bl, i = parse_chain(toks, i + 1)      # s'arrête sur ']'
            branches.append((bl, b))
            continue
        if t == "]":
            return prev,link, i + 1
        if t.startswith("("):
            link = t[1: -1]
            i += 1
            continue

        # résidu
        children = list(branches)
        branches = []
        if prev is not None:
            children.append((link, prev))
        prev = (t, children)
        link = None
        i += 1

    return prev, link, i

def canon(n):
    name, ch = n
    parts = sorted(f"{l}:{canon(c)}" for l, c in ch)
    return name + ("{"+"|".join(parts)+"}" if parts else "")

def norm(s):
    s = s.strip()
    toks = TOK.findall(s)
    root, _, _ = parse_chain(toks, 0)
    return canon(root)

def strip_reducing(s):                 # enlève '(b1-' ou '(a1-3' final éventuel
    return re.sub(r"\([^()]*\)?$","",s) if re.search(r"\([^()]*$",s) else s


if __name__ == "__main__":
    args = get_args()
    data_custom = pd.read_csv(args.data_custom)
    data_glycanAA = pd.read_csv(args.data_glycanAA)["IUPAC Condensed"]

    iupac_custom = data_custom["iupac"].tolist()
    iupac_glycanAA = data_glycanAA.tolist()
    # iupac_custom = [strip_reducing(x) for x in iupac_custom]
    # iupac_glycanAA = [strip_reducing(x) for x in iupac_glycanAA]

    data_glycanAA_df = pd.DataFrame({"iupac" : data_glycanAA.tolist()})
    idx1 = data_custom.index[data_custom["iupac"].isin(data_glycanAA_df["iupac"])]
    idx2 = data_glycanAA_df.index[data_glycanAA_df["iupac"].isin(data_custom["iupac"])]

    # Export des glycans canoniques communs aux deux datasets
    data_custom.loc[idx1].to_csv(args.output_dir / "common_glycans_iupac.csv", index=False)

    # 1) comparaison textuelle exacte (après retrait du lien terminal de B)
    set_iupac_custom = set(iupac_custom)
    set_iupac_glycanAA = set(iupac_glycanAA)

    # 2) forme canonique (ordre des branches ignoré)
    canonic_iupac_custom = [norm(x) for x in iupac_custom]
    canonic_iupac_glycanAA = [norm(x) for x in iupac_glycanAA]
    set_canonic_iupac_custom = set(canonic_iupac_custom)
    set_canonic_iupac_glycanAA = set(canonic_iupac_glycanAA)

    pd.DataFrame({"iupac": iupac_custom, "canon": canonic_iupac_custom}).to_csv(f"{args.output_dir}/data_custom.csv", index=False)
    pd.DataFrame({"iupac": iupac_glycanAA, "canon": canonic_iupac_glycanAA}).to_csv(f"{args.output_dir}/data_glycanAA.csv", index=False)
    pd.DataFrame({"canon": list(set_canonic_iupac_custom & set_canonic_iupac_glycanAA)}).to_csv(f"{args.output_dir}/data_common_canon_glycans.csv", index=False)
    pd.DataFrame({"canon": list(set_canonic_iupac_custom - set_canonic_iupac_glycanAA)}).to_csv(f"{args.output_dir}/data_exclusive_canon_glycans_custom.csv", index=False)
    pd.DataFrame({"canon": list(set_canonic_iupac_glycanAA - set_canonic_iupac_custom)}).to_csv(f"{args.output_dir}/data_exclusive_canon_glycans_glycanAA.csv", index=False)

    print(f"Dataset_Custom: {len(set_iupac_custom)} IUPAC et {len(set_canonic_iupac_custom)} IUPAC uniques (sans compter les permutations)")
    print(f"Dataset_GlycanAA: {len(set_iupac_glycanAA)} IUPAC et {len(set_canonic_iupac_glycanAA)} IUPAC uniques (sans compter les permutations)")
    print()

    print(f"Nombre de glycans communs aux deux datasets: {len(set_iupac_custom & set_iupac_glycanAA)}")
    print(f"Nombre de glycans du dataset Custom qui sont absents du dataset GlycanAA: {len(set_iupac_custom - set_iupac_glycanAA)} soit {round(100 * len(set_iupac_custom - set_iupac_glycanAA) / len(set_iupac_custom), 2)}%")
    print(f"Nombre de glycans du dataset GlycanAA qui sont absents du dataset Custom: {len(set_iupac_glycanAA - set_iupac_custom)} soit {round(100 * len(set_iupac_glycanAA - set_iupac_custom) / len(set_iupac_glycanAA), 2)}%")
    print(f"{round(100 * len(set_iupac_custom & set_iupac_glycanAA) / len(set_iupac_custom), 2)}% de glycans du dataset Custom présents dans le dataset GlycanAA")
    print(f"{round(100 * len(set_iupac_custom & set_iupac_glycanAA) / len(set_iupac_glycanAA), 2)}% de glycans du dataset GlycanAA présents dans le dataset Custom")
    print()

    print(f"Nombre de glycans canoniques communs aux deux datasets: {len(set_canonic_iupac_custom & set_canonic_iupac_glycanAA)}")
    print(f"Nombre de glycans canoniques du dataset Custom qui sont absents du dataset GlycanAA: {len(set_canonic_iupac_custom - set_canonic_iupac_glycanAA)} soit {round(100 * len(set_canonic_iupac_custom - set_canonic_iupac_glycanAA) / len(set_canonic_iupac_custom), 2)}%")
    print(f"Nombre de glycans canoniques du dataset GlycanAA qui sont absents du dataset Custom: {len(set_canonic_iupac_glycanAA - set_canonic_iupac_custom)} soit {round(100 * len(set_canonic_iupac_glycanAA - set_canonic_iupac_custom) / len(set_canonic_iupac_glycanAA), 2)}%")
    print(f"{round(100 * len(set_canonic_iupac_custom & set_canonic_iupac_glycanAA) / len(set_canonic_iupac_custom), 2)}% de glycans (sans les permutations) du dataset Custom présents dans le dataset GlycanAA")
    print(f"{round(100 * len(set_canonic_iupac_custom & set_canonic_iupac_glycanAA) / len(set_canonic_iupac_glycanAA), 2)}% de glycans (sans les permutations) du dataset GlycanAA présents dans le dataset Custom")
    print()

    print(f"Indice de Jaccard: {round(len(set_iupac_custom & set_iupac_glycanAA) / len(set_iupac_custom | set_iupac_glycanAA), 2)}")
    print(f"Indice de Jaccard sans les permutations: {round(len(set_canonic_iupac_custom & set_canonic_iupac_glycanAA) / len(set_canonic_iupac_custom | set_canonic_iupac_glycanAA), 2)}")

