import argparse
import pandas as pd
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--common_iupac", type=str, default="outputs/compare_datasets/common_canon_glycan_iupac/common_glycans_iupac.csv")
    parser.add_argument("--common_wurcs", type=str, default="outputs/compare_datasets/common_canon_glycan_wurcs/common_glycans_wurcs.csv")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    data_common_iupac = pd.read_csv(args.common_iupac)
    data_common_wurcs = pd.read_csv(args.common_wurcs)
    idx_iupac = set(data_common_iupac["source_row_id"].tolist())
    idx_wurcs = set(data_common_wurcs["source_row_id"].tolist())
    common_idx = idx_iupac & idx_wurcs

    print(f"Nombre de glycannes en communs (mêmes IUPAC): {len(idx_iupac)} glycans")
    print(f"Nombre de glycannes en communs (mêmes WURCS): {len(idx_wurcs)} glycans")
    print(f"Nombre de glycannes en communs (mêmes IUPAC et WURCS): {len(common_idx)} glycans")
    print(f"Indice de Jaccard (mêmes IUPAC et WURCS): {round(len(common_idx) / len(idx_iupac | idx_wurcs), 2)}")
