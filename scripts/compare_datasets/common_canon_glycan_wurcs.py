import argparse
import pandas as pd
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_custom", type=str, default="data/pretraining_atom_full_glyles_1_2_2_parsable.csv")
    parser.add_argument("--data_glycanAA_wurcs", type=str, default="data/glycan_pretrain_wurcs.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/compare_datasets/common_canon_glycan_wurcs")
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    args = get_args()
    data_custom = pd.read_csv(args.data_custom)
    data_glycanAA = pd.read_csv(args.data_glycanAA_wurcs)
    
    # Récupérer les WURCS
    wurcs_total_custom = len(list(data_custom["wurcs"].dropna()))
    wurcs_total_glycanAA = data_glycanAA.shape[0]
    wurcs1 = set(wurcs for wurcs in data_custom["wurcs"].dropna() if "|" not in wurcs)
    wurcs2 = set(wurcs for wurcs in data_glycanAA["WURCS"].dropna() if "|" not in wurcs)
    wurcs1_na = data_custom["wurcs"].isna().sum()
    wurcs2_na = data_glycanAA["WURCS"].isna().sum()
    wurcs1_ambiguous = list(wurcs for wurcs in data_custom["wurcs"].dropna() if "|" in wurcs)
    wurcs2_ambiguous = list(wurcs for wurcs in data_glycanAA["WURCS"].dropna() if "|" in wurcs)

    common_wurcs = wurcs1 & wurcs2
    only_1 = wurcs1 - wurcs2
    only_2 = wurcs2 - wurcs1

    print(f"Glycannes Custom non ambigus : {len(wurcs1)} glycans soit {round(100 * len(wurcs1) / wurcs_total_custom, 2)}%")
    print(f"Glycannes Custom ambigus : {len(wurcs1_ambiguous)} glycans soit {round(100 * len(wurcs1_ambiguous) / wurcs_total_custom, 2)}%")
    print(f"Glycannes GlycanAA non ambigus : {len(wurcs2)} glycans soit {round(100 * len(wurcs2) / wurcs_total_glycanAA, 2)}%")
    print(f"Glycannes GlycanAA ambigus : {len(wurcs2_ambiguous)} glycans soit {round(100 * len(wurcs2_ambiguous) / wurcs_total_glycanAA, 2)}%")
    print(f"Glycannes GlycanAA sans WURCS trouvés : {wurcs2_na} glycans soit {round(100 * wurcs2_na / wurcs_total_glycanAA, 2)}%")
    print(f"Glycannes communs non ambigus : {len(common_wurcs)}")
    print()

    print(f"Glycans du dataset Custom présents dans le dataset GlycanAA: {len(common_wurcs)} glycans soit {round(100 * len(common_wurcs) / len(wurcs1), 2)}%")
    print(f"Glycans du dataset GlycanAA présents dans le dataset Custom: {len(common_wurcs)} glycans soit {round(100 * len(common_wurcs) / len(wurcs2), 2)}%")

    print(f"Glycans du dataset Custom qui sont absents du dataset GlycanAA: {len(only_1)} glycans soit {round(100 * len(only_1) / len(wurcs1), 2)}%")
    print(f"Glycans du dataset GlycanAA qui sont absents du dataset Custom: {len(only_2)} glycans soit {round(100 * len(only_2) / len(wurcs2), 2)}%")

    print(f"Indice de Jaccard: {round(len(common_wurcs) / len(wurcs1 | wurcs2), 2)}")

    common_data_custom = data_custom[data_custom["wurcs"].isin(common_wurcs)]
    common_data_custom.to_csv(f"{args.output_dir}/common_glycans_wurcs.csv", index=False)
