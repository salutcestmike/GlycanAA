import argparse
from pathlib import Path

import requests
import pandas as pd
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from tqdm.auto import tqdm

BASE_URL = (
    "https://api.glycosmos.org/"
    "glycanformatconverter/2.10.4/"
    "iupaccondensed2wurcs/"
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/glycan_pretrain.csv")
    parser.add_argument("--output_file", type=str, default="data/glycan_pretrain_wurcs.csv")
    args = parser.parse_args()
    args.output_file = Path(args.output_file)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    return args

def iupac_to_wurcs(iupac, max_retries=3):

    url = BASE_URL + quote(iupac, safe="")

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(2 ** attempt)
                continue

            response.raise_for_status()
            result = response.json()

            return {
                "IUPAC": iupac,
                "WURCS": result.get("wurcs"),
                "GlyTouCan": result.get("id"),
                "Error": None
            }

        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "IUPAC": iupac,
                    "WURCS": None,
                    "GlyTouCan": None,
                    "Error": str(e) if e else "NA"
                }

            time.sleep(2 ** attempt)


def parallel_conversion(glycans, max_workers=5):

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {
            executor.submit(iupac_to_wurcs, glycan): glycan
            for glycan in glycans
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Conversion WURCS",
            unit="glycan"
        ):
            results.append(future.result())

    return pd.DataFrame(results)


if __name__ == "__main__":
    args = get_args()
    glycans = pd.read_csv(args.data)["IUPAC Condensed"].tolist()
    df = parallel_conversion(glycans, max_workers=100)

    df.to_csv(args.output_file, index=False)