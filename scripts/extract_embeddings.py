import os, sys, argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils import data as torch_data
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from torchdrug import data, utils
from module.custom_data import all_atom_glycan
from module.custom_models.glycanaa import GlycanAA



class DS(torch_data.Dataset):
    def __len__(self): return len(graphs)
    def __getitem__(self, i): return {"graph": graphs[i]}



def getArgs():

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required = True)
    ap.add_argument("--col", required = True, help = "colonne IUPAC (ex. 'IUPAC Condensed' ou 'target')")
    ap.add_argument("--ckpt", required = True, help = "model_epoch_N.pth du pré-entraînement")
    ap.add_argument("--out", default = "outputs/embeddings")
    ap.add_argument("--batch_size", type = int, default = 256)
    ap.add_argument("--gpu", type = int, default = None)

    return ap.parse_args()



if __name__ == "__main__":

    args = getArgs()
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cpu")

    # Encodeur, mêmes hyperparamètres que dans configs/pretrain/GlycanAA/base.yaml
    model = GlycanAA(input_dim = 128,
                     hidden_dims = [128, 128, 128], 
                     num_unit = 210, 
                     num_relation = 89,
                     batch_norm = True, 
                     activation = "gelu", 
                     concat_hidden = True,
                     readout = "dual")
    
    state = torch.load(os.path.expanduser(args.ckpt), map_location = "cpu")
    print(model.load_state_dict(state, strict=True))      # doit afficher <All keys matched successfully>
    model.to(device).eval()                         

    # Graphes
    iupacs = pd.read_csv(args.csv)[args.col].tolist()
    kw = dict(unit_feature = "default",
              link_feature = "default",
              atom_feature = "symbol",
              bond_feature = "symbol")
    graphs, kept, skipped = [], [], []
    for i, s in enumerate(tqdm(iupacs, desc = "graphes")):
        try:
            graphs.append(all_atom_glycan.HeterogeneousAllAtomGlycan.from_iupac(s, **kw))
            kept.append(i)
        except Exception as e:                            # unité hors vocabulaire 
            skipped.append((i, s, repr(e)))


    loader = data.DataLoader(DS(), batch_size = args.batch_size, shuffle = False)

    # Passage dans l'encodeur
    feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc = "embeddings"):
            batch = utils.cuda(batch, device = device)
            g = batch["graph"]
            out = model(g, g.node_feature.float())
            feats.append(out["graph_feature"].cpu())
    emb = torch.cat(feats).numpy()

    np.save(args.out + ".npy", emb)
    pd.DataFrame({"row_in_csv": kept, "iupac": [iupacs[i] for i in kept]}).to_csv(args.out + "_index.csv", index=False)
    pd.DataFrame(skipped, columns=["row_in_csv", "iupac", "erreur"]).to_csv(args.out + "_skipped.csv", index=False)
    print("embeddings :", emb.shape, "| ignorés :", len(skipped))