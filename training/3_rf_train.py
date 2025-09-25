import joblib
import csv
import time
import os

import numpy as np

import torch
import torch.nn.functional as F

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

from models.classifiers import PureClassifier

# IMPORTANT:
# + this is an example script, paths need to be adapted!

EMBEDDINGS = "/path/embeddings/anon_sa_1069/"
CHECKPOINTS = {
    "gender": "/path/checkpoints_sa/gender/best_checkpoint.pt",
    "age": "/path/checkpoints_sa/age/best_checkpoint.pt",
    "profession": "/path/checkpoints_sa/profession/best_checkpoint.pt",
    "nationality": "/path/checkpoints_sa/nationality/best_checkpoint.pt"
    }
OUTPUT_MODEL = "/path/models/sa_rf.joblib"

# trial,path-file-one,path-file-two
# 1,/ds-slt/audio/VoxCeleb2/data/id03705/apHIwI99QVM/00294.wav,/ds-slt/audio/VoxCeleb2/data/id03705/YsLI7Wh82xE>
# 1,/ds-slt/audio/VoxCeleb2/data/id03705/WrM22180bOQ/00249.wav,/ds-slt/audio/VoxCeleb2/data/id03705/BcrfkTyANWU>
TRAIN_PAIRS_FILE = "/path/set_splits/step_2_vox_2/train_trials.csv"
VAL_PAIRS_FILE = "/path/set_splits/step_2_vox_2/val_trials.csv"

IN_DIM = 192
OUT_DIM = {
    "gender": 2,
    "age": 3,
    "profession": 7,
    "nationality": 10
}
FEATURES = [
    "gender",
    "age",
    "profession",
    "nationality"
]


def load_models():
    models = []
    for f in FEATURES:
        model = PureClassifier(in_dim=IN_DIM, out_dim=OUT_DIM[f])
        state = torch.load(CHECKPOINTS[f])
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        models.append(model)
    return models

def load_pairs(path):
    pairs = []  # (label, path_emb1, path_emb2)
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if len(row) == 3:
                label, path1, path2 = row
                path1 = path1.replace("/ds-slt/audio/VoxCeleb2/data/",
                                      EMBEDDINGS)
                path1 = path1.replace(".wav", ".npy")
                path2 = path2.replace("/ds-slt/audio/VoxCeleb2/data/",
                                      EMBEDDINGS)
                path2 = path2.replace(".wav", ".npy")
                pairs.append((int(label), path1.strip(), path2.strip()))
    return pairs

def get_softmax(model, emb_path):
    emb = np.load(emb_path)
    inp = torch.tensor(emb).unsqueeze(0).float()
    with torch.no_grad():
        out = model(inp)
        return F.softmax(out, dim=1).numpy().squeeze()

def prepare_embeddings(pairs, models):
    X, y = [], []
    i = 0
    for label, f1, f2 in pairs:
        sims = []
        for model in models:
            s1 = get_softmax(model, f1)
            s2 = get_softmax(model, f2)
            sim = cosine_similarity([s1], [s2])[0][0]
            sims.append(sim)
        X.append(sims)
        y.append(label)
        i += 1
        if i % 1000 == 0:
            print(f"Processed {i}/{len(pairs)} pairs", flush=True)
    X = np.array(X)
    y = np.array(y)
    return X, y


def main():
    # load models
    start = time.time()
    models = load_models()
    print(f"Loaded {len(models)} models: {', '.join(FEATURES)}", flush=True)
    print(f"Model loading took {time.time() - start:.2f} seconds", flush=True)

    # load pairs
    pairs = load_pairs(TRAIN_PAIRS_FILE)
    print(f"Loaded {len(pairs)} pairs from {TRAIN_PAIRS_FILE}", flush=True)
    print(f"Loaded pairs in {time.time() - start:.2f} seconds", flush=True)

    # prepare data for training
    X, y = prepare_embeddings(pairs, models)
    print(f"Prepared {len(X)} training samples with {X.shape[1]} features each.", flush=True)
    print(f"Data preparation took {time.time() - start:.2f} seconds", flush=True)

    # train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    print("Random Forest trained.", flush=True)
    print(f"Training took {time.time() - start:.2f} seconds", flush=True)

    # save model
    os.makedirs(OUTPUT_MODEL, exist_ok=True)
    joblib.dump(clf, OUTPUT_MODEL)
    print(f"Saved to {OUTPUT_MODEL}", flush=True)

if __name__ == "__main__":
    print("Start.", flush=True)
    main()
    print("Done.", flush=True)
