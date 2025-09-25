import joblib
import csv
import time

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity

from models.classifiers import PureClassifier

# IMPORTANT:
# + this is an example script, paths need to be adapted!

# adjust here depending on anonymisation method!
MODE = "SA-Anon"

# CONFIGURATION
EMBEDDINGS = "/path/embeddings/anon_sa_1069/"

CHECKPOINTS = {
    "gender": "/path/checkpoints_sa/gender/best_checkpoint.pt",
    "age": "/path/checkpoints_sa/age/best_checkpoint.pt",
    "profession": "/path/checkpoints_sa/profession/best_checkpoint.pt",
    "nationality": "/path/checkpoints_sa/nationality/best_checkpoint.pt"
}
TEST_PAIRS_FILE = "/path/set_splits/step_2_vox_2/test_trials.csv"
MODEL_PATH = "/path/models/sa_rf.joblib"

IN_DIM = 192
OUT_DIM = {
    "gender": 2,
    "age": 3,
    "profession": 7,
    "nationality": 10
}
FEATURES = ["gender", "age", "profession", "nationality"]


def load_models():
    models = []
    for f in FEATURES:
        model = PureClassifier(in_dim=IN_DIM, out_dim=OUT_DIM[f])
        state = torch.load(CHECKPOINTS[f])
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        models.append(model)
    return models


def load_pairs(path, emb_path):
    pairs = []
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if len(row) == 3:
                label, path1, path2 = row
                path1 = path1.replace("/ds-slt/audio/VoxCeleb2/data/", emb_path).replace(".wav", ".npy")
                path2 = path2.replace("/ds-slt/audio/VoxCeleb2/data/", emb_path).replace(".wav", ".npy")
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
    for i, (label, f1, f2) in enumerate(pairs):
        sims = []
        for model in models:
            s1 = get_softmax(model, f1)
            s2 = get_softmax(model, f2)
            sim = cosine_similarity([s1], [s2])[0][0]
            sims.append(sim)
        X.append(sims)
        y.append(label)
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(pairs)} pairs", flush=True)
    return np.array(X), np.array(y)


def calculate_eer(y_true, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    return eer, fpr, tpr


def plot_feature_importance(importances, name):
    plt.figure()
    plt.title("Random Forest Feature Importance - " + MODE)
    plt.bar(FEATURES, importances)
    plt.ylabel("Importance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(name)


def plot_roc(fpr, tpr, name):
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (AUC = {:.2f})".format(auc(fpr, tpr)))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - " + MODE)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(name)


def main():
    print("Evaluation started.", flush=True)
    start = time.time()

    # Load trained model
    clf = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")

    # Load classifiers and test pairs
    models = load_models()
    pairs_test = load_pairs(TEST_PAIRS_FILE, EMBEDDINGS)
    print(f"Loaded {len(pairs_test)} test pairs")

    # Prepare embeddings
    X_test, y_test = prepare_embeddings(pairs_test, models)
    print(f"Prepared {len(X_test)} test samples")

    # Predict
    probs_norm = clf.predict_proba(X_test)[:, 1]
    y_pred_norm = clf.predict(X_test)

    # EER
    eer, fpr, tpr = calculate_eer(y_test, probs_norm)
    print(f"EER on test set: {eer:.3f}")

    # AUC
    auc_score = auc(fpr, tpr)
    print(f"AUC on test set: {auc_score:.3f}")

    # Confusion Matrix
    cm_norm = confusion_matrix(y_test, y_pred_norm)
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm)
    disp_norm.plot()
    plt.title("Confusion Matrix - " + MODE)
    plt.savefig(MODE + "confusion_matrix.png")

    # Feature Importance
    importances_norm = clf.feature_importances_
    print("\nFeature Importances:")
    for name, score in zip(FEATURES, importances_norm):
        print(f"{name:12}: {score:.4f}")
    plot_feature_importance(importances_norm, MODE + "feature_importance.png")

    # ROC
    plot_roc(fpr, tpr, MODE + "roc_curve.png")
    print(f"Total evaluation time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
