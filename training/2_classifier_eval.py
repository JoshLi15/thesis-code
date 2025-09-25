import argparse
import torch
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from models.classifiers import PureClassifier
from data_io.SpeakerDataset import SpeakerDataset

# IMPORTANT:
# + this is an example script, paths need to be adapted!


def load_model(checkpoint_path, in_dim, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Always have trained model nbr of features as input dimension
    out_dim = checkpoint['model_state_dict']['classifier.8.weight'].shape[0]

    model = PureClassifier(in_dim, out_dim)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def evaluate(model, dataset, classifier_head, device):
    test_feats, label_dict, all_utts = dataset.get_test_items_new()
    valid_feats = []
    valid_labels = []

    c = 0
    for i, feat in enumerate(test_feats):
        if isinstance(feat, torch.Tensor) and feat.shape == (192,):
            valid_feats.append(feat)
            valid_labels.append(label_dict[classifier_head][i])
        else:
            c += 1
    print(f"Skipped: {c}", flush=True)
    
    inputs = torch.stack(valid_feats).to(device)
    targets = torch.tensor(valid_labels).to(device)

    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

    # Accuracy
    correct = (preds == targets).sum().item()
    accuracy = 100 * correct / len(targets)
    print(f"Accuracy: {accuracy:.2f}%", flush=True)

    # Confusion Matrix
    cm = confusion_matrix(targets.cpu(), preds.cpu())
    print("\nConfusion Matrix:", flush=True)
    print(cm)

    # Macro-F1 Score
    macro_f1 = f1_score(targets.cpu(), preds.cpu(), average='macro')
    print(f"\nMacro-F1 Score: {macro_f1:.4f}", flush=True)

    # Top-2 Accuracy (only if enough classes!)
    topk = 2
    if outputs.size(1) >= topk:
        _, topk_preds = outputs.topk(topk, dim=1)
        correct_topk = (topk_preds == targets.unsqueeze(1)).any(dim=1).sum().item()
        topk_accuracy = 100 * correct_topk / len(targets)
        print(f"Top-{topk} Accuracy: {topk_accuracy:.2f}%", flush=True)
    else:
        print(f"Top-{topk}: Not enough classes!", flush=True)

    # Top-5 Accuracy (only if enough classes!)
    topk = 5
    if outputs.size(1) >= topk:
        _, topk_preds = outputs.topk(topk, dim=1)
        correct_topk = (topk_preds == targets.unsqueeze(1)).any(dim=1).sum().item()
        topk_accuracy = 100 * correct_topk / len(targets)
        print(f"Top-{topk} Accuracy: {topk_accuracy:.2f}%", flush=True)
    else:
        print(f"Top-{topk}: Not enough classes!", flush=True)

    # Precision / Recall / F1 per Class
    enc_dict = dataset.get_class_encs()[classifier_head]
    inv_enc_dict = {v: k for k, v in enc_dict.items()}
    class_names = [inv_enc_dict[i] for i in range(len(inv_enc_dict))]
    
    print("\nClassification Report:", flush=True)
    print(classification_report(targets.cpu(), preds.cpu(), target_names=class_names), flush=True)

    return accuracy

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    classifier_heads = args.classifier_heads.split(",")
    test_paths = args.test_paths.split(",")
    assert len(classifier_heads) == len(test_paths), "ERROR: classifier_heads != test_sets!"

    class_enc_dict = None

    for head, path in zip(classifier_heads, test_paths):
        print(f"\nEvaluating classifier '{head}' on: {path}", flush=True)
        dataset = SpeakerDataset(path, test_mode=True, class_enc_dict=class_enc_dict)
        print(f"Classifier head: {dataset.get_class_encs()}", flush=True)
        
        if class_enc_dict is None:
            class_enc_dict = dataset.get_class_encs()
        
        model = load_model(args.checkpoint, args.in_dim,  device)
        acc = evaluate(model, dataset, head, device)
        print(f"Test Accuracy for '{head}': {acc:.2f}%", flush=True)

if __name__ == "__main__":
    # Example usage:
    #     python ./classifier_eval.py \
    #   --checkpoint /path/checkpoints_sa/nationality/best_checkpoint.pt \
    #   --test_paths /path/embeddings/anon_sa_1069_split_pkls/nationality/test.pkl \
    #   --classifier_heads nationality \
    #   --in_dim 192 \
    #   --gpu 0
    parser = argparse.ArgumentParser(description="Evaluate trained classifier(s) on test set(s).")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to best_checkpoint.pt")
    parser.add_argument('--test_paths', type=str, required=True, help="Comma-separated list of test .pkl paths")
    parser.add_argument('--classifier_heads', type=str, required=True, help="Comma-separated list of classifier heads")
    parser.add_argument('--in_dim', type=int, default=192, help="Input dimension to the classifier (e.g. 192 for ECAPA)")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to use (default: 0)")
    args = parser.parse_args()
    main(args)
