import torch
from torch.utils.data import Dataset
import pickle
from collections import defaultdict, Counter


class SpeakerDataset(Dataset):
    def __init__(self, path_to_data, test_mode=False, class_enc_dict=None):
        super().__init__()
        with open(path_to_data, "rb") as f:
            self.data = pickle.load(f)

        self.test_mode = test_mode
        self.features = self.data["features"]
        self.utt_ids = self.data.get("utt_ids", [str(i) for i in range(len(self.features))])
        self.label_keys = [k for k in self.data.keys() if k not in ["features", "utt_ids"]]

        self.labels = defaultdict(list)
        self.class_enc_dict = class_enc_dict or {}

        for label_key in self.label_keys:
            values = self.data[label_key]

            # map labels to integers if not already done
            if label_key not in self.class_enc_dict:
                classes = sorted(set(values))
                self.class_enc_dict[label_key] = {cls: i for i, cls in enumerate(classes)}

            encoder = self.class_enc_dict[label_key]
            for val in values:
                self.labels[label_key].append(encoder[val])

        # transform to torch.Tensor-compatible lists
        self.labels = {k: torch.tensor(v, dtype=torch.long) for k, v in self.labels.items()}

        # number classes
        self.num_classes = {k: len(enc) for k, enc in self.class_enc_dict.items()}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        if self.test_mode:
            return x
        y = {k: self.labels[k][idx] for k in self.labels}
        return x, y
    
    def get_test_items_new(self):
        test_feats = [torch.tensor(feat, dtype=torch.float32) for feat in self.features]
        label_dict = {k: v for k, v in self.labels.items()}
        all_utts = self.utt_ids
        return test_feats, label_dict, all_utts


    def get_class_encs(self):
        return self.class_enc_dict

    def get_class_counts(self):
        return {k: dict(Counter(v.tolist())) for k, v in self.labels.items()}

    def get_batches(self, batch_size=64, max_seq_len=350):
        n = len(self)
        indices = torch.randperm(n)
        
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i + batch_size]

            valid_inputs = []
            valid_labels_dict = defaultdict(list)

            for idx in batch_idx:
                feat, label_dict = self[idx]

                # skip embeddings with wrong shape
                if not isinstance(feat, torch.Tensor) or feat.ndim != 1 or feat.shape[0] != 192:
                    continue

                valid_inputs.append(feat)
                for k, v in label_dict.items():
                    valid_labels_dict[k].append(v)

            # skip this batch if no valid inputs
            if len(valid_inputs) == 0:
                continue

            inputs = torch.stack(valid_inputs)
            labels_dict = {k: torch.tensor(v, dtype=torch.long) for k, v in valid_labels_dict.items()}
            yield inputs, labels_dict
