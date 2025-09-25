import torch.nn as nn

class PureClassifier(nn.Module):
    def __init__(self, in_dim=192, out_dim=2, dropout=0.3):  # 192 because of ECAPA
        super().__init__()

        # WITH droput and batch for better generalization
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=dropout),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.classifier(x)
