import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim=6, dropout=0.6):
        unit_1 = 72
        unit_2 = 128
        unit_3 = 256

        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, unit_1),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),

            nn.Linear(unit_1, unit_2),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),

            nn.Linear(unit_2, unit_3),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),

            nn.Linear(unit_3, unit_2),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),

            nn.Linear(unit_2, unit_1),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(unit_1, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, features):
        x = self.layers(features)
        x = self.output_layer(x)

        return x