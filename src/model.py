import torch, torch.nn as nn
class TwoViewCNN(nn.Module):
    """Astronet-style: Global branch (2001) + Local branch (201) -> concat -> FC"""
    def __init__(self):
        super().__init__()
        def branch(in_len):
            return nn.Sequential(
                nn.Conv1d(1,16,5,padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16,32,5,padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32,64,5,padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(32),
                nn.Flatten(),
            )
        self.global_branch = branch(2001)
        self.local_branch = branch(201)
        # 64*32 = 2048 per branch
        self.head = nn.Sequential(nn.Linear(4096,256), nn.ReLU(), nn.Dropout(0.3),
                                   nn.Linear(256,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, g, l):
        gf = self.global_branch(g.unsqueeze(1))
        lf = self.local_branch(l.unsqueeze(1))
        return self.head(torch.cat([gf,lf], dim=1)).squeeze(1)
