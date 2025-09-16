import torch
import torch.nn as nn

# Classifier definition
class ClipClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# 8 subtype classes (must match training dataset order!)
class_names = [
    "ADENOSIS",
    "Fibro adenoma",
    "Phyllodes-tumor",
    "Tubular Adenoma",
    "Ductal_Carcinoma",
    "Lobular_Carcinoma",
    "Mucinous_Carcinoma",
    "Papillary_carcinoma"
]

# Map subtype -> main type
def get_main_type(subtype):
    if subtype in ["ADENOSIS", "Fibro adenoma", "Phyllodes-tumor", "Tubular Adenoma"]:
        return "benign"
    elif subtype in ["Ductal_Carcinoma", "Lobular_Carcinoma", "Mucinous_Carcinoma", "Papillary_carcinoma"]:
        return "malignant"
    else:
        return "unknown"
