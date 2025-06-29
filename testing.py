from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import random
import torch
import os

test_data_path='./'


g = torch.Generator()

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, include_distortions=True):
        self.transform = transform
        self.data = {}
        self.people = []
        self.samples = []

        for person_id in sorted(os.listdir(root_dir)):
            person_path = os.path.join(root_dir, person_id)
            if not os.path.isdir(person_path):
                continue

            all_images = []
            for file in os.listdir(person_path):
                if file.endswith(".jpg") and file != "distortion":
                    all_images.append(os.path.join(person_path, file))
            if include_distortions:
                distortion_dir = os.path.join(person_path, "distortion")
                if os.path.exists(distortion_dir):
                    for dfile in os.listdir(distortion_dir):
                        if dfile.endswith(".jpg"):
                            all_images.append(os.path.join(distortion_dir, dfile))

            if len(all_images) >= 2:
                self.data[person_id] = all_images
                self.people.append(person_id)

        for person_id in self.people:
            anchors, positive_ids=[], []
            for img_path in self.data[person_id]:
                if "distortion" in img_path:
                    positive_ids.append(img_path)
                else:
                    anchors.append(img_path)

            for anchor in anchors:
                for positive_id in positive_ids:
                    if anchor.split("/")[-1].strip(".jpg") in positive_id:
                        neg_person = random.choice([p for p in self.people if p != person_id])
                        self.samples.append((anchor, positive_id, random.choice(self.data[neg_person])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        a_path, p_path, n_path = self.samples[idx]

        anchor = Image.open(a_path).convert("RGB")
        positive = Image.open(p_path).convert("RGB")
        negative = Image.open(n_path).convert("RGB")

        anchor_label = os.path.basename(os.path.dirname(os.path.dirname(a_path))) \
        if "distortion" in a_path else os.path.basename(os.path.dirname(a_path))

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative, anchor_label
    
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=128,pretrained=True):
        super().__init__()
        self.backbone = resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_size)

    def forward(self, x):
        return self.backbone(x)
    
def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def evaluate_model(custom_model, loader, device, threshold=0.5):
    y_true = []
    y_pred = []
    count=0
    with torch.no_grad():
        for anchor, positive, negative, _ in loader:
            count+=1
            print(f"\rProcessing Evaluation... {count*100/len(loader):.6f}%", end="")
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_emb = custom_model(anchor)
            positive_emb = custom_model(positive)
            negative_emb = custom_model(negative)

            sim_ap = F.cosine_similarity(anchor_emb, positive_emb, dim=1)
            sim_an = F.cosine_similarity(anchor_emb, negative_emb, dim=1)

            pred_ap = (sim_ap >= threshold).int().tolist()
            pred_an = (sim_an >= threshold).int().tolist()

            y_true += [1] * len(pred_ap) + [0] * len(pred_an)
            y_pred += pred_ap + pred_an

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    print(f"\nEvaluation: Accuracy : {acc:.6f}\tPrecision: {prec:.6f}\tRecall: {rec:.6f}\tF1 Score : {f1:.6f}")

test_dataset = TripletFaceDataset(
    root_dir=os.path.join(test_data_path),
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=2,
    generator=g,
    pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbeddingNet(embedding_size=128, pretrained=False)
checkpoint_path = os.path.join("resnet18-face_recognition-epoch12-v05.pth")
state_dict = torch.load(checkpoint_path, map_location="cuda")
model.load_state_dict(state_dict)
model.to(device)
model.eval()

threshold=0.6
evaluate_model(model, test_loader, device, threshold)
