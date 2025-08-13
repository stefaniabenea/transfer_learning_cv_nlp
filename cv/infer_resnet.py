import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import os
import argparse


weights = models.ResNet18_Weights.DEFAULT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "resnet18_cifar10_best.pth"
image_path = "D:/learning/complete_cv_nlp_project/test_images_cifar/3.jpg"  


cifar10_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


infer_transform = weights.transforms()


def load_best_model(checkpoint_path, device):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 10)
    state = torch.load(checkpoint_path, map_location=device)
    m.load_state_dict(state)
    m.to(device)
    m.eval()
    return m

@torch.no_grad()
def predict_image(image_path, model, topk=3):
    assert os.path.isfile(image_path), f"The file does not exist: {image_path}"
    img = Image.open(image_path).convert("RGB")
    x = infer_transform(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = logits.softmax(dim=1).squeeze(0)
    topk_probs, topk_idx = probs.topk(topk)

    results = []
    for p, idx in zip(topk_probs.tolist(), topk_idx.tolist()):
        results.append({"class_id": idx, "class_name": cifar10_classes[idx], "prob": float(p)})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with ResNet18 fine-tuned on CIFAR-10")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    
    model = load_best_model(checkpoint_path, device)
    preds = predict_image(args.image, model, topk=3)
    print("\nPredictions:")
    for pred in preds:
        print(f"{pred['class_name']} ({pred['prob']*100:.2f}%)")
        
