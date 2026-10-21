import torch

if __name__ == "__main__":
    scores = torch.tensor([0.2, 0.02, 0.15, 0.15, 1.3, 0.5, 0.06, 1.1, 0.05, 3.75])
    probabilities = torch.softmax(scores, dim=0)
    print(probabilities)
    print(probabilities.sum())
