import argparse

import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn import metrics
from tqdm import tqdm


class Classifier(nn.Module):
    def __init__(self, positive_rate=None, pretrained=True):
        super(Classifier, self).__init__()
        if pretrained:
            self.backbone = torchvision.models.efficientnet_b1(
                weights="IMAGENET1K_V2",
            )
        else:
            self.backbone = torchvision.models.efficientnet_b1()
        self.output = nn.Linear(1000, 1)

        if positive_rate is not None:
            self.output.bias.data = torch.zeros_like(self.output.bias.data) + torch.log(
                torch.tensor(positive_rate) / (1 - positive_rate)
            )

    def forward(self, x):
        x = self.backbone(x)
        x = self.output(x)
        return torch.sigmoid(x)


def evaluate_model(model, test_loader):
    with torch.inference_mode():
        y_true = []
        y_pred = []
        for batch_idx, data in tqdm(
            enumerate(test_loader),
            desc="Evaluating model",
            leave=False,
            total=len(test_loader),
        ):
            inputs, labels = data
            labels = (labels == 9).long()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().detach().numpy())

        roc_auc = metrics.roc_auc_score(y_true, y_pred)
        xentropy = metrics.log_loss(y_true, y_pred)
    return roc_auc, xentropy


def train(
    n_epochs: int, model: nn.Module, train_loader, test_loader, device, type
) -> pd.DataFrame:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    criterion = nn.BCELoss()
    model.train()
    model.to(device)
    test_loss_history = []
    roc_auc_history = []
    bias_history = []

    roc_auc, xentropy = evaluate_model(model, test_loader)
    roc_auc_history.append(roc_auc)
    test_loss_history.append(xentropy)
    bias_history.append(model.output.bias.data.item())

    for _ in tqdm(range(n_epochs), total=n_epochs, leave=False, desc="Epochs"):
        for batch_idx, data in tqdm(
            enumerate(train_loader),
            leave=False,
            total=len(train_loader),
            desc="Batches",
        ):
            inputs, labels = data
            labels = (labels == 9).float()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels.view(-1))
            loss.backward()
            optimizer.step()

        roc_auc, xentropy = evaluate_model(model, test_loader)
        roc_auc_history.append(roc_auc)
        test_loss_history.append(xentropy)
        bias_history.append(model.output.bias.data.item())

    results_df = pd.DataFrame(
        {
            "loss": test_loss_history,
            "roc_auc": roc_auc_history,
            "type": type,
            "n_epochs": [epoch_idx for epoch_idx in range(len(test_loss_history))],
            "output_bias_value": bias_history,
        }
    )

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V2.transforms(),
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V2.transforms(),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True
    )

    positive_rate = 0.1
    criterion = nn.BCELoss()

    n_tries = 10
    n_epochs = 5

    naive_results_dfs = []
    biased_results_dfs = []

    for trial_idx in tqdm(range(n_tries), desc="Trial"):
        naive_model = Classifier(pretrained=args.pretrained)
        biased_model = Classifier(positive_rate, args.pretrained)

        naive_results_df = train(
            n_epochs, naive_model, train_loader, test_loader, device, "naive"
        )
        biased_results_df = train(
            n_epochs, biased_model, train_loader, test_loader, device, "biased"
        )
        naive_results_df["trial_idx"] = trial_idx
        biased_results_df["trial_idx"] = trial_idx
        naive_results_dfs.append(naive_results_df)
        biased_results_dfs.append(biased_results_df)

        # Save the models
        torch.save(naive_model.state_dict(), f"naive_model_{trial_idx}.pth")
        torch.save(biased_model.state_dict(), f"biased_model_{trial_idx}.pth")

    naive_results_df = pd.concat(naive_results_dfs)
    biased_results_df = pd.concat(biased_results_dfs)

    plotting_df = pd.concat([naive_results_df, biased_results_df])

    if args.pretrained:
        plotting_df.to_csv("results_pretrained.csv", index=False)
    else:
        plotting_df.to_csv("results.csv", index=False)
