import json
import os
from string import Template

import torch
import torchmetrics
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    model = "google/gemma-2b-it"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_template = Template(
        "<bos><start_of_turn>user\n"
        "Classify the following text into one of the following 4 classes:[World,Sports,Business,Technology]"
        "<separator>Text: $text<separator>Classification:"
        "<end_of_turn>\n"
        "<start_of_turn>model"
    )
    dataset = load_dataset("sh0416/ag_news")
    dataset = dataset.map(
        lambda example: {
            "text": prompt_template.substitute(text=example["title"]),
            "label": example["label"] - 1,
        },
        num_proc=os.cpu_count(),
    )

    # Create train and validation sets from training data
    train_val_split = dataset["train"].train_test_split(test_size=0.1)
    dataset["train"] = train_val_split["train"]
    dataset["validation"] = train_val_split["test"]

    train_loader = DataLoader(dataset["train"], batch_size=8, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(dataset["validation"], batch_size=8, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(dataset["test"], batch_size=8, shuffle=False, pin_memory=True, num_workers=4)

    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
    tokenizer.pad_token = tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        model, device_map="auto", torch_dtype=torch.float32
    )

    target_tokens = tokenizer(
        ["World", "Sports", "Business", "Technology"],
        return_tensors="pt",
        padding=False,
        add_special_tokens=False,
    )["input_ids"].squeeze()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    train_acc_history = []
    val_acc_history = []

    train_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=4)
    val_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=4)

    for epoch in tqdm(range(3)):
        train_loss = 0
        for batch in tqdm(train_loader, desc="Training", leave=False):
            input_ids = tokenizer(batch["text"], return_tensors="pt", padding=True)
            input_ids = {k: v.to(device) for k, v in input_ids.items()}
            outputs = model(**input_ids)
            logits = outputs.logits[:, -1, :]
            filtered_logits = logits[:, target_tokens]
            preds = torch.softmax(filtered_logits, dim=-1)

            loss = criterion(preds, batch["label"].to(device))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()
            train_acc(preds, batch["label"])
        train_acc_history.append(train_acc.compute().item())

        with torch.no_grad():
            val_loss = 0
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                input_ids = tokenizer(batch["text"], return_tensors="pt", padding=True)
                input_ids = {k: v.to(device) for k, v in input_ids.items()}
                outputs = model(**input_ids)
                logits = outputs.logits[:, -1, :]
                filtered_logits = logits[:, target_tokens].squeeze()
                preds = torch.softmax(filtered_logits, dim=-1)

                val_loss += criterion(preds, batch["label"]).item()
                val_acc(preds, batch["label"])
            val_acc_history.append(val_acc.compute().item())

            print(f"Epoch {epoch}: Validation Loss: {val_loss/len(val_loader)}")

        train_loss_hist.append(train_loss / len(train_loader))
        val_loss_hist.append(val_loss / len(val_loader))
        torch.save(model.state_dict(), f"model_{epoch}.pt")
        print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader)}")

    test_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=4)

    with torch.no_grad():
        test_loss = 0
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            input_ids = tokenizer(batch["text"], return_tensors="pt", padding=True)
            input_ids = {k: v.to(device) for k, v in input_ids.items()}
            outputs = model(**input_ids)
            logits = outputs.logits[:, -1, :]
            filtered_logits = logits[:, target_tokens].squeeze()
            preds = torch.softmax(filtered_logits, dim=-1)

            test_loss += criterion(preds, batch["label"]).item()
            test_acc(torch.softmax(filtered_logits, dim=-1), batch["label"])

        print(f"Test AUC: {test_acc.compute().item()}")
        print(f"Test Loss: {test_loss/len(test_loader)}")

    with open("metrics.json", "w") as f:
        json.dump(
            {
                "train_loss": train_loss_hist,
                "val_loss": val_loss_hist,
                "test_loss": test_loss / len(test_loader),
                "train_auc": train_acc_history,
                "val_auc": val_acc_history,
                "test_auc": test_acc.compute(),
            },
            f,
        )
