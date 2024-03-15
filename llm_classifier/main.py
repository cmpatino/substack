import os
from string import Template

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    prompt_template = Template(
        "Classify the following text into one of the following 4 classes:[World,Sports,Business,Technology]"
        "\n\nText: $text\nClassification:"
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

    train_loader = DataLoader(dataset["train"], batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=8, shuffle=False)
    test_loader = DataLoader(dataset["test"], batch_size=8, shuffle=False)

    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2", padding_side="left"
    )
    tokenizer.pad_token = tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    target_tokens = tokenizer(
        ["World", "Sports", "Business", "Technology"],
        return_tensors="pt",
        padding=False,
        add_special_tokens=False,
    )["input_ids"].squeeze()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    train_loss_hist = []
    val_loss_hist = []
    train_roc_auc_hist = []
    val_roc_auc_hist = []

    for epoch in tqdm(range(3)):
        train_loss = 0
        for batch in tqdm(train_loader, desc="Training", leave=False):
            input_ids = tokenizer(batch["text"], return_tensors="pt", padding=True)
            outputs = model(**input_ids)
            import pdb

            pdb.set_trace()
            logits = outputs.logits[:, -1, :]
            filtered_logits = logits[:, target_tokens]

            loss = criterion(torch.softmax(filtered_logits, dim=-1), batch["label"])
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()

        with torch.no_grad():
            val_loss = 0
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                input_ids = tokenizer(batch["text"], return_tensors="pt", padding=True)
                outputs = model(**input_ids)
                logits = outputs.logits[:, -1, :]
                filtered_logits = logits[:, target_tokens].squeeze()

                val_loss += criterion(
                    torch.softmax(filtered_logits, dim=-1), batch["label"]
                ).item()

            print(f"Epoch {epoch}: Validation Loss: {val_loss/len(val_loader)}")

        train_loss_hist.append(train_loss / len(train_loader))
        val_loss_hist.append(val_loss / len(val_loader))
        torch.save(model.state_dict(), f"model_{epoch}.pt")
        print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader)}")
