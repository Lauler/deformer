import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, AutoTokenizer, AdamW
from tqdm import tqdm
from src.dataset import DedemDataset
from src.evaluate import predict

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


model = BertForTokenClassification.from_pretrained("KB/bert-base-swedish-cased", num_labels=3)
model.to(device)
model.train()

df = pd.read_feather("data/dedem_corpus.feather")
df = df.sample(frac=1, random_state=5).reset_index(drop=True)


train_dataset = DedemDataset(
    df["text"][0:1500000], de_to_dem_prob=0.47, dem_to_de_prob=0.4, cased=False
)
df_valid = df[1500000:1531202].reset_index(drop=True)
df_valid = df_valid[~(df_valid["text"].str.contains(" dem? som"))].reset_index(drop=True)
valid_dataset = DedemDataset(
    df_valid["text"], de_to_dem_prob=0.47, dem_to_de_prob=0.4, cased=False,
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=20, shuffle=False)

optim = AdamW(model.parameters(), lr=1e-6)
loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optim, max_lr=1.5e-6, epochs=2, steps_per_epoch=len(train_loader), pct_start=0.1
)

log_list = []
for epoch in range(2):
    print(f"epoch: {epoch + 1} started")
    running_loss = 0
    for i, batch in enumerate(tqdm(train_loader)):
        optim.zero_grad()
        # [batch_size, 1, seq_len] -> [batch_size, seq_len]
        batch = {key: torch.squeeze(batch[key]) for key in batch.keys()}
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)

        # active_loss = labels.view(-1) != -100
        active_loss = attention_mask.view(-1) == 1
        active_logits = outputs["logits"].view(-1, 3)  # num_labels=3
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(loss_fn.ignore_index).type_as(labels)
        )
        loss = loss_fn(active_logits, active_labels)
        dedem_positions = labels.view(-1) != 0
        loss[dedem_positions] = loss[dedem_positions] * 10
        loss = torch.mean(loss)
        # loss = outputs["loss"]

        running_loss += loss.item()

        if i % 50 == 49:
            print(f"iter: {i+1}, loss: {running_loss/50:.8f}, lr: {scheduler.get_last_lr()}")
            log_list.append({"iter": i + 1, "loss": running_loss / 50})
            running_loss = 0

        loss.backward()
        optim.step()
        scheduler.step()


# model.load_state_dict(torch.load("checkpoints/demformer.pt"))
model.eval()

res = predict(model, valid_loader)

sum(res[0]) / len(res[0])  # de accuracy
sum(res[1]) / len(res[1])  # dem accuracy

tokenizer = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
model.config.id2label = {"0": "ord", "1": "DE", "2": "DEM"}  # uncased version
model.config.label2id = {"ord": "0", "DE": "1", "DEM": "2"}  # uncased version
model.save_pretrained("deformer")
tokenizer.save_pretrained("deformer")
torch.save(model.state_dict(), "checkpoints/deformer.pt")


# tokenizer = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
# tokenized_text = tokenizer(
#     "Dem trettio männen tittade på dem åtta åsnorna.",
#     return_tensors="pt",
# ).to(device)

# with torch.no_grad():
#     outputs = model(**tokenized_text)

# predictions = outputs["logits"].argmax(dim=2)
