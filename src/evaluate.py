import torch
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipe = pipeline("ner", model="deformer")


def print_pred(text):
    pred = pipe(text)

    print(" ".join([d["word"] for d in pred]))
    print(" ".join([d["entity"] for d in pred]))
    print([d["score"] for d in pred if d["entity"] in ["DE", "DEM"]])


def predict(model, valid_loader):
    model.eval()

    de_labels_list = torch.empty(0).to(device)
    dem_labels_list = torch.empty(0).to(device)
    preds_de_list = torch.empty(0).to(device)
    preds_dem_list = torch.empty(0).to(device)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(valid_loader)):
            batch = {key: torch.squeeze(batch[key]) for key in batch.keys()}
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device).view(-1)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs["logits"], dim=2)
            preds = probs.argmax(dim=2).view(-1)

            de_label_index = torch.where(labels == 1)
            dem_label_index = torch.where(labels == 2)

            de_labels_list = torch.cat((de_labels_list, labels[de_label_index]))
            dem_labels_list = torch.cat((dem_labels_list, labels[dem_label_index]))

            preds_de_list = torch.cat((preds_de_list, preds[de_label_index]))
            preds_dem_list = torch.cat((preds_dem_list, preds[dem_label_index]))

    return preds_de_list == de_labels_list, preds_dem_list == dem_labels_list
