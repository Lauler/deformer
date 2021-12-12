import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class DedemDataset(Dataset):
    def __init__(self, sentences, de_to_dem_prob=0.4, dem_to_de_prob=0.1, cased=False):
        self.sentences = sentences
        self.dem_prob = de_to_dem_prob
        self.de_prob = dem_to_de_prob
        self.tokenizer = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
        self.de_id = self.tokenizer.convert_tokens_to_ids("de")
        self.De_id = self.tokenizer.convert_tokens_to_ids("De")
        self.dem_id = self.tokenizer.convert_tokens_to_ids("dem")
        self.Dem_id = self.tokenizer.convert_tokens_to_ids("Dem")
        self.de_label = 1
        self.dem_label = 2
        self.cased = cased

        if self.cased:
            self.de_label = 1
            self.De_label = 2
            self.dem_label = 3
            self.Dem_label = 4

    def __len__(self):
        return len(self.sentences)

    def substitute_tokens(self, tokens, from_token, to_token, prob):
        """
        Randomly change one token id to another with probability prob 
        """
        from_token_locs = torch.where(tokens == from_token)[0]

        for from_token_loc in from_token_locs:
            # Sample 0 or 1 to decide if we randomly change from_token to to_token.
            change_token = torch.multinomial(torch.tensor([1 - prob, prob]), num_samples=1)

            if change_token:
                tokens[from_token_loc] = to_token

        return tokens

    def get_labels(self, tokens):
        """
        Construct vector of labels.
        """
        labels = torch.zeros(len(tokens), dtype=int)
        labels[torch.where(tokens == self.de_id)] = self.de_label
        labels[torch.where(tokens == self.dem_id)] = self.dem_label
        labels[torch.where(tokens == self.De_id)] = self.de_label
        labels[torch.where(tokens == self.Dem_id)] = self.dem_label

        if self.cased:
            labels[torch.where(tokens == self.De_id)] = self.De_label
            labels[torch.where(tokens == self.Dem_id)] = self.Dem_label

        return labels

    def __getitem__(self, index):

        tokenized_text = self.tokenizer(
            self.sentences[index],
            padding="max_length",
            truncation=True,
            max_length=250,
            return_tensors="pt",
        )

        tokens = tokenized_text["input_ids"][0]
        labels = self.get_labels(tokens)

        if not self.cased:
            De_index = torch.where(tokens == self.De_id)
            Dem_index = torch.where(tokens == self.Dem_id)
            tokens[De_index] = self.de_id
            tokens[Dem_index] = self.dem_id

        tokens = self.substitute_tokens(
            tokens=tokens, from_token=self.de_id, to_token=self.dem_id, prob=self.dem_prob,
        )

        tokens = self.substitute_tokens(
            tokens=tokens, from_token=self.dem_id, to_token=self.de_id, prob=self.de_prob,
        )

        if self.cased:
            tokens = self.substitute_tokens(
                tokens=tokens, from_token=self.De_id, to_token=self.Dem_id, prob=self.dem_prob,
            )

            tokens = self.substitute_tokens(
                tokens=tokens, from_token=self.Dem_id, to_token=self.De_id, prob=self.de_prob,
            )

            # Cased examples are rare, so we oversample them by changing som uncased to cased
            tokens = self.substitute_tokens(
                tokens=tokens, from_token=self.de_id, to_token=self.Dem_id, prob=self.dem_prob,
            )

            tokens = self.substitute_tokens(
                tokens=tokens, from_token=self.dem_id, to_token=self.De_id, prob=self.de_prob,
            )

        tokenized_text["input_ids"][0] = tokens
        tokenized_text["labels"] = labels
        tokenized_text["sentence"] = self.sentences[index]

        return tokenized_text
