import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from pathlib import Path
import pandas as pd
import json

from morph_base import MorphBase


class ResNetBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class CharResNetModel(nn.Module):
    def __init__(self, vocab_size, num_labels, hidden_dim=256, num_layers=10, kernel_size=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.dropout_emb = nn.Dropout(dropout)
        self.layers = nn.ModuleList([ResNetBlock1D(hidden_dim, kernel_size, dropout) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        x = self.dropout_emb(x)
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 1)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return logits, loss


class CharTokenizer:
    def __init__(self):
        self.char2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2char = {0: "<PAD>", 1: "<UNK>"}

    def build_vocab(self, texts):
        chars = set()
        for text in texts:
            chars.update(list(text))
        for char in sorted(list(chars)):
            if char not in self.char2id:
                idx = len(self.char2id)
                self.char2id[char] = idx
                self.id2char[idx] = char

    def __len__(self):
        return len(self.char2id)

    def encode(self, text):
        return [self.char2id.get(c, self.char2id["<UNK>"]) for c in text]

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f: json.dump(self.char2id, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f: self.char2id = json.load(f)
        self.id2char = {v: k for k, v in self.char2id.items()}


class MorphDataset(Dataset):
    def __init__(self, token_ids, label_ids):
        self.token_ids = token_ids
        self.label_ids = label_ids

    def __len__(self): return len(self.token_ids)

    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.token_ids[idx], dtype=torch.long),
                "labels": torch.tensor(self.label_ids[idx], dtype=torch.long)}


def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)
    return input_ids, labels


class MorphResNet(MorphBase):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.tokenizer = CharTokenizer()

    def _load_pretrained_training_state(self, pretrained_dir: Path):
        required_files = ["model.pt", "vocab.json", "labels.json", "config.json"]
        missing_files = [name for name in required_files if not (pretrained_dir / name).exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Cannot initialize ResNet from {pretrained_dir}: missing {', '.join(missing_files)}."
            )

        print(f"Initializing ResNet weights from: {pretrained_dir}")
        self.tokenizer.load(pretrained_dir / "vocab.json")
        with open(pretrained_dir / "labels.json", "r") as f:
            data = json.load(f)
            self.label_list, self.label_to_id = data["label_list"], data["label_to_id"]
        with open(pretrained_dir / "config.json", "r") as f:
            model_params = json.load(f)

        model = CharResNetModel(**model_params).to(self.device)
        model.load_state_dict(torch.load(pretrained_dir / "model.pt", map_location=self.device))
        return model, model_params

    def _validate_training_labels(self, file_path: Path):
        if self.label_to_id is None:
            return

        unknown_labels = set()
        for _, parsing in self._read_pairs(file_path):
            for label in self._convert2bmes(parsing):
                if label not in self.label_to_id:
                    unknown_labels.add(label)

        if unknown_labels:
            formatted_labels = ", ".join(sorted(unknown_labels))
            raise ValueError(
                f"Training file {file_path} contains labels absent from the pretrained checkpoint: "
                f"{formatted_labels}. Use a checkpoint with the same label set or train with --full_set from scratch."
            )

    def _prepare_dataset(self, file_path: Path):
        token_ids_list, label_ids_list, lemmas_list = [], [], []

        for lemma, parsing in self._read_pairs(file_path):
            tokens = list(lemma)
            bmes = self._convert2bmes(parsing)

            if not bmes: continue
            if len(tokens) != len(bmes): continue

            input_ids = self.tokenizer.encode(lemma)
            label_ids = [self.label_to_id.get(l, -100) for l in bmes]

            token_ids_list.append(input_ids)
            label_ids_list.append(label_ids)
            lemmas_list.append(lemma)

        return MorphDataset(token_ids_list, label_ids_list), lemmas_list

    def _make_unlabeled_dataset(self, lemmas: list[str]):
        token_ids_list = [self.tokenizer.encode(lemma) for lemma in lemmas]
        label_ids_list = [[0] * len(token_ids) for token_ids in token_ids_list]
        return MorphDataset(token_ids_list, label_ids_list)

    def _predict_dataset(self, dataset: MorphDataset, lemmas: list[str]) -> list[str] | list[list[str]]:
        loader = DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=False,
            collate_fn=collate_fn
        )

        all_preds = []
        self.model.eval()
        with torch.no_grad():
            for batch_ids, _ in tqdm(loader, desc="Predicting"):
                batch_ids = batch_ids.to(self.device)
                logits, _ = self.model(batch_ids, None)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for i in range(len(probs)):
                    length = torch.sum(batch_ids[i] != 0).item()
                    pred_seq = self._decode_valid_bmes(probs[i], length)
                    all_preds.append(pred_seq)

        return [self._convert_prediction(lemma, bmes) for lemma, bmes in zip(lemmas, all_preds)]

    def train(self, train_file: Path, output_dir: Path, pretrained_dir: Path | str | None = None):
        train_file = Path(train_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Train file: {train_file}")

        if pretrained_dir is not None:
            pretrained_dir = Path(pretrained_dir)
            model, config_to_save = self._load_pretrained_training_state(pretrained_dir)
            self._validate_training_labels(train_file)
            print(f"Vocab size: {len(self.tokenizer)}")
        else:
            self._determine_labels_from_file(train_file)

            all_lemmas = []
            for lemma, _ in self._read_pairs(train_file):
                all_lemmas.append(lemma)
            self.tokenizer.build_vocab(all_lemmas)
            print(f"Vocab size: {len(self.tokenizer)}")

            config_to_save = {
                "vocab_size": len(self.tokenizer), "num_labels": len(self.label_list),
                "hidden_dim": self.config.get("hidden_dim", 256), "num_layers": self.config.get("num_layers", 10),
                "dropout": self.config.get("dropout", 0.1)
            }

            model = CharResNetModel(**config_to_save).to(self.device)

        train_dataset, _ = self._prepare_dataset(train_file)
        train_loader = DataLoader(train_dataset, batch_size=self.config.get("batch_size", 32), shuffle=True,
                                  collate_fn=collate_fn)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.get("learning_rate", 1e-3))

        model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_ids, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                batch_ids, batch_labels = batch_ids.to(self.device), batch_labels.to(self.device)
                optimizer.zero_grad()
                _, loss = model(batch_ids, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f}")

        torch.save(model.state_dict(), output_dir / "model.pt")
        self.tokenizer.save(output_dir / "vocab.json")
        with open(output_dir / "labels.json", "w") as f:
            json.dump({"label_list": self.label_list, "label_to_id": self.label_to_id}, f)

        with open(output_dir / "config.json", "w") as f:
            json.dump(config_to_save, f)
        print("Training complete.")

    def predict(self, model_dir: Path, test_file: Path, predicted_file: Path):
        model_dir = Path(model_dir)
        test_file = Path(test_file)
        predicted_file = Path(predicted_file)
        predicted_file.parent.mkdir(parents=True, exist_ok=True)

        self.load_for_inference(model_dir)

        test_dataset, test_lemmas = self._prepare_dataset(test_file)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        target_text = []
        for lemma, parsing in self._read_pairs(test_file):
            if not self._convert2bmes(parsing): continue
            if len(list(lemma)) != len(self._convert2bmes(parsing)): continue
            target_text.append(parsing)

        all_preds = []
        with torch.no_grad():
            for batch_ids, _ in tqdm(test_loader):
                batch_ids = batch_ids.to(self.device)
                logits, _ = self.model(batch_ids, None)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for i in range(len(probs)):
                    length = torch.sum(batch_ids[i] != 0).item()
                    pred_seq = self._decode_valid_bmes(probs[i], length)
                    all_preds.append(pred_seq)

        predicted_text = [self._convert_prediction(lemma, bmes) for lemma, bmes in zip(test_lemmas, all_preds)]

        test_df = pd.DataFrame({
            "input_text": test_lemmas,
            "target_text": target_text,
            **self._prediction_columns(predicted_text)
        })
        test_df.to_csv(predicted_file, index=False)

        metrics = self._measure_quality(target_text, predicted_text)
        print("Metrics:", metrics)
        self._save_metrics(metrics, predicted_file.parent)

    def load_for_inference(self, model_dir: Path):
        model_dir = Path(model_dir)
        print(f"Loading model for inference from: {model_dir}")
        self.tokenizer.load(model_dir / "vocab.json")
        with open(model_dir / "labels.json", "r") as f:
            data = json.load(f)
            self.label_list, self.label_to_id = data["label_list"], data["label_to_id"]
        with open(model_dir / "config.json", "r") as f:
            model_params = json.load(f)

        self.model = CharResNetModel(**model_params).to(self.device)
        self.model.load_state_dict(torch.load(model_dir / "model.pt", map_location=self.device))
        self.model.eval()

    def predict_batch(self, lemmas: list[str]) -> list[str] | list[list[str]]:
        if not hasattr(self, "model"):
            raise RuntimeError("Model not loaded. Call load_for_inference(model_dir) first.")

        dataset = self._make_unlabeled_dataset(lemmas)
        return self._predict_dataset(dataset, lemmas)

    def predict_file(self, input_file: str | Path, output_file: str | Path | None = None,
                     model_dir: Path | None = None):
        if not hasattr(self, "model"):
            if model_dir is None:
                raise RuntimeError("Model not loaded. Call load_for_inference(model_dir) first or pass model_dir.")
            self.load_for_inference(model_dir)

        input_file = Path(input_file)
        lemmas = list(self._read_lemmas(input_file))
        predicted_text = self.predict_batch(lemmas)

        if output_file is None:
            output_file = input_file.with_suffix(".predicted.csv")
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({
            "input_text": lemmas,
            **self._prediction_columns(predicted_text),
        }).to_csv(output_file, index=False)

        return predicted_text
