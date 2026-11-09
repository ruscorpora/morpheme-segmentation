from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from pathlib import Path
import pandas as pd
import numpy as np
import sys

from morph_base import MorphBase


class MorphBERT(MorphBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = None
        self.data_collator = None

    def _load_tokenizer(self, tokenizer_path: str | Path):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        except OSError as exc:
            raise OSError(
                f"Could not load tokenizer from {tokenizer_path}. "
                "For inference, the fine-tuned model directory must contain tokenizer files. "
                "Retrain the model or save the tokenizer into that directory."
            ) from exc
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

    def _resolve_initial_model_name(self, pretrained_dir: Path | str | None = None):
        if pretrained_dir is not None:
            return str(pretrained_dir)
        if "{fold}" in self.config["model_name"]:
            raise ValueError(
                "This BERT config has a fold-dependent model_name. "
                "Pass the concrete pretrained_dir to train()."
            )
        return self.config["model_name"]

    def _load_data_for_hf(self, file_path: Path) -> Dataset:
        data = {"tokens": [], "labels": [], "lemmas": [], "languages": []}

        for language, lemma, parsing in self._read_rows(file_path):
            tokens = list(lemma)
            bmes = self._convert2bmes(parsing)

            if not bmes: continue
            if len(tokens) != len(bmes):
                continue

            if self.use_lemma:
                tokens = [lemma] + tokens
                bmes = ["0"] + bmes

            if self.use_language:
                tokens = [language] + tokens
                bmes = ["0"] + bmes

            data["tokens"].append(tokens)
            data["labels"].append(bmes)
            data["lemmas"].append(lemma)
            data["languages"].append(language)

        return Dataset.from_dict(data)

    def _tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        labels = []
        word_ids_list = []

        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)

            clean_word_ids = [-1 if w is None else w for w in word_ids]
            word_ids_list.append(clean_word_ids)

            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_to_id.get(label[word_idx], -100))
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["word_ids"] = word_ids_list
        return tokenized_inputs

    @staticmethod
    def _softmax(logits):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def _reconstruct_tags(self, predictions, dataset_with_word_ids, original_dataset, *, topk_scores=None):
        final_text_outputs = []

        for i, token_logits in enumerate(predictions):
            word_ids = dataset_with_word_ids[i]["word_ids"]
            original_tokens = original_dataset[i]["tokens"]
            lemma_str = original_dataset[i]["lemmas"]

            num_source_tokens = len(original_tokens)
            token_probs = self._softmax(token_logits)
            reconstructed_probs = [None] * num_source_tokens

            for probs, word_id in zip(token_probs, word_ids):
                if word_id != -1 and word_id < num_source_tokens:
                    if reconstructed_probs[word_id] is None:
                        reconstructed_probs[word_id] = probs

            fallback_probs = np.full(len(self.label_list), 1.0 / len(self.label_list), dtype=np.float64)
            for j, probs in enumerate(reconstructed_probs):
                if probs is None:
                    if j > 0 and reconstructed_probs[j - 1] is not None:
                        reconstructed_probs[j] = reconstructed_probs[j - 1]
                    else:
                        reconstructed_probs[j] = fallback_probs

            prefix_length = int(self.use_lemma) + int(self.use_language)
            probs_to_decode = np.asarray(reconstructed_probs[prefix_length:])
            tags_to_parse = self._decode_valid_bmes(
                probs_to_decode, len(lemma_str), topk_scores=topk_scores
            )

            final_text_outputs.append(
                self._convert_prediction(lemma_str, tags_to_parse)
            )

        return final_text_outputs

    def _make_unlabeled_dataset(self, lemmas: list[str], languages: list[str] | None = None) -> Dataset:
        if self.use_language and (
            languages is None or len(languages) != len(lemmas) or not all(languages)
        ):
            raise ValueError("With use_language=True, provide one non-empty language code per lemma.")
        data_dict = {"tokens": [], "labels": [], "lemmas": []}
        dummy_tag = self.label_list[0]

        for i, lemma in enumerate(lemmas):
            tokens = list(lemma)
            dummy_labels = [dummy_tag] * len(tokens)

            if self.use_lemma:
                tokens = [lemma] + tokens
                dummy_labels = ["0"] + dummy_labels

            if self.use_language:
                tokens = [languages[i]] + tokens
                dummy_labels = ["0"] + dummy_labels

            data_dict["tokens"].append(tokens)
            data_dict["labels"].append(dummy_labels)
            data_dict["lemmas"].append(lemma)

        return Dataset.from_dict(data_dict)

    def _predict_dataset(self, dataset: Dataset, *, topk_scores=None) -> list[str] | list[list[str]]:
        tokenized_dataset = dataset.map(
            self._tokenize_and_align_labels,
            batched=True,
            desc=None
        )

        trainer = Trainer(
            model=self.model,
            args=self.inference_args,
            data_collator=self.data_collator,
        )

        predictions_output = trainer.predict(tokenized_dataset)

        return self._reconstruct_tags(
            predictions_output.predictions,
            tokenized_dataset,
            dataset,
            topk_scores=topk_scores,
        )

    def train(self, train_file: Path, output_dir: Path, pretrained_dir: Path | str | None = None):
        train_file = Path(train_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Train file: {train_file}")

        self._determine_labels_from_file(train_file)
        model_name = self._resolve_initial_model_name(pretrained_dir)
        self._load_tokenizer(model_name)

        train_dataset = self._load_data_for_hf(train_file)
        tokenized_train_dataset = train_dataset.map(self._tokenize_and_align_labels, batched=True)

        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_list),
            id2label={i: l for i, l in enumerate(self.label_list)},
            label2id=self.label_to_id,
            trust_remote_code=(self.model_config_name == "hplt"),
            use_safetensors=(self.model_config_name not in ["hplt", "bel_roberta"]),
        )

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.num_epochs,
            learning_rate=self.config["learning_rate"],
            per_device_train_batch_size=16,
            per_device_eval_batch_size=128,
            overwrite_output_dir=True,
            save_steps=-1,
            save_total_limit=1,
            logging_steps=100,
            bf16=True,
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
            report_to=None,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        print("Training complete.")

    def predict(self, model_dir: Path, test_file: Path, predicted_file: Path):
        model_dir = Path(model_dir)
        test_file = Path(test_file)
        predicted_file = Path(predicted_file)
        predicted_file.parent.mkdir(parents=True, exist_ok=True)

        self.load_for_inference(model_dir)

        test_dataset = self._load_data_for_hf(test_file)
        topk_scores = [] if self.top_k > 1 else None
        predicted_text = self._predict_dataset(test_dataset, topk_scores=topk_scores)

        prefix_length = int(self.use_lemma) + int(self.use_language)
        test_df = pd.DataFrame({
            "input_text": test_dataset["lemmas"],
            "target_text": [
                self._convert2parsing(p['lemmas'], p['labels'][prefix_length:])
                for p in test_dataset
            ],
            **self._prediction_columns(predicted_text, topk_scores=topk_scores)
        })
        if self.use_language:
            test_df.insert(0, "language", test_dataset["languages"])
        test_df.to_csv(predicted_file, index=False)

        metrics = self._measure_quality(test_df.target_text.to_list(), predicted_text)
        print("Metrics:", metrics)
        self._save_metrics(metrics, predicted_file.parent)

    def load_for_inference(self, model_dir: Path):
        model_dir = Path(model_dir)
        print(f"Loading model for inference from: {model_dir}")
        try:
            self._load_tokenizer(model_dir)
            self.model = AutoModelForTokenClassification.from_pretrained(
                str(model_dir),
                trust_remote_code=(self.model_config_name == "hplt"),
            )
        except OSError as exc:
            print(f"Error: Could not load model from {model_dir}: {exc}")
            sys.exit(1)

        self.label_to_id = self.model.config.label2id
        self.label_list = [self.model.config.id2label[i] for i in sorted(self.model.config.id2label.keys())]

        self.inference_args = TrainingArguments(
            output_dir=str(model_dir),
            report_to="none",
            per_device_eval_batch_size=256
        )

    def predict_batch(self, lemmas: list[str], languages: list[str] | None = None) -> list[str] | list[list[str]]:
        if not hasattr(self, 'model'):
            raise RuntimeError("Model not loaded. Call load_for_inference(model_dir) first.")

        dataset = self._make_unlabeled_dataset(lemmas, languages=languages)
        return self._predict_dataset(dataset)

    def predict_file(self, input_file: str | Path, output_file: str | Path | None = None,
                     model_dir: Path | None = None):
        if not hasattr(self, 'model'):
            if model_dir is None:
                raise RuntimeError("Model not loaded. Call load_for_inference(model_dir) first or pass model_dir.")
            self.load_for_inference(model_dir)

        input_file = Path(input_file)
        rows = list(self._read_rows(input_file, labeled=False))
        lemmas = [row[1] for row in rows]
        languages = [row[0] for row in rows] if self.use_language else None
        dataset = self._make_unlabeled_dataset(lemmas, languages=languages)
        topk_scores = [] if self.top_k > 1 else None
        predicted_text = self._predict_dataset(dataset, topk_scores=topk_scores)

        if output_file is None:
            output_file = input_file.with_suffix(".predicted.csv")
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        result_df = pd.DataFrame({
            "input_text": lemmas,
            **self._prediction_columns(predicted_text, topk_scores=topk_scores),
        })
        if self.use_language:
            result_df.insert(0, "language", languages)
        result_df.to_csv(output_file, index=False)

        return predicted_text
