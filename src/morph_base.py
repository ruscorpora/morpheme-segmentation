import logging
import json
import sys
from pathlib import Path
from enum import Enum
from collections import defaultdict
from math import exp
import numpy as np
import pandas as pd

from decode_bmes import decode_valid_bmes, decode_valid_bmes_topk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SplitMode(Enum):
    SPLIT_BY_FORM = "form"
    SPLIT_BY_LEMMA = "lemma"
    SPLIT_BY_ROOTS = "roots"
    SPLIT_BY_RANDOM = "random"

class MorphBase:
    """
    Base class for Morphological Segmentation models.
    Handles configuration, data utilities, and metrics.
    """
    def __init__(
        self,
        model_config_name: str,
        dataset_config_name: str,
        models_config_path: str = "src/models.json",
        datasets_config_path: str = "src/datasets.json",
        num_epochs: int = 30,
        outputs_dir: str = "models",
        data_dir: str = "data",
        results_dir: str = "results",
        use_lemma: bool = False,
        full_set: bool = False,
        mode: SplitMode = SplitMode.SPLIT_BY_RANDOM,
        use_language: bool = False,
        top_k: int = 1,
    ):
        self.model_config_name = model_config_name
        self.dataset_config_name = dataset_config_name
        self.num_epochs = num_epochs
        self.outputs_dir = Path(outputs_dir)
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.use_lemma = use_lemma
        self.use_language = use_language
        self.full_set = full_set
        self.mode = mode
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        self.top_k = top_k

        try:
            with open(models_config_path, "r") as f:
                model_conf = json.load(f)[model_config_name]
        except KeyError:
            print(f"Error: Model configuration '{model_config_name}' not found in '{models_config_path}'")
            sys.exit(1)

        try:
            with open(datasets_config_path, "r") as f:
                dataset_conf = json.load(f)[dataset_config_name]
        except KeyError:
            print(f"Error: Dataset configuration '{dataset_config_name}' not found in '{datasets_config_path}'")
            sys.exit(1)

        self.config = {**model_conf, **dataset_conf}
        if self.use_language and self.config.get("model_type") not in {"bert", "xlmroberta"}:
            raise ValueError("--use_language is supported only for BERT-like models.")
        
        self.label_list = None
        self.label_to_id = None

    def _lemma_suffix(self):
        return "with_lemma" if self.use_lemma else "no_lemma"

    def _read_rows(self, file_path: Path, labeled: bool = True):
        """Yield (language, lemma, parsing); language is None in the old format."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                parts = line.strip().split("\t")
                offset = int(self.use_language)
                required_columns = offset + (2 if labeled else 1)
                if self.use_language and (
                    len(parts) < required_columns or not all(parts[:required_columns])
                ):
                    expected = "language<TAB>lemma<TAB>parsing" if labeled else "language<TAB>lemma"
                    raise ValueError(f"{file_path}:{line_number}: expected {expected}.")
                if len(parts) < required_columns:
                    continue
                yield (
                    parts[0] if self.use_language else None,
                    parts[offset],
                    parts[offset + 1] if labeled else None,
                )

    def _read_pairs(self, file_path: Path):
        """Generator that yields (lemma, parsing) tuples from a file."""
        for _, lemma, parsing in self._read_rows(file_path):
            yield lemma, parsing

    @staticmethod
    def _read_lemmas(file_path: Path):
        """Generator that yields lemmas from a plain list or the first TSV column."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield line.split("\t", 1)[0]

    @staticmethod
    def _convert2bmes(parsing: str) -> list:
        if parsing in ["FAILED", "WRONG_LEN"]:
            return []
        bmes = []
        for m in parsing.split("/"):
            if ":" not in m:
                continue
            m_str, m_type = m.rsplit(":", 1)
            
            if len(m_str) == 1:
                bmes.append(f"S-{m_type}")
            else:
                bmes.append(f"B-{m_type}")
                bmes.extend([f"M-{m_type}"] * (len(m_str) - 2))
                bmes.append(f"E-{m_type}")
        return bmes

    @staticmethod
    def _convert2parsing(lemma: str, bmes: list) -> str:
        parsing = []
        current_mtext = ""
        current_mtype = ""
        for letter, label in zip(lemma, bmes):
            if not "-" in label:
                parsing.append(f"{letter}:UNK")
                current_mtext, current_mtype = "", ""
                continue
            pos, mtype = label.split("-")
            if pos == "S":
                if current_mtext:
                    parsing.append(f"{current_mtext}:{current_mtype}")
                parsing.append(f"{letter}:{mtype}")
                current_mtext, current_mtype = "", ""
            elif pos == "B":
                if current_mtext:
                    parsing.append(f"{current_mtext}:{current_mtype}")
                current_mtext = letter
                current_mtype = mtype
            else:
                current_mtext += letter
        if current_mtext:
            parsing.append(f"{current_mtext}:{current_mtype}")
        return "/".join(parsing)

    def _determine_labels_from_file(self, file_path: Path):
        """Scans a file to build the label_list based on unique morpheme types."""
        unique_types = set()
        for _, parsing in self._read_pairs(file_path):
            if parsing in ["FAILED", "WRONG_LEN"]:
                continue
            try:
                for m in parsing.split("/"):
                    if ":" in m:
                        _, m_type = m.rsplit(":", 1)
                        unique_types.add(m_type)
            except ValueError:
                continue

        if self.full_set:
            unique_types |= {"PREF", "ROOT", "SUFF", "END", "HYPN", "POST", "LINK"}
        
        sorted_types = sorted(list(unique_types))
        custom_labels = []
        if self.use_lemma or self.use_language:
            custom_labels.append("0") # Label for auxiliary BERT inputs: lemma and language

        
        for mtype in sorted_types:
            custom_labels.extend(
                [f"B-{mtype}", f"M-{mtype}", f"E-{mtype}", f"S-{mtype}"]
            )
        
        self.label_list = custom_labels
        self.label_to_id = {l: i for i, l in enumerate(self.label_list)}
        print(f"Detected {len(sorted_types)} unique morpheme types. Total labels: {len(self.label_list)}")

    def _decode_valid_bmes(
        self,
        probs: np.ndarray,
        length: int,
        *,
        topk_scores: list[list[dict[str, float]]] | None = None,
    ) -> list[str] | list[list[str]]:
        if self.top_k > 1:
            log_scores = [] if topk_scores is not None else None
            paths = decode_valid_bmes_topk(
                probs, self.label_list, self.top_k, length, self.label_to_id,
                log_scores=log_scores,
            )
            if topk_scores is not None:
                topk_scores.append([
                    {"log_score": score, "confidence": exp(score / length)}
                    for score in log_scores
                ])
            return paths
        return decode_valid_bmes(probs, self.label_list, length, self.label_to_id)

    def _convert_prediction(self, lemma: str, bmes: list) -> str | list[str]:
        if self.top_k > 1:
            return [self._convert2parsing(lemma, path) for path in bmes]
        return self._convert2parsing(lemma, bmes)

    def _prediction_columns(
        self, predicted: list, *, topk_scores: list[list[dict[str, float]]] | None = None
    ) -> dict:
        if self.top_k == 1:
            return {"predicted_text": predicted}
        columns = {
            "predicted_text": [paths[0] if paths else "FAILED" for paths in predicted],
            "predicted_text_topk": [json.dumps(paths, ensure_ascii=False) for paths in predicted],
        }
        if topk_scores is not None:
            columns["predicted_text_topk_scores"] = [json.dumps(scores) for scores in topk_scores]
        return columns

    @staticmethod
    def word_accuracy_at_k(targets: list[str], predicted_topk: list[list[str]], k: int) -> float:
        """Fraction of words whose full gold parsing is among the first k candidates."""
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        if len(targets) != len(predicted_topk):
            raise ValueError("targets and predicted_topk must have the same length.")
        correct = sum(
            target in candidates[:k]
            for target, candidates in zip(targets, predicted_topk)
        )
        return correct / len(targets) if targets else 0.0

    @staticmethod
    def _f_measure(targets, predicted, SE):
        TP, FP, FN, equal, total = 0, 0, 0, 0, 0
        corr_words = 0
        for corr, pred in zip(targets, predicted):
            corr_len = len(corr)
            pred_len = len(pred)
            boundaries = [i for i in range(corr_len) if corr[i] in SE]
            pred_boundaries = [i for i in range(pred_len) if pred[i] in SE]
            common = [x for x in boundaries if x in pred_boundaries]
            TP += len(common)
            FN += len(boundaries) - len(common)
            FP += len(pred_boundaries) - len(common)
            equal += sum(int(x == y) for x, y in zip(corr, pred))
            total += len(corr)
            corr_words += corr == pred
        return TP, FP, FN, equal, total, corr_words

    def _measure_quality(self, targets, predicted):
        # Offline CSV evaluators may create a MorphBase without calling __init__.
        top_k = getattr(self, "top_k", 1)
        topk_accuracy = None
        if top_k > 1:
            topk_accuracy = self.word_accuracy_at_k(targets, predicted, top_k)
            predicted = [paths[0] if paths else "FAILED" for paths in predicted]
        targets = [self._convert2bmes(x) for x in targets]
        predicted = [self._convert2bmes(x) for x in predicted]

        SE = [
            "{}-{}".format(x, y)
            for x in "SE"
            for y in ["ROOT", "PREF", "SUFF", "END", "LINK", "STEM", "POST", "X", "INTERFIX", "CONNECTOR"]
        ]
        TP, FP, FN, equal, total, corr_words = self._f_measure(targets, predicted, SE)

        results = {
            "Precision": TP / (TP + FP) if (TP + FP) > 0 else 0,
            "Recall": TP / (TP + FN) if (TP + FN) > 0 else 0,
            "F1": TP / (TP + 0.5 * (FP + FN)) if (TP + 0.5 * (FP + FN)) > 0 else 0,
            "Accuracy": equal / total if total > 0 else 0,
            "WordAccuracy": corr_words / len(targets) if len(targets) > 0 else 0,
        }
        if topk_accuracy is not None:
            results[f"WordAccuracy@{top_k}"] = topk_accuracy
        return results

    @staticmethod
    def _save_metrics(metrics, results_dir: Path):
        results_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = results_dir / "metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
        else:
            metrics_data = defaultdict(list)
            
        for k, v in metrics.items():
            metrics_data.setdefault(k, []).append(v)

        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=1)

    def train(self, train_file: Path, output_dir: Path, pretrained_dir: Path | str | None = None):
        raise NotImplementedError

    def predict(self, model_dir: Path, test_file: Path, predicted_file: Path):
        raise NotImplementedError

    def load_for_inference(self, model_dir: Path):
        raise NotImplementedError

    def predict_file(self, input_file, output_file=None, model_dir: Path | None = None):
        raise NotImplementedError
