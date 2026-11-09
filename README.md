# RNC Morpheme Segmentation

This code trains morpheme segmentation models and applies them to words. The main data format is:

```text
word	morpheme:TYPE/morpheme:TYPE/...
```

Example:

```text
аварийный	аварий:ROOT/н:SUFF/ый:END
ага-хан	ага:ROOT/-:HYPN/хан:ROOT
```

Supported morpheme types include `PREF`, `ROOT`, `STEM`, `SUFF`, `END`, `POST`, `LINK`, `INTERFIX`, and `HYPN`.

## Usage

There are two entry points:

- `src/run.py` — a standard run without cross-validation.
- `src/run_cv.py` — N-fold cross-validation.

Both support the same modes:

- `train` — train a model only.
- `inference` — load a model, annotate a test file with gold annotations, and compute evaluation metrics.
- `annotate` — load a model and annotate a test file without gold annotations.
- `train_and_inference` — train and evaluate a model.
- `train_and_annotate` — train a model and annotate a file without gold annotations.

### Running without cross-validation

```bash
python src/run.py train_and_annotate \
  --model resnet \
  --dataset bel \
  --train_file data/morphodict.txt \
  --test_file data/new_words.txt
```

By default, outputs from a standard run are saved to:

```text
models/{dataset}/{model}/{mode}/{with_lemma|no_lemma}
results/{dataset}/{model}/{mode}/{with_lemma|no_lemma}/predictions.csv
```

### Cross-validation

```bash
python src/run_cv.py train_and_inference \
  --model xlmr_large \
  --dataset bel \
  --num_folds 5
```

To run a single fold:

```bash
python src/run_cv.py annotate --model resnet --dataset bel --only_fold 1
```

Cross-validation outputs include the fold identifier in the final path component:

```text
models/{dataset}/{model}/{mode}/{with_lemma|no_lemma}/{fold}
results/{dataset}/{model}/{mode}/{with_lemma|no_lemma}/{fold}.csv
```

## Configuration

`src/datasets.json` specifies paths relative to `data/`:

```json
{
  "bel": {
    "train_file": "morphodict-bel/{fold}/train.txt",
    "test_file": "morphodict-bel/{fold}/test.txt"
  }
}
```

In `run_cv.py`, paths may contain `{fold}` and `{mode}` placeholders. For a standard run with `run.py`, the configuration must not require `{fold}`; if it does, provide `--train_file` and `--test_file` explicitly.

`src/models.json` specifies model architectures and hyperparameters:

```json
{
  "resnet": {
    "model_type": "resnet",
    "learning_rate": 0.001,
    "batch_size": 32
  }
}
```

Main fields:

- `model_type`: `xlmroberta`/`bert`, `resnet`, or `bilstm`.
- `model_name`: a base Hugging Face model identifier or local path for BERT-like models.
- `pretrained_model_path`: a checkpoint path for fine-tuning ResNet/BiLSTM models.
- `learning_rate`, `batch_size`, `hidden_dim`, `num_layers`, `dropout` — training and architecture parameters.

## Algorithms

`MorphBERT` uses Hugging Face's `AutoModelForTokenClassification`; tokens are aligned with word characters through `word_ids`. With `--use_lemma`, the complete lemma is added as an extra input token but is not decoded as part of the output.

`MorphResNet` is a character-level 1D ResNet model: character embeddings, a stack of residual Conv1D blocks, and a linear classifier at each position.

`MorphBiLSTM` is a character-level BiLSTM model: character embeddings, a bidirectional LSTM, and a linear classifier at each position.

All models predict BMES labels such as `B-ROOT`, `M-ROOT`, `E-ROOT`, and `S-ROOT`. After softmax, constrained decoding from `decode_bmes.py` is applied: Viterbi selects the most probable sequence that satisfies the allowed transitions between morpheme types. `STEM` is treated like `ROOT`, `INTERFIX` like `LINK`, and hyphens are encoded as `HYPN`.

## Multilingual BERT-like models

The `--use_language` flag in `run.py` and `run_cv.py` enables reading language codes
from the first column of the training and test files (without a header):

```text
rus	аварийный	аварий:ROOT/н:SUFF/ый:END
bel	дом	дом:ROOT
```

With `--use_language --use_lemma`, the input takes the form
`["rus", "аварийный", "а", "в", "а", "р", "и", "й", "н", "ы", "й"]`.
The language code and the complete lemma receive the label `"0"`; BMES labels apply only
to the characters of the word. The language code and the complete lemma are tokenized
using the model's regular tokenizer. Both prefixes are excluded from decoding and metric computation.
The flags are independent: `--use_language` also works without `--use_lemma`.

Example of training and evaluation on combined files containing multiple languages:

```bash
python src/run.py train_and_inference \
  --model xlmr_large --dataset main \
  --use_lemma --use_language \
  --train_file data/multilingual_train.txt \
  --test_file data/multilingual_test.txt
```

When subsequently running `inference` or `annotate`, pass the same flags
used during training. For `annotate`, two columns are sufficient:
`language<TAB>lemma` (a third column, if present, is ignored).
In `predict_batch`, language codes are passed as a separate list:
`model.predict_batch(["дом", "дом"], languages=["rus", "bel"])`.

A `language` column is added to the output CSV. Model and result directories
receive the `_with_language` suffix, for example, `with_lemma_with_language`.
Without `--use_language`, the original two-column format and output paths are preserved.
This flag is not supported for ResNet/BiLSTM models.

## Citation

If you use this code in your research, please cite the paper or papers most relevant to your work using the BibTeX entries below.

1. [**BERT-like Models for Slavic Morpheme Segmentation (ACL 2025)**](https://aclanthology.org/2025.acl-long.337/)

```bibtex
@inproceedings{morozov-etal-2025-bert,
  title = "{BERT}-like Models for {S}lavic Morpheme Segmentation",
  author = "Morozov, Dmitry and
    Astapenka, Lizaveta and
    Glazkova, Anna and
    Garipov, Timur and
    Lyashevskaya, Olga",
  editor = "Che, Wanxiang and
    Nabende, Joyce and
    Shutova, Ekaterina and
    Pilehvar, Mohammad Taher",
  booktitle = "Proceedings of the 63rd Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
  month = jul,
  year = "2025",
  address = "Vienna, Austria",
  publisher = "Association for Computational Linguistics",
  pages = "6795--6815",
  doi = "10.18653/v1/2025.acl-long.337",
  url = "https://aclanthology.org/2025.acl-long.337/"
}
```

2. [**Searching for a Unified Approach to Morpheme Segmentation: Benchmarking Supervised and LLM Architectures Across 20 Languages (Findings EMNLP 2026)**](https://aclanthology.org/2026.findings-emnlp.715/)

```bibtex
@inproceedings{morozov-etal-2026-searching,
  title = "Searching for a Unified Approach to Morpheme Segmentation: Benchmarking Supervised and {LLM} Architectures Across 20 Languages",
  author = "Morozov, Dmitry and
    Glazkova, Anna and
    Lyashevskaya, Olga",
  editor = "Lee, Hung-Yi and
    Dev, Sunipa and
    Li, Jessy and
    Elliott, Desmond",
  booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
  month = oct,
  year = "2026",
  address = "Budapest, Hungary",
  publisher = "Association for Computational Linguistics",
  pages = "15013--15037",
  doi = "10.18653/2026.findings-emnlp.715",
  url = "https://aclanthology.org/2026.findings-emnlp.715/"
}
```

3. [**Improving Low-Resource Morpheme Segmentation via Cross-Lingual Sequential Fine-Tuning (AIST 2026)**](https://link.springer.com/chapter/10.1007/978-3-032-41546-2_11)

```bibtex
@inproceedings{morozov-shaimukhamedov-2027-improving,
  title = "Improving Low-Resource Morpheme Segmentation via Cross-Lingual Sequential Fine-Tuning",
  author = "Morozov, Dmitry and
    Shaimukhamedov, Alexander",
  editor = "Tsymbalov, Evgenii and
    Gubanov, Dmitry and
    Ignatov, Dmitry I. and
    Kutuzov, Andrey and
    Loukachevitch, Natalia and
    Nikishina, Irina and
    Panchenko, Alexander and
    Panov, Maxim and
    Rykov, Elisei and
    Savchenko, Andrey and
    Takhanov, Rustem and
    Tutubalina, Elena",
  booktitle = "Analysis of Images, Social Networks and Texts",
  series = "Lecture Notes in Computer Science",
  volume = "16795",
  year = "2027",
  publisher = "Springer Nature Switzerland",
  address = "Cham",
  pages = "163--177",
  doi = "10.1007/978-3-032-41546-2_11",
  url = "https://link.springer.com/chapter/10.1007/978-3-032-41546-2_11"
}
```