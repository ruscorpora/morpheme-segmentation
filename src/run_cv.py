import argparse
import time
from pathlib import Path

from morph_base import MorphBase
from run_common import (
    RUN_MODES,
    SPLIT_MODES,
    format_duration,
    lemma_suffix,
    make_model,
    resolve_run_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run N-fold cross-validation for a morpheme segmentation model. "
            "Use the positional command or --run_mode to choose whether to train, "
            "run inference, or do both."
        )
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=RUN_MODES,
        help=(
            "Execution mode: 'train' only trains fold models; 'inference' only "
            "loads existing fold models and predicts against gold labels; 'annotate' "
            "loads existing fold models and annotates the test file without requiring "
            "gold labels; 'train_and_inference' reproduces the old train-then-predict "
            "behavior; 'train_and_annotate' trains and then annotates without gold labels."
        ),
    )
    parser.add_argument(
        "--run_mode",
        choices=RUN_MODES,
        default=None,
        help="Same as the positional command; useful for scripts where named arguments are preferred.",
    )

    parser.add_argument("--model", type=str, required=True, help="Name of the model configuration from models.json")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset configuration from datasets.json")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs for each fold")
    parser.add_argument("--top_k", type=int, default=1, help="Return k best valid Viterbi parsings; 1 keeps the original top-1 mode.")
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=SPLIT_MODES,
        help="Split mode for data.",
    )
    parser.add_argument("--use_lemma", action="store_true", default=False, help="Add lemma to the features")
    parser.add_argument("--use_language", action="store_true", default=False, help="BERT only: read language in the first TSV column and add it to the features")
    parser.add_argument("--full_set", action="store_true", default=False, help="Use full set of types: PREF, ROOT, STEM, SUFF, END, HYPN, POST, LINK, INTERFIX")
    parser.add_argument("--only_fold", type=int, default=-1, help="If set, only runs fold with this number")

    args = resolve_run_mode(parser.parse_args(), parser)

    if args.num_folds < 1:
        parser.error("--num_folds must be >= 1.")
    if args.only_fold != -1 and not (1 <= args.only_fold <= args.num_folds):
        parser.error("--only_fold must be between 1 and --num_folds, or -1 to run all folds.")

    return args


def get_folds(num_folds: int, only_fold: int) -> list[int]:
    if only_fold != -1:
        return [only_fold]
    return list(range(1, num_folds + 1))


def get_fold_model_dir(args: argparse.Namespace, fold: int) -> Path:
    return (
        Path("models")
        / args.dataset
        / args.model
        / args.mode
        / lemma_suffix(args)
        / str(fold)
    )


def get_fold_results_dir(args: argparse.Namespace) -> Path:
    path = (
        Path("results")
        / args.dataset
        / args.model
        / args.mode
        / lemma_suffix(args)
    )
    if args.top_k > 1:
        path = path / f"top_k_{args.top_k}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_dataset_path(morph_model: MorphBase, key: str, fold: int) -> Path:
    return morph_model.data_dir / morph_model.config[key].format(
        fold=fold,
        mode=morph_model.mode.value,
    )


def resolve_pretrained_dir(morph_model: MorphBase, fold: int) -> Path | str | None:
    pretrained_model_path = morph_model.config.get("pretrained_model_path")
    if pretrained_model_path:
        return Path(pretrained_model_path.format(fold=fold, mode=morph_model.mode.value))

    model_name = morph_model.config.get("model_name")
    if isinstance(model_name, str) and "{fold}" in model_name:
        return model_name.format(fold=fold, mode=morph_model.mode.value)

    return None


def ensure_fold_model_exists(model_dir: Path, fold: int) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Cannot run inference for fold {fold}: model directory does not exist: {model_dir}. "
            "Run this fold in train mode first."
        )


def annotate_fold_test_file(morph_model: MorphBase, test_file: Path, predicted_file: Path, model_dir: Path, fold: int) -> None:
    print(f"\nAnnotating Fold {fold}")
    print(f"Input file: {test_file}")
    print(f"Output file: {predicted_file}")
    morph_model.predict_file(test_file, predicted_file, model_dir=model_dir)


def run_fold(args: argparse.Namespace, fold: int) -> None:
    fold_start_time = time.time()
    print("\n" + "=" * 50)
    print(f" FOLD {fold} of {args.num_folds}")
    print("=" * 50)

    morph_model = make_model(args)
    model_dir = get_fold_model_dir(args, fold)
    results_dir = get_fold_results_dir(args)
    predicted_file = results_dir / f"{fold}.csv"

    if args.run_mode in {"train", "train_and_inference", "train_and_annotate"}:
        train_file = format_dataset_path(morph_model, "train_file", fold)
        pretrained_dir = resolve_pretrained_dir(morph_model, fold)
        print(f"\nTraining Fold {fold}")
        morph_model.train(train_file, model_dir, pretrained_dir=pretrained_dir)

    if args.run_mode in {"inference", "train_and_inference"}:
        test_file = format_dataset_path(morph_model, "test_file", fold)
        if args.run_mode == "inference":
            ensure_fold_model_exists(model_dir, fold)
        print(f"\nPredicting Fold {fold}")
        morph_model.predict(model_dir, test_file, predicted_file)

    if args.run_mode == "annotate":
        ensure_fold_model_exists(model_dir, fold)

    if args.run_mode in {"annotate", "train_and_annotate"}:
        test_file = format_dataset_path(morph_model, "test_file", fold)
        annotate_fold_test_file(morph_model, test_file, predicted_file, model_dir, fold)

    print(f"\nFold {fold} completed in {format_duration(time.time() - fold_start_time)}.")


def main() -> None:
    args = parse_args()
    start_time = time.time()
    selected_folds = get_folds(args.num_folds, args.only_fold)

    print(
        f"Starting {args.num_folds}-fold cross-validation for model '{args.model}' "
        f"on dataset '{args.dataset}' in split-by-{args.mode} scenario. "
        f"Run mode: {args.run_mode}; lemma used: {args.use_lemma}; language used: {args.use_language}; "
        f"folds: {selected_folds}."
    )

    for fold in selected_folds:
        run_fold(args, fold)

    print("\n" + "=" * 50)
    print("Cross-validation complete.")
    print(f"Run mode: {args.run_mode}")
    print(f"Total execution time: {format_duration(time.time() - start_time)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
