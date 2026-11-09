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
            "Run a morpheme segmentation model on explicit train/test files. "
            "Use the positional command or --run_mode to choose whether to train, "
            "run inference, annotate, or combine these steps."
        )
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=RUN_MODES,
        help=(
            "Execution mode: 'train' only trains a model; 'inference' loads an "
            "existing model and predicts against gold labels; 'annotate' loads an "
            "existing model and annotates the test file without requiring gold "
            "labels; 'train_and_inference' trains then evaluates; "
            "'train_and_annotate' trains then annotates without gold labels."
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
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--top_k", type=int, default=1, help="Return k best valid Viterbi parsings; 1 keeps the original top-1 mode.")
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=SPLIT_MODES,
        help="Split mode for formatting dataset paths.",
    )
    parser.add_argument("--use_lemma", action="store_true", default=False, help="Add lemma to the features")
    parser.add_argument("--use_language", action="store_true", default=False, help="BERT only: read language in the first TSV column and add it to the features")
    parser.add_argument("--full_set", action="store_true", default=False, help="Use full set of types: PREF, ROOT, STEM, SUFF, END, HYPN, POST, LINK, INTERFIX")
    parser.add_argument("--train_file", type=Path, default=None, help="Training file path. Defaults to dataset train_file.")
    parser.add_argument("--test_file", type=Path, default=None, help="Test/annotation file path. Defaults to dataset test_file.")
    parser.add_argument("--model_dir", type=Path, default=None, help="Directory for saving/loading the model.")
    parser.add_argument("--results_dir", type=Path, default=None, help="Directory for prediction outputs and metrics.")
    parser.add_argument("--pretrained_dir", type=Path, default=None, help="Optional pretrained model directory.")

    return resolve_run_mode(parser.parse_args(), parser)


def get_model_dir(args: argparse.Namespace) -> Path:
    if args.model_dir is not None:
        return args.model_dir
    return Path("models") / args.dataset / args.model / args.mode / lemma_suffix(args)


def get_results_dir(args: argparse.Namespace) -> Path:
    if args.results_dir is not None:
        path = args.results_dir
    else:
        path = Path("results") / args.dataset / args.model / args.mode / lemma_suffix(args)
        if args.top_k > 1:
            path = path / f"top_k_{args.top_k}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_dataset_path(morph_model: MorphBase, key: str) -> Path:
    value = morph_model.config[key]
    if "{fold}" in value:
        raise ValueError(
            f"Dataset config '{key}' contains '{{fold}}'. "
            f"Pass --{key} explicitly or use src/run_cv.py."
        )
    return morph_model.data_dir / value.format(mode=morph_model.mode.value)


def get_train_file(args: argparse.Namespace, morph_model: MorphBase) -> Path:
    if args.train_file is not None:
        return args.train_file
    return format_dataset_path(morph_model, "train_file")


def get_test_file(args: argparse.Namespace, morph_model: MorphBase) -> Path:
    if args.test_file is not None:
        return args.test_file
    return format_dataset_path(morph_model, "test_file")


def resolve_pretrained_dir(args: argparse.Namespace, morph_model: MorphBase) -> Path | str | None:
    if args.pretrained_dir is not None:
        return args.pretrained_dir

    pretrained_model_path = morph_model.config.get("pretrained_model_path")
    if pretrained_model_path:
        if "{fold}" in pretrained_model_path:
            raise ValueError("pretrained_model_path contains '{fold}'. Pass --pretrained_dir explicitly.")
        return Path(pretrained_model_path.format(mode=morph_model.mode.value))

    model_name = morph_model.config.get("model_name")
    if isinstance(model_name, str) and "{fold}" in model_name:
        raise ValueError("model_name contains '{fold}'. Pass --pretrained_dir explicitly.")

    return None


def ensure_model_exists(model_dir: Path) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Cannot run inference: model directory does not exist: {model_dir}. "
            "Run train first or pass --model_dir."
        )


def run(args: argparse.Namespace) -> None:
    start_time = time.time()
    morph_model = make_model(args)
    model_dir = get_model_dir(args)
    results_dir = get_results_dir(args)
    predicted_file = results_dir / "predictions.csv"

    print(
        f"Starting run for model '{args.model}' on dataset '{args.dataset}'. "
        f"Run mode: {args.run_mode}; lemma used: {args.use_lemma}; language used: {args.use_language}."
    )

    if args.run_mode in {"train", "train_and_inference", "train_and_annotate"}:
        train_file = get_train_file(args, morph_model)
        pretrained_dir = resolve_pretrained_dir(args, morph_model)
        print("\nTraining")
        print(f"Model dir: {model_dir}")
        morph_model.train(train_file, model_dir, pretrained_dir=pretrained_dir)

    if args.run_mode in {"inference", "train_and_inference"}:
        test_file = get_test_file(args, morph_model)
        if args.run_mode == "inference":
            ensure_model_exists(model_dir)
        print("\nPredicting")
        print(f"Input file: {test_file}")
        print(f"Output file: {predicted_file}")
        morph_model.predict(model_dir, test_file, predicted_file)

    if args.run_mode == "annotate":
        ensure_model_exists(model_dir)

    if args.run_mode in {"annotate", "train_and_annotate"}:
        test_file = get_test_file(args, morph_model)
        print("\nAnnotating")
        print(f"Input file: {test_file}")
        print(f"Output file: {predicted_file}")
        morph_model.predict_file(test_file, predicted_file, model_dir=model_dir)

    print("\n" + "=" * 50)
    print("Run complete.")
    print(f"Run mode: {args.run_mode}")
    print(f"Total execution time: {format_duration(time.time() - start_time)}")
    print("=" * 50)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
