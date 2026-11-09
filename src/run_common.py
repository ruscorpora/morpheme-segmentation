import argparse
import json
import time
from typing import Type

from morph_base import MorphBase, SplitMode


RUN_MODES = ("train", "inference", "annotate", "train_and_inference", "train_and_annotate")
SPLIT_MODES = ("random", "roots")


def format_duration(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def get_model_class(model_config_name: str) -> Type[MorphBase]:
    """Choose a model implementation from the model config name."""
    try:
        with open("src/models.json", "r", encoding="utf-8") as f:
            model_type = json.load(f)[model_config_name].get("model_type")
    except KeyError:
        raise ValueError(f"Model configuration '{model_config_name}' not found in src/models.json") from None

    if model_type in {"xlmroberta", "bert"}:
        from morph_bert import MorphBERT

        print(f"Initializing MorphBERT (Transformer) for config: {model_config_name}")
        return MorphBERT
    if model_type == "resnet":
        from morph_resnet import MorphResNet

        print(f"Initializing MorphResNet for config: {model_config_name}")
        return MorphResNet
    if model_type in {"bilstm"}:
        from morph_lstm import MorphBiLSTM

        print(f"Initializing MorphBiLSTM for config: {model_config_name}")
        return MorphBiLSTM

    raise ValueError(
        f"Unsupported model_type '{model_type}' for config '{model_config_name}'. "
        "Expected one of: xlmroberta, bert, resnet, bilstm."
    )


def resolve_run_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    if args.top_k < 1:
        parser.error("--top_k must be >= 1.")
    if args.command and args.run_mode and args.command != args.run_mode:
        parser.error(f"Conflicting run modes: positional '{args.command}' and --run_mode '{args.run_mode}'.")

    if args.command:
        args.run_mode = args.command
    elif args.run_mode:
        pass
    else:
        args.run_mode = "train_and_inference"

    return args


def make_model(args: argparse.Namespace) -> MorphBase:
    ModelClass = get_model_class(args.model)
    return ModelClass(
        model_config_name=args.model,
        dataset_config_name=args.dataset,
        num_epochs=args.epochs,
        mode=SplitMode(args.mode),
        use_lemma=args.use_lemma,
        use_language=args.use_language,
        full_set=args.full_set,
        top_k=args.top_k,
    )


def lemma_suffix(args: argparse.Namespace) -> str:
    suffix = "with_lemma" if args.use_lemma else "no_lemma"
    return suffix + "_with_language" if args.use_language else suffix
