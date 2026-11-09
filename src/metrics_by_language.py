#!/usr/bin/env python3
"""Посчитать метрики MorphBase отдельно для каждого языка в CSV.

Положите скрипт в src/ рядом с morph_base.py и decode_bmes.py.
Передайте --top_k K для расчёта WordAccuracy@K по predicted_text_topk.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from morph_base import MorphBase


def calculate_metrics(frame: pd.DataFrame, top_k: int = 1) -> pd.DataFrame:
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k должен быть целым числом >= 1.")

    prediction_column = "predicted_text_topk" if top_k > 1 else "predicted_text"
    required = {"language", "target_text", prediction_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"В CSV отсутствуют столбцы: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("В CSV нет строк с предсказаниями.")
    for column in ("language", "target_text"):
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"В столбце {column!r} есть пустые значения.")

    # Для оценки готового CSV достаточно задать режим расчёта метрик.
    evaluator = MorphBase.__new__(MorphBase)
    evaluator.top_k = top_k
    rows = []
    for language, group in frame.groupby("language", sort=True):
        predicted = group[prediction_column].tolist()
        if top_k > 1:
            predicted = [json.loads(value) for value in predicted]
            if any(
                not isinstance(candidates, list)
                or any(not isinstance(candidate, str) for candidate in candidates)
                for candidates in predicted
            ):
                raise ValueError(
                    f"Для языка {language!r} столбец {prediction_column} "
                    "должен содержать JSON-массивы строк."
                )
        metrics = evaluator._measure_quality(
            group["target_text"].tolist(),
            predicted,
        )
        rows.append({"language": language, "n_words": len(group), **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_file", type=Path, help="CSV с готовыми предсказаниями")
    parser.add_argument("--out", type=Path, help="Выходной CSV с метриками по языкам")
    parser.add_argument(
        "--top_k", type=int, default=1,
        help="Число кандидатов для WordAccuracy@k из predicted_text_topk (по умолчанию 1).",
    )
    args = parser.parse_args()

    # Пустое предсказание остаётся строкой, а не превращается в NaN.
    frame = pd.read_csv(args.predictions_file, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    result = calculate_metrics(frame, top_k=args.top_k)
    output = args.out or args.predictions_file.with_name(
        args.predictions_file.stem + ".metrics_by_language.csv"
    )
    if output.resolve() == args.predictions_file.resolve():
        raise ValueError("Путь к результату должен отличаться от пути к исходному CSV.")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8")
    print(result.to_string(index=False, float_format="{:.6f}".format))
    print(f"\nРезультат сохранён: {output}")


if __name__ == "__main__":
    main()
