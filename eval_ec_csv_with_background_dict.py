import ast
import json

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer


def ec_to_level(ec, level):
    """Extract EC prefix up to a given level, e.g. 3.2.1.21 -> 3.2.1 for level=3."""
    if not isinstance(ec, str) or not ec.strip():
        return None
    parts = ec.strip().split(".")
    if len(parts) < level:
        return None
    return ".".join(parts[:level])


def collect_level_ec(ec_list, level):
    """Collect all EC labels at a given level from an EC list (supports ';' separated ECs)."""
    result = set()
    if not ec_list:
        return result

    for ec in ec_list:
        if not ec or not isinstance(ec, str):
            continue
        sub_ecs = [e.strip() for e in ec.split(";") if e.strip()]
        for sub_ec in sub_ecs:
            lvl = ec_to_level(sub_ec, level)
            if lvl:
                result.add(lvl)
    return result


def safe_eval(x):
    """Safely parse a Python-literal string to a Python object."""
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except Exception:
        return []


def get_available_methods(df):
    """Return method columns that exist in CSV, preserving preferred order."""
    preferred_methods = [
        "Clean",
        "GraphEC",
        "DeepECTransformer",
        "Proteinfer",
        "RAMER Top1",
        "RAMER Max-sep",
        "protrek",
        # Backward-compatible fallback names:
        "RAM-enzyme Top1",
        "RAM-enzyme Max-sep",
    ]
    return [m for m in preferred_methods if m in df.columns]


def evaluate_ec_csv_with_background_dict(
    csv_path,
    background_library_dict_json,
    levels=(1, 2, 3, 4),
):
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if "true_ec" not in df.columns:
        raise ValueError("CSV must contain a 'true_ec' column.")
    df["true_ec_list"] = df["true_ec"].apply(safe_eval)

    print(f"Loading background library dict: {background_library_dict_json}")
    with open(background_library_dict_json, "r") as f:
        background_library_dict = json.load(f)

    if not isinstance(background_library_dict, dict):
        raise ValueError("Background library JSON must be a dict: {protein_id: [EC,...]}.")

    methods = get_available_methods(df)
    if not methods:
        raise ValueError("No valid method columns found in CSV.")

    print("Data loaded successfully.\n")
    print(f"Detected method columns: {methods}\n")

    for level in levels:
        print("=" * 50)
        print(f"EC Level {level} Evaluation")
        print("=" * 50)

        # Collect all background labels at the current EC level.
        background_ec_level = set()
        for ecs in background_library_dict.values():
            if isinstance(ecs, str):
                ecs = [ecs]
            elif not isinstance(ecs, list):
                continue
            background_ec_level.update(collect_level_ec(ecs, level))

        print(f"Background label count at level {level}: {len(background_ec_level)}\n")

        for method in methods:
            df[f"{method}_list"] = df[method].apply(safe_eval)

            y_true = [collect_level_ec(ec_list, level) for ec_list in df["true_ec_list"]]
            y_pred = [collect_level_ec(pred, level) for pred in df[f"{method}_list"]]

            # Keep label space consistent across background, truth, and prediction.
            all_labels = background_ec_level.union(*y_true).union(*y_pred)

            if not all_labels:
                print(f"{method:20}  P=0.0000  R=0.0000  F1=0.0000")
                continue

            mlb = MultiLabelBinarizer()
            mlb.fit([list(all_labels)])

            true_m = mlb.transform(y_true)
            pred_m = mlb.transform(y_pred)

            precision = precision_score(true_m, pred_m, average="weighted", zero_division=0)
            recall = recall_score(true_m, pred_m, average="weighted", zero_division=0)
            f1 = f1_score(true_m, pred_m, average="weighted", zero_division=0)

            print(f"{method:20}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")

        print()

    print("Evaluation finished.")


if __name__ == "__main__":
    # Input paths
    csv_path = "/root/fsas/A_RAMER/data/test_data/new392_result.csv"
    background_library_dict_json = "/root/fsas/A_RAMER/Background_library/clean100_set_dict.json"

    evaluate_ec_csv_with_background_dict(
        csv_path=csv_path,
        background_library_dict_json=background_library_dict_json,
    )
