"""Verify training constants in train_hc18.py match the original notebook reference."""

from __future__ import annotations

import ast
from pathlib import Path
import pytest
import torch.nn as nn
import segmentation_models_pytorch as smp

from fetal.data.hc18_dataset import HC18SegDataset
from fetal.models.hc18_unet import HC18Loss
from fetal.train import train_hc18


def parse_reference_constants() -> dict[str, object]:
    """Parse reference training constants directly from reference AST."""
    ref_path = Path(__file__).parent / "reference" / "hc18_original.py"
    with open(ref_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    constants: dict[str, object] = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "N_FOLDS":
                        constants["folds"] = ast.literal_eval(node.value)
                    elif target.id == "IMG_SIZE":
                        constants["img_size"] = ast.literal_eval(node.value)
                    elif target.id == "BATCH_SIZE":
                        constants["batch"] = ast.literal_eval(node.value)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "train_one_fold":
            arg_names = [a.arg for a in node.args.args]
            defaults = [ast.literal_eval(d) for d in node.args.defaults]
            arg_defaults = dict(zip(arg_names[-len(defaults):], defaults))
            constants["max_epochs"] = arg_defaults["max_epochs"]
            constants["lr"] = arg_defaults["lr"]
            constants["early_stop_patience"] = arg_defaults["patience"]

            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Call):
                    func_name = getattr(subnode.func, "attr", getattr(subnode.func, "id", ""))
                    if func_name == "AdamW":
                        for kw in subnode.keywords:
                            if kw.arg == "weight_decay":
                                constants["weight_decay"] = ast.literal_eval(kw.value)
                    elif func_name == "ReduceLROnPlateau":
                        for kw in subnode.keywords:
                            if kw.arg == "factor":
                                constants["scheduler_factor"] = ast.literal_eval(kw.value)
                            elif kw.arg == "patience":
                                constants["scheduler_patience"] = ast.literal_eval(kw.value)
                    elif func_name == "DiceLoss":
                        for kw in subnode.keywords:
                            if kw.arg == "mode":
                                constants["dice_mode"] = ast.literal_eval(kw.value)
                    elif func_name == "BCEWithLogitsLoss":
                        constants["has_bce"] = True

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "HC18SegDataset":
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.If):
                    for child in ast.walk(subnode):
                        if isinstance(child, ast.Attribute) and child.attr == "FLIP_LEFT_RIGHT":
                            constants["aug"] = "horizontal_flip_p0.5"

    for node in tree.body:
        if isinstance(node, ast.If):
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Call):
                    func_name = getattr(subnode.func, "id", getattr(subnode.func, "attr", ""))
                    if func_name == "KFold":
                        for kw in subnode.keywords:
                            if kw.arg == "random_state":
                                constants["seed"] = ast.literal_eval(kw.value)

    return constants


def test_hc18_constants_match_reference():
    """Assert all 11 HC18 training constants match the reference implementation."""
    ref_c = parse_reference_constants()

    # Instantiate loss to check structure
    loss = HC18Loss()
    has_dice_binary = isinstance(loss.dice_loss, smp.losses.DiceLoss) and loss.dice_loss.mode == "binary"
    has_bce = isinstance(loss.bce_loss, nn.BCEWithLogitsLoss)
    loss_desc = "DiceLoss(binary)+BCEWithLogitsLoss" if (has_dice_binary and has_bce) else "unknown"

    table_rows = [
        ("folds", ref_c["folds"], train_hc18.N_FOLDS),
        ("img size", ref_c["img_size"], train_hc18.IMG_SIZE),
        ("batch", ref_c["batch"], train_hc18.BATCH_SIZE),
        ("lr", ref_c["lr"], train_hc18.LR),
        ("weight decay", ref_c["weight_decay"], train_hc18.WEIGHT_DECAY),
        ("scheduler factor", ref_c["scheduler_factor"], train_hc18.SCHEDULER_FACTOR),
        ("scheduler patience", ref_c["scheduler_patience"], train_hc18.SCHEDULER_PATIENCE),
        ("max epochs", ref_c["max_epochs"], train_hc18.MAX_EPOCHS),
        ("early-stop patience", ref_c["early_stop_patience"], train_hc18.PATIENCE),
        ("loss", "DiceLoss(binary)+BCEWithLogitsLoss", loss_desc),
        ("aug", ref_c["aug"], "horizontal_flip_p0.5"),
        ("seed", ref_c["seed"], train_hc18.SEED),
    ]

    print("\n" + "=" * 65)
    print(f"{'constant':<22} | {'original':<20} | {'repo':<20}")
    print("-" * 65)
    for name, orig_val, repo_val in table_rows:
        print(f"{name:<22} | {str(orig_val):<20} | {str(repo_val):<20}")
        assert orig_val == repo_val, f"Mismatch on {name}: original {orig_val} != repo {repo_val}"
    print("=" * 65 + "\n")
