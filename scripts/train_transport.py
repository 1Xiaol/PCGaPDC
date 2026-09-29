#!/usr/bin/env python3
"""Reference single-process trainer for the dense transport data contract."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import torch

from recdiffusion.checkpoints import save_transport
from recdiffusion.data import load_training_npz
from recdiffusion.io import read_json
from recdiffusion.transport import TransportConfig, build_transport, parameter_groups
from recdiffusion.training import rectified_flow_loss


def load_model_config(path: Path) -> tuple[str, TransportConfig]:
    raw = read_json(path)
    route = raw.get("route")
    if route not in ("history_conditioned_rfm", "dual_view_conditioned_rfm"):
        raise ValueError("model config must declare a supported route")
    names = {field.name for field in fields(TransportConfig)}
    missing = sorted(names - set(raw))
    if missing:
        raise ValueError(f"model config is missing fields: {missing}")
    return route, TransportConfig(**{name: raw[name] for name in names})


def load_training_config(path: Path) -> dict:
    raw = read_json(path)
    config = raw.get("transport")
    required = {
        "encoder_lr",
        "transport_lr",
        "info_nce_weight",
        "warmup_updates",
    }
    if not isinstance(config, dict) or required - set(config):
        raise ValueError(f"training config transport fields missing: {sorted(required - set(config or {}))}")
    return raw


def set_warmup_lr(optimizer: torch.optim.Optimizer, update: int, warmup_updates: int) -> None:
    factor = 1.0 if warmup_updates <= 0 else min(1.0, update / warmup_updates)
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * factor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--gradient-clip", type=float, required=True)
    parser.add_argument("--log-interval", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    arrays = load_training_npz(args.input)
    device = torch.device(args.device)
    route, model_config = load_model_config(args.model_config)
    training_config = load_training_config(args.training_config)
    train = training_config["transport"]
    if training_config.get("route") not in (None, route):
        raise ValueError("training and model routes differ")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = build_transport(route, model_config, seed=args.seed).to(device)
    optimizer = torch.optim.AdamW(
        parameter_groups(
            model,
            encoder_lr=float(train["encoder_lr"]),
            transport_lr=float(train["transport_lr"]),
        ),
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    users = len(arrays["source"])
    for update in range(args.updates):
        index = torch.randint(users, (args.batch_size,), generator=generator)
        batch = {
            name: torch.from_numpy(values[index.numpy()]).to(device)
            for name, values in arrays.items()
        }
        losses = rectified_flow_loss(
            model,
            batch["source"],
            batch["target"],
            batch["history"],
            batch["history_mask"],
            batch["evidence"],
            batch["evidence_mask"],
            context_weight=float(train["info_nce_weight"]),
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        set_warmup_lr(optimizer, update + 1, int(train["warmup_updates"]))
        optimizer.step()
        if update % args.log_interval == 0 or update + 1 == args.updates:
            print(json.dumps({"update": update + 1, **{k: float(v.detach()) for k, v in losses.items()}}))
    save_transport(args.output, model, training_config=training_config)
    print(json.dumps({"checkpoint": str(args.output), "route": route}))


if __name__ == "__main__":
    main()
