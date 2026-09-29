#!/usr/bin/env python3
"""Train conditional source calibration against a frozen transport."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import torch

from recdiffusion.checkpoints import load_transport, save_source_calibration
from recdiffusion.data import load_source_calibration_npz
from recdiffusion.geometry import spherical_heun
from recdiffusion.io import read_json
from recdiffusion.source_calibration import (
    ConditionalSourceCalibration,
    SourceCalibrationConfig,
    source_calibration_loss,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--heun-steps", type=int, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--gradient-clip", type=float, required=True)
    parser.add_argument("--log-interval", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    arrays = load_source_calibration_npz(args.input)
    device = torch.device(args.device)
    transport = load_transport(args.transport, device).eval()
    transport.requires_grad_(False)
    model_raw = read_json(args.model_config)
    names = {field.name for field in fields(SourceCalibrationConfig)}
    missing = sorted(names - set(model_raw))
    if missing:
        raise ValueError(f"source-calibration model config is missing fields: {missing}")
    config = SourceCalibrationConfig(**{name: model_raw[name] for name in names})
    training_config = read_json(args.training_config)
    train = training_config.get("source_calibration")
    required = {"batch_size", "warmup_updates"}
    if not isinstance(train, dict) or required - set(train):
        raise ValueError(f"training config source_calibration fields missing: {sorted(required - set(train or {}))}")
    expected_cond_dim = transport.config.condition_dim * (transport.config.global_tokens + 1)
    if config.cond_dim != expected_cond_dim or config.semantic_dim != transport.config.semantic_dim:
        raise ValueError("source-calibration dimensions do not match the transport")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    calibration = ConditionalSourceCalibration(config).to(device)
    optimizer = torch.optim.AdamW(
        calibration.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    users = len(arrays["source"])
    for update in range(args.updates):
        index = torch.randint(users, (int(train["batch_size"]),), generator=generator)
        batch = {
            name: torch.from_numpy(values[index.numpy()]).to(device)
            for name, values in arrays.items()
        }
        with torch.no_grad():
            memory = transport.encode(
                batch["history"],
                batch["history_mask"],
                batch["evidence"],
                batch["evidence_mask"],
            )
        adjusted, movement = calibration(batch["source"], memory, return_movement=True)
        generated = spherical_heun(
            lambda value, time: transport.velocity(value, time, memory),
            adjusted,
            steps=args.heun_steps,
        )
        losses = source_calibration_loss(
            generated, batch["target"], batch["target_valid"], movement
        )
        loss = losses["loss"].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(calibration.parameters(), args.gradient_clip)
        factor = 1.0 if int(train["warmup_updates"]) <= 0 else min(
            1.0, (update + 1) / int(train["warmup_updates"])
        )
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * factor
        optimizer.step()
        if update % args.log_interval == 0 or update + 1 == args.updates:
            record = {"update": update + 1, **{name: float(value.mean().detach()) for name, value in losses.items()}}
            print(json.dumps(record))
    save_source_calibration(
        args.output,
        calibration,
        transport_route=transport.route,
        transport_config=transport.config,
        training_config=training_config,
    )
    print(json.dumps({"checkpoint": str(args.output)}))


if __name__ == "__main__":
    main()
