"""Run from BranchyNet: python -m src.measure --f-ref 2e9."""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


GLOBAL_PREFIXES = ("conv1", "bn1", "layer1", "layer2", "layer3.0", "layer3.1")


def is_global(name):
    return any(name == prefix or name.startswith(prefix + ".")
               for prefix in GLOBAL_PREFIXES)


def measure_size(model):
    parameters = {n: p for n, p in model.named_parameters() if is_global(n)}
    payload = {n: t for n, t in model.state_dict().items() if is_global(n)}
    parameter_bytes = sum(p.numel() * p.element_size() for p in parameters.values())
    payload_bytes = sum(t.numel() * t.element_size() for t in payload.values())
    return {
        "global_parameters": sum(p.numel() for p in parameters.values()),
        "parameters_only_bytes": parameter_bytes,
        "parameters_only_bits": parameter_bytes * 8,
        "parameters_and_buffers_bytes": payload_bytes,
        "parameters_and_buffers_bits": payload_bytes * 8,
    }


def forward_at(model, x, depth):
    if depth not in (20, 28, 34):
        raise ValueError("depth must be 20, 28, or 34")
    x = model.relu(model.bn1(model.conv1(x)))
    x = model.layer2(model.layer1(x))
    for block in model.layer3[:2]:
        x = block(x)
    if depth == 20:
        return model._classify(x, model.exit20)
    for block in model.layer3[2:]:
        x = block(x)
    if depth == 28:
        return model._classify(x, model.exit28)
    return model._classify(model.layer4(x), model.fc)


def measure_training(model, dataset, depth, epochs, batch_size,
                     make_optimizer, seed=0, repeats=3, warmup_steps=3):
    if len(dataset) == 0:
        raise ValueError("Cannot measure an empty client dataset")

    def run(warmup=False):
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, generator=generator)
        local = copy.deepcopy(model).cpu().train()
        optimizer = make_optimizer(local)
        elapsed = 0.0
        steps = 0
        for _ in range(warmup_steps if warmup else epochs):
            for images, labels in loader:
                images, labels = images.cpu(), labels.cpu()
                # Time computation only; data loading and model setup are excluded.
                start = time.perf_counter()
                optimizer.zero_grad()
                loss = F.cross_entropy(forward_at(local, images, depth), labels)
                loss.backward()
                optimizer.step()
                elapsed += time.perf_counter() - start
                steps += 1
                if warmup and steps >= warmup_steps:
                    return elapsed
        return elapsed

    run(warmup=True)
    samples = [run() for _ in range(repeats)]
    return {"seconds_median": statistics.median(samples), "seconds_runs": samples}


def main():
    from .dataset import get_datasets, split_dataset
    from .models import eenet34
    from .utils import load_config, select_optimizer, set_seed

    parser = argparse.ArgumentParser(description="Measure EENet34 shared size and CPU training")
    parser.add_argument("--f-ref", type=float, required=True,
                        help="Assumed reference cycles/second, e.g. 2e9")
    parser.add_argument("--client", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="results/measure.json")
    args = parser.parse_args()
    if args.f_ref <= 0 or args.repeats < 1:
        parser.error("f-ref and repeats must be positive")

    root = Path(__file__).resolve().parents[1]
    config = load_config(str(root))
    training, fl = config["training"], config["FL"]
    dataset_name = training["dataset_name"]
    if dataset_name not in ("CIFAR10", "CIFAR100"):
        parser.error("This measurement supports 3-channel CIFAR10/CIFAR100 inputs")
    batch_size, epochs = training["batch_size"], training["epochs"]
    if batch_size < 1 or epochs < 1:
        parser.error("batch_size and epochs must be positive")
    if not 0 <= args.client < fl["num_clients"]:
        parser.error("client index is outside num_clients")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    set_seed(args.seed)
    classes = 10 if dataset_name == "CIFAR10" else 100
    train_loader, _, _ = get_datasets(batch_size, dataset_name)
    subsets = split_dataset(train_loader.dataset, fl["num_clients"],
                            fl["dirichlet"], num_classes=classes)
    model = eenet34(num_classes=classes).cpu()
    make_optimizer = lambda local: select_optimizer(
        training["optim_name"], local, training["lr"])
    report = {
        "size": measure_size(model),
        "conditions": {
            "dataset": dataset_name, "client": args.client,
            "client_samples": len(subsets[args.client]), "local_epochs": epochs,
            "batch_size": batch_size, "optimizer": training["optim_name"],
            "lr": training["lr"], "seed": args.seed, "repeats": args.repeats,
            "cpu_threads": 1, "f_ref_assumed_cycles_per_second": args.f_ref,
            "loss": "selected exit cross entropy only",
            "timed_region": "forward, loss, backward, optimizer update",
            "cycles_kind": "equivalent estimate, not hardware counter cycles",
        },
        "exits": {},
    }
    print("Shared size:", report["size"], flush=True)
    for depth in (20, 28, 34):
        print(f"Measuring Exit{depth}...", flush=True)
        result = measure_training(model, subsets[args.client], depth, epochs,
                                  batch_size, make_optimizer, args.seed, args.repeats)
        result["equivalent_cycles"] = result["seconds_median"] * args.f_ref
        report["exits"][str(depth)] = result
        print(result, flush=True)
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
