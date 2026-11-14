import torch
import math
import numpy as np


def average_models(client_updates, num_samples):
    if not client_updates or len(client_updates) != len(num_samples):
        raise ValueError("更新数とデータ数のリストを確認してください")

    if any(n <= 0 for n in num_samples):
        raise ValueError("各端末のデータ数は正の値が必要です")

    total = sum(num_samples)
    avg_state = {}

    for key in client_updates[0].keys():
        ref = client_updates[0][key]
        if not torch.is_floating_point(ref):
            avg_state[key] = ref.detach().clone()
            continue
 
        acc = ref.detach().clone().float() * num_samples[0]        # clone: 元を壊さない
        for i in range(1, len(num_samples)):
            acc = acc + client_updates[i][key].detach().float() * num_samples[i]   # 非 in-place
        avg_state[key] = acc.to(ref.dtype)

    return avg_state


def federated_learning(client_updates, global_model, num_samples):
    avg_state = average_models(client_updates, num_samples)
    global_model.load_state_dict(avg_state, strict=False)
    return global_model