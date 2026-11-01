import torch


def average_models(client_updates, num_samples):
    if not client_updates or len(client_updates) != len(num_samples):
        raise ValueError("更新数とデータ数のリストを確認してください")

    if any(n <= 0 for n in num_samples):
        raise ValueError("各端末のデータ数は正の値が必要です")

    total = sum(num_samples)
    avg_state = {}

    for key in client_updates[0]:
        values = [update[key] for update in client_updates]

        if values[0].is_floating_point():
            result = torch.zeros_like(values[0])

            for value, n in zip(values, num_samples):
                result.add_(value, alpha=n / total)

            avg_state[key] = result
        else:
            # 整数バッファは、これまでと同じ最大値の方針
            avg_state[key] = torch.stack(values).amax(dim=0)

    return avg_state


def federated_learning(client_updates, global_model, num_samples):
    avg_state = average_models(client_updates, num_samples)
    global_model.load_state_dict(avg_state, strict=False)
    return global_model