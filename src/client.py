import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import MODEL_SPECS


def train(
    optimizer, model, train_loader, device, model_name,
    frozen_prefixes=(),
):
    # 評価関数の定義
    criterion = nn.CrossEntropyLoss()

    # model 学習モードに設定
    model.train()

    if frozen_prefixes:
        for name, module in model.named_modules():
            if name and (name + ".").startswith(frozen_prefixes):
                module.eval()

    for i, (images, labels) in enumerate(train_loader):
        if model_name in ["DNN"]:
            # viewで1次元配列に変更
            images, labels = images.view(-1, 28*28).to(device), labels.to(device)
        
        else:
            images, labels = images.to(device), labels.to(device)
        
        # 勾配をリセット
        optimizer.zero_grad(set_to_none=True)

        if model_name in MODEL_SPECS:
            out_shallow, out_middle, out_final = model(images)
            loss = (
                0.2 * criterion(out_shallow, labels)
                + 0.3 * criterion(out_middle, labels)
                + 0.5 * criterion(out_final, labels)
            )
        
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        # 誤差逆伝播
        loss.backward()
        # パラメータ更新
        optimizer.step()


def test(model, test_loader, device):

    # model 評価モードに設定
    model.eval()

    with torch.no_grad():
        correct_preds = 0
        total_preds = 0

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if isinstance(outputs, (tuple, list)):
                outputs = outputs[-1]
            
            _, preds = torch.max(outputs, 1)
            correct_preds += preds.eq(labels).sum().item()
            total_preds += outputs.size(0)

        acc = 100*correct_preds/total_preds
    
    return acc


def early_test(model, test_loader, device, threshold, model_name):
    model.eval()
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if model_name in (
                "eenet20",
                "eenet34",
                "eenet34_1525",
                "eenet34_1525_half",
                "eenet34_1525_quarter",
            ):
                outputs = model.forward_early(inputs, threshold)
                preds = outputs.argmax(dim=1)

                correct_preds += preds.eq(labels).sum().item()
                total_preds += labels.size(0)

        acc = 100*correct_preds/total_preds
    
    return acc


def test_exit(model, test_loader, device, exit_layer):
    """指定したExitまでだけ計算して精度を評価する。"""
    allowed_exits = getattr(model, "exit_ids", (20, 28, 34))
    
    if exit_layer not in allowed_exits:
        raise ValueError(f"未対応のExit: {exit_layer}")

    model.eval()
    correct_preds = 0
    total_preds = 0

    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model.forward_exit(
                inputs, exit_layer
            )
            preds = outputs.argmax(dim=1)

            correct_preds += preds.eq(labels).sum().item()
            total_preds += labels.size(0)

    if total_preds == 0:
        raise ValueError("評価データがありません")

    return 100.0 * correct_preds / total_preds


def fine_tune_personal(
    optimizer,
    model,
    train_loader,
    device,
    model_name,
    shared_prefixes,
    epochs=1,
):
    original_flags = {
        name: p.requires_grad
        for name, p in model.named_parameters()
    }

    try:
        # 共有部分の重みを固定
        for name, p in model.named_parameters():
            if name.startswith(shared_prefixes):
                p.requires_grad_(False)

        for _ in range(epochs):
            train(
                optimizer,
                model,
                train_loader,
                device,
                model_name,
                frozen_prefixes=shared_prefixes,
            )

    finally:
        # 次回の通常学習に向けて元に戻す
        for name, p in model.named_parameters():
            p.requires_grad_(original_flags[name])