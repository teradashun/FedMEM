import copy
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .models import resnet18, resnet34, eenet34
from .dataset import get_datasets, split_dataset
from .utils import set_seed, choose_clients, select_optimizer, load_config, acc_average
from .trainer import train, test, early_test
from .server import federated_learning


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))   # BranchyNet/src
    parent_dir = os.path.dirname(current_dir)                  # BranchyNet
    sys.path.append(parent_dir)

    config = load_config(parent_dir)

    batch_size    = config['training']['batch_size']
    epochs        = config['training']['epochs']
    lr            = config['training']['lr']
    optimizer     = config['training']['optimizer']
    model_name    = config['training']['model_name']
    dataset       = config['training']['dataset']
    early_exit    = config['training']['early_exit']
    threshold     = config['training']['threshold']

    global_rounds = config['FL']['global_rounds']
    ite_num       = config['FL']['ite_num']


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader = get_datasets(batch_size, dataset, val_ratio=0.2)

    num_classes = 10 if dataset in ["MNIST", "CIFAR10"] else 100

    if model_name == "resnet18":
        model = resnet18(num_classes=num_classes).to(device)
    elif model_name == "resnet34":
        model = resnet34(num_classes=num_classes).to(device)
    elif model_name == "eenet34":
        model = eenet34(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    optimizer = select_optimizer(optimizer, model, lr)

    model.train()

    acc_history = [[] for _ in range(global_rounds)]

    for ite in range(ite_num):

        print(f"iteration {ite+1}/ {ite_num}")

        set_seed(ite)

        for round in range(global_rounds):

            print(f"Round {round+1}/ {global_rounds}")

            train(optimizer, model, train_loader, device, model_name)

            # モデルを評価
            if early_exit:
                test_acc = early_test(model, test_loader, device, threshold, model_name)
            else:
                test_acc = test(model, test_loader, device)

            acc_history[round].append(test_acc)
    
    final_acc = acc_average(acc_history)

    # 出力先のパス設定
    save_dir = os.path.join(
        parent_dir,
        "results",
        "early-exit"
        f"{dataset}_"
        f"{model_name}"
        )

    os.makedirs(save_dir, exist_ok=True)

    # csv出力
    df = pd.DataFrame({
        "round": np.arange(1, global_rounds + 1),
        "accuracy": final_acc
    })
    csv_path = os.path.join(save_dir, f"{model_name}.csv")
    df.to_csv(csv_path, index=False)

    #グラフ出力
    plt.figure(figsize=(10, 6))

    rounds = np.arange(1, global_rounds + 1) # グラフのX軸（ラウンド数）

    plt.plot(rounds, final_acc, linestyle='-', color='b', label='Early Exit Accuracy')

    # グラフのタイトルとラベル
    plt.title(f'{model_name}', fontsize=16)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True) # グリッド線を表示
    plt.legend() # 凡例

    # 範囲指定
    plt.xlim(0, global_rounds)
    plt.ylim(0, 100)

    combined_path = os.path.join(save_dir, f"{model_name}.png")

    plt.savefig(combined_path)
        