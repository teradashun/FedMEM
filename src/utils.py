import os
import yaml
import random
import numpy as np
import torch
import torch.optim as optim


def load_config(dir):   #引数のパスはFedAvg
    config_path = os.path.join(dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_optimizer(optim_name, local_model, lr):
    if optim_name == "SGD":
        return optim.SGD(local_model.parameters(), lr=lr)
    
    elif optim_name == "Adam":
        return optim.Adam(local_model.parameters(), lr=lr)
    
    else:
        raise NameError(f"オプティマイザの名前 '{optim_name}' はサポートされていません")


def choose_clients(num_clients, cohort):
    return random.sample(range(num_clients), cohort)


def acc_average(acc_history: list[list]) -> list:
    acc_history = np.array(acc_history)
    final_acc = np.mean(acc_history, axis=1)
    return final_acc


def generate_client_distances(
    num_clients,
    mean=0.0,
    std=150.0,
    min_distance=35.0,
    max_distance=300.0,
):
    distances = []

    while len(distances) < num_clients:

        # Step 1: X, Yを正規分布から生成
        x = np.random.normal(mean, std)
        y = np.random.normal(mean, std)

        # Step 2: 基地局(0, 0)からの距離
        distance = np.sqrt(x**2 + y**2)

        # Step 3: 35m以上1000m以下なら採用
        if min_distance <= distance <= max_distance:
            distances.append(distance)

    return distances


def generate_channel_gain(distance_m, RXG=0.0, TXG=0.0):
    # m → km
    distance_km = distance_m / 1000.0

    # RIS.pyと同じパスロスモデル
    PL_dB = 128.1 + 37.6 * np.log10(distance_km)

    BS_Gain = 10 ** (RXG / 10)
    User_Gain = 10 ** (TXG / 10)

    G_link_lin = (
        BS_Gain
        * User_Gain
        * 10 ** (-PL_dB / 10.0)
    )

    # パスロスによる振幅減衰
    path_gain = np.sqrt(G_link_lin)

    # Rayleigh fading
    h = (
        np.random.randn()
        + 1j * np.random.randn()
    ) / np.sqrt(2)

    return path_gain * h


def calculate_percentile_thresholds(model, val_loader, device, num_thresholds=10):
    """
    検証データからエントロピーの分布を取得し、パーセンタイルに基づいた閾値を計算する
    """
    model.eval()
    entropies = []
    
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            out_1, _ = model(inputs)
            
            # Softmaxからエントロピーを計算
            probs = torch.softmax(out_1, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            entropies.extend(entropy.cpu().numpy())
            
    # エントロピーの分布から分位点（パーセンタイル）を計算して閾値とする
    entropies = np.array(entropies)
    
    # 0%から100%まで、num_thresholds個の等間隔なパーセンタイルを生成
    percentiles = np.linspace(0, 100, num_thresholds)
    
    # 各パーセンタイルに対応するエントロピーの値を閾値として取得
    thresholds = np.percentile(entropies, percentiles)
    
    return thresholds


def calculate_b_alexnet_thresholds(model, val_loader, device, num_thresholds=8):
    """
    2つのExit層に対して、パーセンタイルに基づいた閾値を計算する
    """
    model.eval()
    entropies_exit1 = []
    entropies_exit2 = []

    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            out_1, out_2, _ = model(inputs)
        
            prob1 = torch.softmax(out_1, dim=1)
            entropy1 = -torch.sum(prob1 * torch.log(prob1 + 1e-8), dim=1)
            entropies_exit1.extend(entropy1.cpu().numpy())
            
            # Exit 2 のエントロピー
            prob2 = torch.softmax(out_2, dim=1)
            entropy2 = -torch.sum(prob2 * torch.log(prob2 + 1e-8), dim=1)
            entropies_exit2.extend(entropy2.cpu().numpy())

    percentiles = np.linspace(0, 100, num_thresholds)
    
    th1_list = np.percentile(entropies_exit1, percentiles)
    th2_list = np.percentile(entropies_exit2, percentiles)

    # list(list)形式にする
    thresholds = [[th1, th2] for th1, th2 in zip(th1_list, th2_list)]

    # 最後に全てMain Exitに行く閾値を追加
    thresholds.append([float('inf'), float('inf')])

    return thresholds


def calculate_b_resnet_thresholds(model, val_loader, device, num_thresholds=8):
    """
    3つのExit層に対して、パーセンタイルに基づいた閾値を計算する
    """
    model.eval()
    entropies_exit1 = []
    entropies_exit2 = []
    entropies_exit3 = []

    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            out_1, out_2, out_3, _ = model(inputs)
        
            prob1 = torch.softmax(out_1, dim=1)
            entropy1 = -torch.sum(prob1 * torch.log(prob1 + 1e-8), dim=1)
            entropies_exit1.extend(entropy1.cpu().numpy())
            
            # Exit 2 のエントロピー
            prob2 = torch.softmax(out_2, dim=1)
            entropy2 = -torch.sum(prob2 * torch.log(prob2 + 1e-8), dim=1)
            entropies_exit2.extend(entropy2.cpu().numpy())

            # Exit 3 のエントロピー
            prob3 = torch.softmax(out_3, dim=1)
            entropy3 = -torch.sum(prob3 * torch.log(prob3 + 1e-8), dim=1)
            entropies_exit3.extend(entropy3.cpu().numpy())

    percentiles = np.linspace(0, 100, num_thresholds)
    
    th1_list = np.percentile(entropies_exit1, percentiles)
    th2_list = np.percentile(entropies_exit2, percentiles)
    th3_list = np.percentile(entropies_exit3, percentiles)

    # list(list)形式にする
    thresholds = [[th1, th2, th3] for th1, th2, th3 in zip(th1_list, th2_list, th3_list)]

    # 最後に全てMain Exitに行く閾値を追加
    thresholds.append([float('inf'), float('inf'), float('inf')])

    return thresholds