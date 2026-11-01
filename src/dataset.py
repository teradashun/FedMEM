import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, random_split


def get_datasets(batch_size, data_name, val_ratio=0.0):
    if data_name == "MNIST":
        full_train_dataset = torchvision.datasets.MNIST(root="./data",
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)

        test_dataset = torchvision.datasets.MNIST(root="./data",
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)
    
    elif data_name == "CIFAR10":
        full_train_dataset = torchvision.datasets.CIFAR10(root="./data/CIFAR_10",
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)

        test_dataset = torchvision.datasets.CIFAR10(root="./data/CIFAR_10",
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)
    
    elif data_name == "CIFAR100":
        full_train_dataset = torchvision.datasets.CIFAR100(root="./data/CIFAR_100",
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)

        test_dataset = torchvision.datasets.CIFAR100(root="./data/CIFAR_100",
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)
    
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")
    
    val_size = int(len(full_train_dataset) * val_ratio)
    train_size = len(full_train_dataset) - val_size

    if val_size == 0:
        train_dataset = full_train_dataset
        val_dataset = Subset(full_train_dataset, [])
    else:
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def split_dataset(dataset, num_clients, dirichlet_alpha, num_classes=10):
    class_indices = {i: [] for i in range(num_classes)}

    for idx, (_, label) in enumerate(dataset):
        class_indices[int(label)].append(idx)
    
    for indices in class_indices.values():
        np.random.shuffle(indices)
    
    client_indices = {i: [] for i in range(num_clients)}

    for indices in class_indices.values():
        proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)

        for i, idx in enumerate(split_indices):
            client_indices[i].extend(idx)
    
    subsets = [Subset(dataset, client_indices[i]) for i in range(num_clients)]
    return subsets


def dataset_labels(dataset):
    """入れ子のSubsetにも対応してラベルを取得する。"""
    if isinstance(dataset, Subset):
        indices = np.asarray(dataset.indices, dtype=np.int64)
        return dataset_labels(dataset.dataset)[indices]

    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)

    return np.asarray(
        [int(dataset[i][1]) for i in range(len(dataset))],
        dtype=np.int64,
    )


def original_indices(dataset):
    """元のデータ集合におけるインデックスを取得する。"""
    if isinstance(dataset, Subset):
        indices = np.asarray(dataset.indices, dtype=np.int64)
        return original_indices(dataset.dataset)[indices]

    return np.arange(len(dataset), dtype=np.int64)


def split_client_tests(
    train_subsets, test_dataset, *, seed, num_classes=10
):
    if not train_subsets or any(len(s) == 0 for s in train_subsets):
        raise ValueError("学習データが空の端末があります")

    train_counts = np.stack([
        np.bincount(dataset_labels(s), minlength=num_classes)
        for s in train_subsets
    ])
    test_labels = dataset_labels(test_dataset)

    if train_counts.shape[1] != num_classes or np.any(
        (test_labels < 0) | (test_labels >= num_classes)
    ):
        raise ValueError("クラス番号を確認してください")

    # 学習に使うグローバルな乱数状態を変更しない。
    rng = np.random.default_rng(seed)
    assigned = [[] for _ in train_subsets]
    test_counts = np.zeros_like(train_counts)

    for c in range(num_classes):
        indices = np.flatnonzero(test_labels == c)
        total = int(train_counts[:, c].sum())

        if total == 0:
            if len(indices):
                raise ValueError(
                    "テストに存在するクラスが学習側にありません"
                )
            continue

        # 整数演算で比例配分と端数を計算する。
        numerators = train_counts[:, c] * len(indices)
        counts = numerators // total
        remainders = numerators % total
        leftover = len(indices) - int(counts.sum())

        # 端数の降順。同値なら端末番号の昇順。
        order = np.lexsort((
            np.arange(len(train_subsets)),
            -remainders,
        ))
        counts[order[:leftover]] += 1

        rng.shuffle(indices)
        start = 0

        for i, count in enumerate(counts):
            stop = start + int(count)
            assigned[i].extend(indices[start:stop].tolist())
            test_counts[i, c] = count
            start = stop

    flat = np.asarray(
        [j for part in assigned for j in part],
        dtype=np.int64,
    )

    if not np.array_equal(
        np.sort(flat), np.arange(len(test_dataset))
    ):
        raise RuntimeError("テスト分割に重複または欠落があります")

    if any(len(part) == 0 for part in assigned):
        raise ValueError("テストデータが空の端末があります")

    test_subsets = [
        Subset(test_dataset, part) for part in assigned
    ]
    return test_subsets, train_counts, test_counts