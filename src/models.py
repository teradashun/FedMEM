import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=100,
        channels=(64, 128, 256, 512),
    ):
        super().__init__()

        if len(channels) != 4 or any(c <= 0 for c in channels):
            raise ValueError("正のチャネル数を4個指定してください")

        c1, c2, c3, c4 = channels
        self.in_channels = c1

        self.conv1 = nn.Conv2d(
            3, c1,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(c1)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(
            block, c1, num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, c2, num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, c3, num_blocks[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, c4, num_blocks[3], stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c4 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class CifarBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        if stride not in (1, 2):
            raise ValueError("strideは1または2を指定してください")
        if out_channels < in_channels:
            raise ValueError("出力チャネル数を減らす構造には対応していません")

        self.stride = stride
        self.channel_padding = out_channels - in_channels

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Option A：空間方向の間引き＋チャネル方向のゼロ埋め
        shortcut = x[:, :, ::self.stride, ::self.stride]

        if self.channel_padding:
            left = self.channel_padding // 2
            right = self.channel_padding - left
            shortcut = F.pad(
                shortcut, (0, 0, 0, 0, left, right)
            )

        return F.relu(out + shortcut)


class EENet20(nn.Module):
    exit_ids = (10, 16, 20)

    def __init__(
        self,
        num_classes=10,
        channels=(16, 32, 64),
    ):
        super().__init__()
        c1, c2, c3 = channels
        self.in_channels = c1

        self.conv1 = nn.Conv2d(
            3, c1,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(c1)

        self.layer1 = self._make_layer(c1, stride=1)
        self.layer2 = self._make_layer(c2, stride=2)
        self.layer3 = self._make_layer(c3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.exit10 = nn.Linear(c2, num_classes)
        self.exit16 = nn.Linear(c3, num_classes)
        self.fc = nn.Linear(c3, num_classes)

    def _make_layer(self, out_channels, stride):
        blocks = []

        for block_stride in (stride, 1, 1):
            blocks.append(
                CifarBasicBlock(
                    self.in_channels,
                    out_channels,
                    stride=block_stride,
                )
            )
            self.in_channels = out_channels

        return nn.Sequential(*blocks)

    def _classify(self, x, head):
        return head(torch.flatten(self.avgpool(x), 1))

    def _shared_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        return self.layer2[0](x)

    def _to_exit16(self, x):
        x = self.layer2[1:](x)
        return self.layer3[0](x)

    def forward(self, x):
        x = self._shared_features(x)
        out10 = self._classify(x, self.exit10)

        x = self._to_exit16(x)
        out16 = self._classify(x, self.exit16)

        x = self.layer3[1:](x)
        out20 = self._classify(x, self.fc)

        return out10, out16, out20

    def forward_exit(self, x, exit_id):
        if exit_id not in self.exit_ids:
            raise ValueError(f"未対応のExit: {exit_id}")

        x = self._shared_features(x)

        if exit_id == 10:
            return self._classify(x, self.exit10)

        x = self._to_exit16(x)

        if exit_id == 16:
            return self._classify(x, self.exit16)

        x = self.layer3[1:](x)
        return self._classify(x, self.fc)

    @torch.no_grad()
    def forward_early(self, x, thresholds):
        if len(thresholds) != 2:
            raise ValueError("閾値は2個指定してください")

        result = x.new_empty((x.size(0), self.fc.out_features))
        remaining = torch.arange(x.size(0), device=x.device)

        x = self._shared_features(x)

        for index, head in enumerate((self.exit10, self.exit16)):
            if index == 1:
                x = self._to_exit16(x)

            logits = self._classify(x, head)
            log_probs = F.log_softmax(logits, dim=1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=1)
            done = entropy <= thresholds[index]

            result[remaining[done]] = logits[done]
            remaining = remaining[~done]
            x = x[~done]

            if remaining.numel() == 0:
                return result

        x = self.layer3[1:](x)
        result[remaining] = self._classify(x, self.fc)
        return result


class EENet34(ResNet):
    exit_ids = (20, 28, 34)
    
    def __init__(self, num_classes=10):
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes)
        self.exit20 = nn.Linear(256, num_classes)
        self.exit28 = nn.Linear(256, num_classes)

    def _classify(self, x, head):
        return head(torch.flatten(self.avgpool(x), 1))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)

        for block in self.layer3[:2]:
            x = block(x)
        out20 = self._classify(x, self.exit20)

        for block in self.layer3[2:]:
            x = block(x)
        out28 = self._classify(x, self.exit28)

        x = self.layer4(x)
        out_final = self._classify(x, self.fc)

        return out20, out28, out_final


    def forward_exit(self, x, exit_id):
        """指定したExitまでだけ計算する。"""
        if exit_id not in (20, 28, 34):
            raise ValueError(f"未対応のExit: {exit_id}")

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)

        for block in self.layer3[:2]:
            x = block(x)

        if exit_id == 20:
            return self._classify(x, self.exit20)

        for block in self.layer3[2:]:
            x = block(x)

        if exit_id == 28:
            return self._classify(x, self.exit28)

        x = self.layer4(x)
        return self._classify(x, self.fc)


    @torch.no_grad()
    def forward_early(self, x, thresholds):
        # エントロピーが閾値以下のサンプルを途中で終了する。
        # model.eval()で呼び出す。
        result = x.new_empty((x.size(0), self.fc.out_features))
        remaining = torch.arange(x.size(0), device=x.device)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)

        stages = (
            (self.layer3[:2], self.exit20, thresholds[0]),
            (self.layer3[2:], self.exit28, thresholds[1]),
        )

        for blocks, head, threshold in stages:
            for block in blocks:
                x = block(x)

            logits = self._classify(x, head)
            log_probs = F.log_softmax(logits, dim=1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=1)
            done = entropy <= threshold

            result[remaining[done]] = logits[done]
            remaining = remaining[~done]
            x = x[~done]

            if remaining.numel() == 0:
                return result

        x = self.layer4(x)
        result[remaining] = self._classify(x, self.fc)
        return result


class EENet34Branch1525(ResNet):
    # 15・25は分岐位置、34は最終出力の名称
    exit_ids = (15, 25, 34)

    def __init__(
        self,
        num_classes=10,
        channels=(64, 128, 256, 512),
    ):
        super().__init__(
            BasicBlock,
            [3, 4, 6, 3],
            num_classes=num_classes,
            channels=channels,
        )

        self.exit15 = nn.Linear(channels[1], num_classes)
        self.exit25 = nn.Linear(channels[2], num_classes)

    def _classify(self, x, head):
        return head(torch.flatten(self.avgpool(x), 1))

    def _to_exit15(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        return self.layer2(x)

    def _to_exit25(self, x):
        return self.layer3[:5](x)

    def _to_final(self, x):
        x = self.layer3[5:](x)
        return self.layer4(x)

    def forward(self, x):
        x = self._to_exit15(x)
        out15 = self._classify(x, self.exit15)

        x = self._to_exit25(x)
        out25 = self._classify(x, self.exit25)

        x = self._to_final(x)
        out34 = self._classify(x, self.fc)

        return out15, out25, out34

    def forward_exit(self, x, exit_id):
        if exit_id not in self.exit_ids:
            raise ValueError(f"未対応のExit: {exit_id}")

        x = self._to_exit15(x)

        if exit_id == 15:
            return self._classify(x, self.exit15)

        x = self._to_exit25(x)

        if exit_id == 25:
            return self._classify(x, self.exit25)

        x = self._to_final(x)
        return self._classify(x, self.fc)

    @torch.no_grad()
    def forward_early(self, x, thresholds):
        if len(thresholds) != 2:
            raise ValueError("閾値は2個指定してください")

        result = x.new_empty((x.size(0), self.fc.out_features))
        remaining = torch.arange(x.size(0), device=x.device)

        x = self._to_exit15(x)

        for index, head in enumerate((self.exit15, self.exit25)):
            if index == 1:
                x = self._to_exit25(x)

            logits = self._classify(x, head)
            log_probs = F.log_softmax(logits, dim=1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=1)
            done = entropy <= thresholds[index]

            result[remaining[done]] = logits[done]
            remaining = remaining[~done]
            x = x[~done]

            if remaining.numel() == 0:
                return result

        x = self._to_final(x)
        result[remaining] = self._classify(x, self.fc)
        return result

def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def eenet20(num_classes=10):
    return EENet20(num_classes)

def eenet34(num_classes=10):
    return EENet34(num_classes)

def eenet34_1525(num_classes=10):
    return EENet34Branch1525(num_classes)

def eenet34_1525_half(num_classes=10):
    return EENet34Branch1525(
        num_classes=num_classes,
        channels=(32, 64, 128, 256),
    )

def eenet34_1525_quarter(num_classes=10):
    return EENet34Branch1525(
        num_classes=num_classes,
        channels=(16, 32, 64, 128),
    )

def eenet20_half(num_classes=10):
    return EENet20(
        num_classes=num_classes,
        channels=(8, 16, 32),
    )

MODEL_SPECS = {
    "eenet20": {
        "factory": eenet20,
        "exit_ids": (10, 16, 20),
        "shared_prefixes": (
            "conv1.",
            "bn1.",
            "layer1.",
            "layer2.0.",
        ),
        "profile_file": "inference_profile_resnet20.json",
    },
    "eenet20_half": {
        "factory": eenet20_half,
        "exit_ids": (10, 16, 20),
        "shared_prefixes": (
            "conv1.",
            "bn1.",
            "layer1.",
            "layer2.0.",
        ),
        "profile_file": "inference_profile_resnet20_half.json",
    },
    "eenet34": {
        "factory": eenet34,
        "exit_ids": (20, 28, 34),
        "shared_prefixes": (
            "conv1.",
            "bn1.",
            "layer1.",
            "layer2.",
            "layer3.0.",
            "layer3.1.",
        ),
        "profile_file": "inference_profile_resnet34_2028.json",
    },
    "eenet34_1525": {
        "factory": eenet34_1525,
        "exit_ids": (15, 25, 34),
        "shared_prefixes": (
            "conv1.",
            "bn1.",
            "layer1.",
            "layer2.",
        ),
        "profile_file": "inference_profile_resnet34_1525.json",
    },
    "eenet34_1525_half": {
        "factory": eenet34_1525_half,
        "exit_ids": (15, 25, 34),
        "shared_prefixes": (
            "conv1.",
            "bn1.",
            "layer1.",
            "layer2.",
        ),
        "profile_file": "inference_profile_resnet34_1525_half.json",
    },
    "eenet34_1525_quarter": {
        "factory": eenet34_1525_quarter,
        "exit_ids": (15, 25, 34),
        "shared_prefixes": (
            "conv1.",
            "bn1.",
            "layer1.",
            "layer2.",
        ),
        "profile_file": "inference_profile_resnet34_1525_quarter.json",
    },
}