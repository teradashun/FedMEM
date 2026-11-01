import torch
import torch.nn as nn
from .models import eenet34

model = eenet34(num_classes=10)

GLOBAL_PREFIXES = (
    "conv1",
    "bn1",
    "layer1",
    "layer2",
    "layer3.0",
    "layer3.1",
)

def count_params_by_prefix(model, prefixes):
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if any(name.startswith(prefix) for prefix in prefixes)
    )

n_global_params = count_params_by_prefix(model, GLOBAL_PREFIXES)

# float32の場合
S_bits = n_global_params * 32
S_bytes = n_global_params * 4

print(n_global_params)
print(S_bits)
print(S_bytes)

model = eenet34(num_classes=10).eval()

def forward_exit20(model, x):
    x = model.relu(model.bn1(model.conv1(x)))
    x = model.layer1(x)
    x = model.layer2(x)
    for block in model.layer3[:2]:
        x = block(x)
    return model._classify(x, model.exit20)

def forward_exit28(model, x):
    x = model.relu(model.bn1(model.conv1(x)))
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    return model._classify(x, model.exit28)

def forward_exit34(model, x):
    x = model.relu(model.bn1(model.conv1(x)))
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    return model._classify(x, model.fc)

def count_macs(model, forward_fn, input_shape=(1, 3, 32, 32)):
    total = 0
    hooks = []

    def hook(module, inputs, output):
        nonlocal total
        x = inputs[0]

        if isinstance(module, nn.Conv2d):
            batch = x.shape[0]
            out_h, out_w = output.shape[2], output.shape[3]
            kh, kw = module.kernel_size
            in_ch = module.in_channels // module.groups
            total += batch * module.out_channels * out_h * out_w * in_ch * kh * kw

        elif isinstance(module, nn.Linear):
            batch = x.shape[0]
            total += batch * module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))

    with torch.no_grad():
        x = torch.randn(*input_shape)
        forward_fn(model, x)

    for h in hooks:
        h.remove()

    return total

mac20 = count_macs(model, forward_exit20)
mac28 = count_macs(model, forward_exit28)
mac34 = count_macs(model, forward_exit34)

print("Exit20 MACs:", mac20)
print("Exit28 MACs:", mac28)
print("Exit34 MACs:", mac34)