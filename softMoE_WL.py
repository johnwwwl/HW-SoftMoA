from gc import freeze
from modulefinder import Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
import joblib
torch.backends.cudnn.benchmark = True
from typing import Tuple, Optional,Union
from torch import Tensor
import pickle
import pywt


catchment = 'Acomb_gardenhouse'
expert_num = 36
history_steps = 32
future_steps = history_steps

batch_size = 128
lr = 3e-4
val_split = 0.3
model_dir = f'times_explainability/softmoe/{catchment}_{expert_num}_cluster_initial_v3'

os.makedirs(model_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import pandas as pd
minmax={'Acomb_gardenhouse':[-0.079,0.686],
        'Acomb_Millersfield':[0.095,1.28],
        'Riding_mill':[-0.032 ,4.559 ],
        'Haltwhistle':[0.192,2.951],
        'Stocksfield':[0.211,1.748],
        'Hepscot':[0.087,6.539]}


train_df = pd.read_csv(f"times_explainability/data1/{catchment}/train.csv")

train_df = train_df.fillna(method='ffill').fillna(method='bfill')

train_data = train_df[['level','rain']].values
min_l, max_l = minmax[catchment]
train_data[:,0] = train_data[:,0] * (max_l - min_l) + min_l
water = train_data[:, 0]
rain = train_data[:, 1]
water_mean = water.mean()
water_std = water.std() + 1e-6
rain_mean = rain.mean()
rain_std = rain.std() + 1e-6
import torch
import torch.nn as nn
import torch.nn.functional as F

def convert_water_to_cwt(water, wavelet='morl'):
    future_steps = water.shape[0]//4
    scales = np.arange(1, future_steps + 1)
    signal = np.copy(water)

    # 计算CWT
    coefficients, _ = pywt.cwt(signal, scales, wavelet)

    # 防止爆炸：处理 inf 和 nan
    coefficients = np.nan_to_num(coefficients, nan=0.0, posinf=1e6, neginf=-1e6)

    # 防止系数太大：归一化处理
    # max_abs = np.max(np.abs(coefficients)) + 1e-8  # 防止除零
    # coefficients = coefficients / max_abs
    # 增加通道维度
    cwt_future = coefficients[..., np.newaxis]
    return cwt_future
def convert_rain_to_cwt(rain, wavelet='mexh'):
    future_steps = rain.shape[0]//4
    scales = np.arange(1, future_steps + 1)
    signal = np.copy(rain)

    # 计算CWT
    coefficients, _ = pywt.cwt(signal, scales, wavelet)

    # 防止爆炸：处理 inf 和 nan
    coefficients = np.nan_to_num(coefficients, nan=0.0, posinf=1e6, neginf=-1e6)

    # 防止系数太大：归一化处理
    # max_abs = np.max(np.abs(coefficients)) + 1e-8
    # coefficients = coefficients / max_abs

    # 增加通道维度
    cwt_future = coefficients[..., np.newaxis]
    return cwt_future
class TimeSeriesCWTDataset(Dataset):
    def __init__(self, data, history_steps=32, future_steps=32, threshold=-1000,cluster_model=None,mode='normal',water_mean=None,water_std=None,rain_mean=None,rain_std=None):
        self.data = data
        self.cluster_model = cluster_model
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.threshold = threshold
        # 样本数量 = T - (history_steps + future_steps) + 1
        self.num_samples = data.shape[0] - (history_steps + future_steps) + 1
        self.filtered_indices = self.filter_indices()  # 先筛选出符合条件的索引
        self.mode=mode
        self.water_mean = water_mean
        self.water_std = water_std
        self.rain_mean = rain_mean
        self.rain_std = rain_std

    def __len__(self):
        return len(self.filtered_indices)

    def filter_indices(self):
        """筛选出符合条件的索引"""
        indices = []
        for idx in range(self.num_samples):
            target = self.data[idx + self.history_steps: idx + self.history_steps + self.future_steps, 0]  # shape (32,)
            if np.any(target > self.threshold):  # 如果 target 中有值超过 threshold
                indices.append(idx)  # 记录索引
        return np.array(indices)  # 返回筛选后的索引数组

    def __getitem__(self, idx):
        idx = self.filtered_indices[idx]
        # 1. 提取样本：
        # 历史数据：从 idx 到 idx+history_steps，形状 (history_steps, 2)
        sample_history = self.data[idx: idx + self.history_steps, :]
        sample_water = sample_history[:, 0].copy()
        sample_rain = sample_history[:, 1].copy()
        # 未来数据：取未来 future_steps 个时间步，
        # 未来降雨：取第二列，形状 (future_steps,)
        future_rain = self.data[idx + self.history_steps: idx + self.history_steps + self.future_steps, 1].copy()
        # 目标：未来水位，即第一列，形状 (future_steps,)
        target = self.data[idx + self.history_steps: idx + self.history_steps + self.future_steps, 0].copy()
        raw_input= np.stack([sample_water, sample_rain, future_rain], axis=1)
        # sample_water = (sample_water - self.water_mean) / self.water_std
        # sample_rain  = (sample_rain  - self.rain_mean)  / self.rain_std
        # future_rain     = (future_rain     - self.rain_mean)  / self.rain_std
        # target = (target - self.water_mean) / self.water_std

        cluster_label = self.cluster_model.predict([raw_input])[0]
        # 2. 使用 CWT 将一维时间序列转换成二维表示：
        # 对历史数据进行 CWT，得到形状 (history_steps, history_steps, 2)
        cwt_history_water = convert_water_to_cwt(sample_water, 'morl')
        cwt_history_rain = convert_rain_to_cwt(sample_rain)
        # 对未来降雨数据进行 CWT，得到形状 (future_steps, future_steps, 1)
        cwt_future = convert_rain_to_cwt(future_rain)

        # 3. 拼接各通道：
        # 假设 history_steps == future_steps == 32，此时拼接后 input_image 形状为 (32, 32, 3)
        input_image = np.concatenate([cwt_history_water, cwt_history_rain, cwt_future], axis=-1)

        # 4. 类型转换及维度调整：
        input_image = input_image.astype(np.float32)
        target = target.astype(np.float32)  # shape: (32,1)
        # PyTorch 要求输入 shape: (channels, height, width)
        # 当前 input_image shape: (32, 32, 3) -> 转置到 (3, 32, 32)
        input_image = np.transpose(input_image, (2, 0, 1))

        # 5. 转换为 tensor 返回
        input_tensor = torch.tensor(input_image)
        target_tensor = torch.tensor(target)
        cluster_tensor = torch.tensor(cluster_label, dtype=torch.long)
        raw_input=torch.tensor(raw_input)
        return input_tensor, target_tensor, cluster_tensor
class CNNExpert(nn.Module):
    def __init__(self, out_steps=32):
        super().__init__()

        self.conv = nn.Sequential(
            # 第一层：Grouped 卷积，每个输入通道独立处理（适配 Grad-CAM）
            nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3),  # (B,3,8,32) → (B,3,8,32)
            nn.ReLU(),
            # nn.Dropout2d(0.5),

            # 第二层：融合通道特征
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # → (B,32,8,32)
            nn.ReLU(),
            # nn.Dropout2d(0.5),

            # 第三层：进一步提取特征
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (B,64,8,32)
            nn.ReLU(),
            # nn.Dropout2d(0.5),

            # 池化：只降采样时间维度（W），保持尺度信息
            nn.MaxPool2d(kernel_size=(1, 2)),  # → (B,64,8,16)

            # 第四层：增加通道维度
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # → (B,128,8,16)
            nn.ReLU(),
            # nn.Dropout2d(0.5),
        )

        # 最终全连接层（使用全局平均池化）
        self.head = nn.Linear(128, out_steps)

    def forward(self, x):
        x = self.conv(x)  # → (B,128,8,16)
        x = x.mean(dim=(2, 3))  # GAP → (B,128)
        return self.head(x)  # → (B,32)
class GatingCNN(nn.Module):

    def __init__(self, num_experts: int):
        super().__init__()
        # 一层 1×1 卷积：3 -> num_experts
        self.conv1x1 = nn.Conv2d(in_channels=3,
                                 out_channels=num_experts,
                                 kernel_size=1,
                                 bias=True)
        # self.dropout = nn.Dropout2d(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)  这里 H=8, W=32
        returns: (B, num_experts) 还没做 softmax
        """
        # 1×1 卷积 -> (B, num_experts, H, W)
        g = self.conv1x1(x)
        # 全局平均池化到 (B, num_experts, 1, 1)
        g = g.mean(dim=[2, 3], keepdim=False)
        # g = self.dropout(g)
        # 输出 (B, num_experts)
        return g
class SoftMoE_CNN(nn.Module):
    def __init__(self, num_experts, out_steps, input_shape=(3,8,32)):
        super().__init__()
        self.num_experts = num_experts
        self.experts     = nn.ModuleList([CNNExpert(out_steps) for _ in range(num_experts)])
        self.gating_net  = GatingCNN(num_experts)

    def forward(self, x, return_gating=False,temp: float = 1.0):
        gate_logits = self.gating_net(x)             # (B, E)
        gate_w      = torch.softmax(gate_logits, dim=1).unsqueeze(-1)  # (B, E, 1)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1) # (B, E, O)
        out = (gate_w * expert_outs).sum(1)             # (B, O)

        return (out, gate_logits) if return_gating else out
    # def forward(self, x, k=17):
    #     return self.experts[k](x)

cluster_model = joblib.load(
    f'times_explainability/softmoe/{catchment}_{history_steps}_{future_steps}_clustering_{expert_num}.pkl'
)

full_dataset = TimeSeriesCWTDataset(train_data,
                                    history_steps, future_steps,
                                    cluster_model=cluster_model,
                                    mode='normal',water_mean=water_mean,water_std=water_std,rain_mean=rain_mean,rain_std=rain_std)

total_len = len(full_dataset)
val_len = int(total_len * val_split)
train_len = total_len - val_len
train_idxs = list(range(train_len))
val_idxs   = list(range(train_len, total_len))
from torch.utils.data import Subset
train_ds = Subset(full_dataset, train_idxs)
val_ds   = Subset(full_dataset, val_idxs)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)



import torch
from collections import Counter

c, h, w = train_loader.dataset[0][0].shape

model=SoftMoE_CNN(num_experts=expert_num,out_steps=future_steps,
                    input_shape=(c, h, w) )


model.to(device)
gate_params     = list(model.gating_net.parameters())
non_gate_params = [p for n,p in model.named_parameters()
                   if not n.startswith('gating_net.')]
lr_exp  = 1e-3
lr_gate = 5e-4

optimizer_exp  = optim.Adam(non_gate_params, lr=lr_exp,  weight_decay=1e-4)
optimizer_gate = optim.Adam(gate_params,     lr=lr_gate, weight_decay=0.0)

sched_exp  = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_exp, mode='min', factor=0.5,
                patience=10, verbose=True, min_lr=1e-6)

mse_loss = nn.MSELoss()
best_model_path = os.path.join(model_dir, 'best.pth')
start_epoch=1
stage1_epochs  = 1200
total_epochs   = 1500
beta = 0.01
tau_start, tau_end = 2.0, 0.5
tau_decay = (tau_end / tau_start) ** (1 / max(1, total_epochs-stage1_epochs))

mse_loss = torch.nn.MSELoss()
E        = model.num_experts
tau = tau_start
best_val = float('inf')

with open(os.path.join(model_dir,'weights.pkl'), "rb") as f:
    expert_weight = pickle.load(f)
counts   = torch.tensor(expert_weight, dtype=torch.float32)
weights  = counts.sum() / counts
weights /= weights.mean()
weights  = weights.to(device)
print('weights ',weights)


if os.path.isfile(best_model_path):
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer_exp.load_state_dict(ckpt['optimizer_exp_state_dict'])
    optimizer_gate.load_state_dict(ckpt['optimizer_gate_state_dict'])
    sched_exp.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_val = ckpt['val_mse']
    tau=ckpt['tau']
    # 获取当前 lr
    current_lr = optimizer_exp.param_groups[0]['lr']
    print(f"Loaded checkpoint from '{best_model_path}' (epoch={ckpt['epoch']}), lr={current_lr:.6e}, best_loss={best_val:.6f}")
else:
    print("No existing best.pth found, starting training from scratch.")


# ====== 辅助函数 ======
@torch.no_grad()
def validate():
    model.eval()
    mse_sum = 0.0
    running_ent = 0.0
    running_usage = torch.zeros(expert_num, device=device)
    count_batch = 0
    for x, y, _ in val_loader:
        x, y = x.to(device), y.to(device)
        preds, gate_logits = model(x, return_gating=True,temp=tau)
        p_gate = F.softmax(gate_logits, dim=1)
        running_ent += (-p_gate * p_gate.log()).sum(dim=1).mean().item()
        running_usage += p_gate.mean(0)
        mse_sum += mse_loss(preds, y).item() * x.size(0)
        count_batch+=1
    avg_loss=mse_sum / len(val_loader.dataset)
    usage = (running_usage / count_batch).cpu().numpy()
    ent = running_ent / count_batch
    return avg_loss
def train_one_epoch(epoch, tau):
    with torch.no_grad():
        w_before = model.experts[0].conv[0].weight.clone()  # 任选一层

    freeze_gating = epoch >= stage1_epochs
    print(f'freeze gating {freeze_gating}')
    for p in model.gating_net.parameters():
        p.requires_grad_(not freeze_gating)

    model.train()
    mse_running, ent_running = 0.0, 0.0
    usage_running = torch.zeros(E, device=device)

    for batch_idx,(x, y, cluster) in enumerate(train_loader):
        x, y, cluster = x.to(device), y.to(device), cluster.to(device)

        optimizer_exp.zero_grad()
        if not freeze_gating:
            optimizer_gate.zero_grad()
        preds, gate_logits = model(x, return_gating=True,temp=tau)
        mse = mse_loss(preds, y)
        loss = mse

        if not freeze_gating:                           # Stage-2
            log_p_gate = F.log_softmax(gate_logits, dim=1)
            p_gate = log_p_gate.exp()
            p_cluster = F.one_hot(cluster, E).float()
            sample_w=weights[cluster].view(-1, 1)

            kl = F.kl_div(log_p_gate, p_cluster, reduction='none').sum(1)
            kl = (sample_w * kl).mean()
            ent = (-p_gate * log_p_gate).sum(dim=1).mean()

            target_ratio = 1.0
            eps = 1e-8
            alpha_raw = target_ratio * mse.detach() / (kl.detach() + eps)
            momentum = 0.9

            if not hasattr(train_one_epoch, 'alpha_ema'):
                train_one_epoch.alpha_ema = alpha_raw
            else:
                train_one_epoch.alpha_ema = (
                        momentum * train_one_epoch.alpha_ema + (1 - momentum) * alpha_raw
                )
            kl_alpha = train_one_epoch.alpha_ema
            loss = mse + kl_alpha * kl

            ent_running   += (-p_gate * p_gate.log()).sum(1).mean().item()
            usage_running += p_gate.mean(0)
        loss.backward()


        optimizer_exp.step()
        if not freeze_gating:
            optimizer_gate.step()
        mse_running += mse.item()

    n_batch = len(train_loader)
    avg_mse = mse_running / n_batch

    if freeze_gating:
        return avg_mse, None, None
    else:
        ent   = ent_running / n_batch
        usage = (usage_running / n_batch).detach().cpu().numpy()
        return avg_mse, ent, usage


for epoch in tqdm(range(start_epoch,total_epochs),desc=f'{catchment}-{expert_num}'):
    avg_mse, ent, usage = train_one_epoch(epoch, tau)
    val_mse = validate()
    sched_exp.step(val_mse)
    freeze_gating = epoch >= stage1_epochs
    if freeze_gating:
        print(f"[Epoch {epoch:03d}] MSE={avg_mse:.9f} Val Mse={val_mse:.9f}(gating frozen)")
    else:
        ent_str = f"{ent:.3f}" if ent is not None else "N/A"
        usage_str = np.array2string(usage, precision=3) if usage is not None else "N/A"

        print(f"[Epoch {epoch:03d}]  "
              f"MSE={avg_mse:.9f}  Val={val_mse:.9f}  "
              f"Entropy={ent_str}  Usage={usage_str}  "
              f"({'gating frozen' if freeze_gating else 'gating trainable'})")


    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_exp_state_dict': optimizer_exp.state_dict(),
        'optimizer_gate_state_dict': optimizer_gate.state_dict(),
        'scheduler_state_dict': sched_exp.state_dict(),
        'val_mse': val_mse,
        'tau': tau,
    }

    torch.save(ckpt, os.path.join(model_dir, f'{epoch:03d}.pth'))
    if epoch > stage1_epochs:
        tau = max(tau_end, tau * tau_decay)
    if val_mse < best_val:
        print(f'New best recorded val mse={val_mse:6f}')
        print(model_dir)
        best_val = val_mse
        torch.save(ckpt, os.path.join(model_dir, f'best.pth'))
        model_cpu = model.cpu().eval()
        example_inputs, _,_ = next(iter(train_loader))
        example_inputs = example_inputs.to('cpu')
        scripted = torch.jit.trace(model_cpu, example_inputs)
        scripted.save(os.path.join(model_dir, 'best.pt'))
        model.to(device)
        model.train()

print("✔ training finished")