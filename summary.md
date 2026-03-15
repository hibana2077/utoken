# test_pico_vit Pipeline 與設計說明

## 1. 目標與整體概覽
`test_pico_vit` 是一個在 CIFAR-10 上訓練的小型 Vision Transformer（ViT）實驗框架，提供兩種變體：
- `special`：在中間兩層插入 `SpecialViTBlock`，加入 sigma 門控與 uDTW 輔助損失。
- `baseline`：全層使用標準 ViT block，不啟用特殊機制。

主流程（training pipeline）如下：
1. 讀取 `TrainConfig`（`config.py` / CLI）。
2. 建立 CIFAR-10 train/val dataloader（`data.py`）。
3. 依 `variant` 建模（`factory.py` -> `model.py`）。
4. 訓練迴圈計算 `CE + aux_weight * aux_loss`（`train.py`）。
5. 每個 epoch 驗證並輸出指標，最後彙整 best val acc。

---

## 2. 設定層（TrainConfig）
`TrainConfig` 定義模型、特殊機制、資料與訓練超參數：
- 模型核心：`image_size=32`, `patch_size=4`, `embed_dim=192`, `depth=8`, `num_heads=3`, `mlp_ratio=4.0`。
- 特殊機制：`aux_weight=0.1`, `merge_mode={mul|add}`, `sigma_hidden_dim=64`, `sigma_a=1.5`, `sigma_b=0.5`, `udtw_gamma=0.1`, `udtw_beta=0.5`。
- 訓練：`epochs=20`, `lr=3e-4`, `weight_decay=0.05`, `batch_size=128`, `workers=4`, `seed=7`。
- 模式：`train_mode={special|baseline|both}`（預設 `both`）。

CLI 可覆蓋部分參數，最後組回 `TrainConfig`。

---

## 3. 資料 Pipeline（data.py）
資料集：`torchvision.datasets.CIFAR10`。

### 3.1 Train transform
- `RandomCrop(32, padding=4)`
- `RandomHorizontalFlip()`
- `ToTensor()`
- `Normalize(mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616))`

### 3.2 Val transform
- `ToTensor()`
- 同樣的 `Normalize(...)`

### 3.3 DataLoader
- train: `shuffle=True`
- val: `shuffle=False`
- 共同：`pin_memory=True`, `persistent_workers=(workers>0)`

---

## 4. 模型組裝流程（factory.py + model.py）
`create_model(cfg, device, variant)` 的行為：
- 僅接受 `model_name=test_pico_vit`。
- `variant=special` -> `use_special_blocks=True`。
- `variant=baseline` -> `use_special_blocks=False`。
- `use_cuda_dtw` 依 `device.type == 'cuda'` 決定是否嘗試 CUDA uDTW。

### 4.1 Token 化與輸入嵌入
`PatchEmbed`（Conv2d, kernel=stride=patch_size）：
- 輸入：`[B, 3, 32, 32]`
- patch_size=4 -> grid=8x8 -> `num_patches=64`
- 輸出 patch tokens：`[B, 64, 192]`

再加入：
- `cls_token`：`[1,1,192]` 展成 `[B,1,192]`
- concat 後序列長度 65：`[B,65,192]`
- 加上 `pos_embed`：`[1,65,192]`
- `pos_drop`

### 4.2 `special_indices` 邏輯（你目前選到的重點）
在 `use_special_blocks=True` 且 `depth>=4` 時：
- `mid = depth // 2`
- `special_indices = {max(1, mid-1), min(depth-2, mid)}`

以預設 `depth=8`：
- `mid=4`
- `special_indices={3,4}`

也就是「中間兩層」改成 `SpecialViTBlock`，其餘層為 `StandardViTBlock`。

> 設計意圖：把較重的特殊運算集中在中段，避免每層都加重計算與訓練不穩定。

### 4.3 Head
`forward_features` 回傳 CLS 特徵後：
- `LayerNorm`
- `Dropout`
- `Linear(embed_dim -> num_classes)`

最終 `forward` 回傳：
- `logits`
- `stats`（至少含 `aux_loss`，special 時還有 sigma 統計）

---

## 5. Block 細節（blocks.py / modules.py）

## 5.1 StandardViTBlock
標準 pre-norm 殘差結構：
1. `x = x + Attention(LN(x))`
2. `x = x + MLP(LN(x))`

回傳 `aux_loss=0`。

## 5.2 SpecialViTBlock 的核心流程
每個 special block 內有兩段（Attention 段 + MLP 段），且都會產生 uDTW 損失。

### Attention 段
1. `norm1_x = LN(x)`
2. `attn_out = Attention(norm1_x)`
3. 取 patch 區域（排除 cls）：
   - `original_patch = original_x[:,1:]`
   - `attn_patch = attn_out[:,1:]`
4. 用兩個 `SigmaNet` 產生 sigma：
   - `sigma_x = SigmaNet_norm1(original_patch)`
   - `sigma_attn = SigmaNet_attn(attn_patch)`
5. `uDTW(original_patch, attn_patch, sigma_x, sigma_attn, beta=udtw_beta)`
   - 得到 `(dtw_attn_d, dtw_attn_s)`
6. merge：
   - cls token：`cls_out = seq_a[:,:1] + seq_b[:,:1]`
   - patch token：
     - `mul`：`seq_a + seq_b * sigma_b`
     - `add`：`seq_a + seq_b + sigma_b`

### MLP 段
1. `norm2_x = LN(x)`
2. `mlp_out = MLP(norm2_x)`
3. 取 patch：`mlp_patch = mlp_out[:,1:]`
4. sigma：
   - `sigma_x = SigmaNet_norm2(original_patch)`
   - `sigma_mlp = SigmaNet_mlp(mlp_patch)`
5. uDTW：`(dtw_mlp_d, dtw_mlp_s)`
6. merge 同上（保持 cls 特殊處理）

### Aux loss 與統計
- `patch_len = max(1, x.size(1)-1)`
- `aux_loss = (mean(dtw_attn_d)+mean(dtw_attn_s)+mean(dtw_mlp_d)+mean(dtw_mlp_s)) / (patch_len^2)`

另外輸出 sigma 監控指標：
- `sigma_attn_mean/max/min/std`
- `sigma_mlp_mean/max/min/std`

## 5.3 uDTW 後端
- 優先使用 `src.udtw.uDTW`（若可 import）。
- 失敗時退回 `FallbackUDTW`：
  - `dist = mean(cdist(x,y)^2)`
  - `sig = mean(cdist(sigma_x,sigma_y, p=1)) * beta`

這讓專案在沒有 numba/CUDA uDTW 依賴時仍可運作。

## 5.4 基礎模組
- `TinyAttention`：手寫 MHA（qkv 線性投影、scaled dot-product、softmax、proj）。
- `TinyMlp`：`Linear -> GELU -> Dropout -> Linear -> Dropout`。
- `SigmaNet`：`Linear -> ReLU -> Linear`，最後在 token 維度做 channel mean，經 `sigmoid_ab(a,b)` 轉成 sigma：
  - `sigma = a * sigmoid(value) + b`

---

## 6. 前向與統計聚合（model.py）
`forward_features` 會遍歷所有 blocks：
- 累加每層 `stats['aux_loss']`。
- 若該層有 sigma stats（special block 才有），先加總再除以 `sigma_count`，得到「special 層平均統計」。

輸出 `stats_out`：
- 一定有 `aux_loss`（baseline 時為 0；special 時為兩個 special block 的總和）。
- special 模式會額外有 8 個 sigma 指標（平均後）。

---

## 7. 訓練與驗證 Pipeline（train.py）

## 7.1 train()
1. 決定 `device`（CUDA 優先）。
2. 建立 dataloader。
3. 依 `train_mode` 決定要跑：
   - `special`
   - `baseline`
   - 或兩者都跑
4. 逐 variant 呼叫 `train_one`，最後印 summary。

## 7.2 train_one()
- 每個 variant 都會 `set_seed(cfg.seed)`，確保可重現。
- 建立模型與 `AdamW`。
- 每 epoch：
  - 訓練階段：
    - 前向得 `logits, stats`
    - `ce_loss = cross_entropy(logits, labels)`
    - `loss = ce_loss + aux_weight * stats['aux_loss']`
    - backward + step
  - 驗證階段：呼叫 `evaluate()` 同公式計算 loss/acc
  - 紀錄並列印：train/val loss、acc、aux
  - special 模式加印 sigma 統計

回傳 `best_val_acc`。

## 7.3 evaluate()
- `model.eval()` + `torch.no_grad()`
- 與 train 同 loss 公式（含 aux 項）
- 回傳 `{loss, acc}`

---

## 8. 特性與實務解讀
1. `special` vs `baseline` 可直接 A/B 比較：
   - 在同一資料、同一 optimizer/超參數下，評估特殊 block 的收益。
2. `aux_weight` 是關鍵平衡旋鈕：
   - 太大可能壓制 CE；太小可能特殊機制學不到。
3. 目前 special block 使用 `original_patch`（block 輸入）作為 attention/MLP 兩段的參考對象：
   - 這代表 MLP 的 uDTW 不是對「attention 後 patch」比，而是對原始 patch 比。
4. `depth<4` 在 special 模式會直接報錯：
   - 因為設計上要保留中間兩層 special，同時避免貼邊層。

---

## 9. 預設尺寸與張量路徑（以 CIFAR-10 為例）
- 輸入影像：`[B, 3, 32, 32]`
- patch embed：`[B, 64, 192]`
- 加 cls + pos：`[B, 65, 192]`
- 經 `depth=8` blocks 後仍為：`[B, 65, 192]`
- 取 `x[:,0]`（cls）-> `LayerNorm`：`[B, 192]`
- head：`[B, 10]` logits

---

## 10. 檔案責任分工
- `config.py`：訓練/模型設定資料結構
- `data.py`：CIFAR-10 資料與 augmentation
- `factory.py`：依 variant 建立模型
- `modules.py`：Attention/MLP/SigmaNet/PatchEmbed 基本模組
- `blocks.py`：標準與特殊 block（含 uDTW）
- `model.py`：完整 ViT 組裝、special block 位置配置、統計聚合
- `train.py`：訓練、驗證、CLI 入口
