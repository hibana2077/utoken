# `test_pico_vit` Pipeline 與目前設計說明

## 1. 目標與整體概覽
`test_pico_vit` 是一個在 CIFAR-10 上訓練小型 Vision Transformer 的實驗框架。整體模型骨幹是標準 ViT block 疊加；`special` 與 `baseline` 的差別，不再是替換某些 block，而是是否啟用一條額外的「跨層 sequence alignment / sigma gating」路徑。

- `baseline`：只跑標準 ViT 前向，`aux_loss=0`
- `special`：在兩個指定 block 的輸出之間啟用 `BlockSequenceAdapter`，額外計算 sigma 與 uDTW-based 輔助損失

主流程如下：
1. 讀取 `TrainConfig`（`config.py` / CLI）
2. 建立 CIFAR-10 train/val dataloader（`data.py`）
3. 依 `variant` 建模（`factory.py` -> `model.py`）
4. 訓練時最終 loss 為 `CE + aux_weight * aux_loss`
5. 每個 epoch 驗證並輸出 loss / acc / calibration 指標，最後彙整最佳 val acc

---

## 2. 設定層（`TrainConfig`）
`TrainConfig` 目前包含以下主要設定：

- 模型：`image_size=32`, `patch_size=4`, `in_chans=3`, `embed_dim=192`, `depth=8`, `num_heads=3`, `mlp_ratio=4.0`, `drop_rate=0.0`
- 特殊路徑：`aux_weight=0.1`, `merge_mode="mul"`, `sequence_pair="el"`, `sigma_hidden_dim=64`, `sigma_a=1.5`, `sigma_b=0.5`, `udtw_gamma=0.1`, `udtw_beta=0.5`
- 訓練：`epochs=20`, `lr=3e-4`, `weight_decay=0.05`, `batch_size=128`, `workers=4`, `seed=7`
- 模式：`train_mode={"special","baseline","both"}`

### CLI 補充
`train.py` 提供對應 CLI 參數，但目前實際組回 `TrainConfig` 時，只有部分欄位會被覆蓋。

目前有兩個值得注意的實作事實：
- `--sigma-a` / `--sigma-b` 雖然有 parse，但沒有傳回 `TrainConfig(...)`，所以實際仍會用 dataclass 預設值 `1.5 / 0.5`
- `--merge-mode` 雖然允許 `mul` 與 `add`，但目前 `BlockSequenceAdapter` 只實作 `mul`，若設成 `add` 會在執行時丟 `ValueError`

---

## 3. 資料 Pipeline（`data.py`）
資料集使用 `torchvision.datasets.CIFAR10`。

### Train transform
- `RandomCrop(32, padding=4)`
- `RandomHorizontalFlip()`
- `ToTensor()`
- `Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))`

### Val transform
- `ToTensor()`
- 相同的 `Normalize(...)`

### DataLoader
- train: `shuffle=True`
- val: `shuffle=False`
- 共同：`pin_memory=True`, `persistent_workers=(workers > 0)`

---

## 4. 模型組裝流程（`factory.py` + `model.py`）
`create_model(cfg, device, variant)` 的行為：

- 只接受 `model_name="test_pico_vit"`
- `variant="special"` -> `enable_sigma_path=True`
- `variant="baseline"` -> `enable_sigma_path=False`
- `use_cuda_dtw` 依 `device.type == "cuda"` 決定

### 4.1 Token 化與輸入嵌入
`PatchEmbed` 使用 `Conv2d(kernel_size=patch_size, stride=patch_size)`：

- 輸入：`[B, 3, 32, 32]`
- `patch_size=4` -> grid `8 x 8` -> `num_patches=64`
- patch tokens 輸出：`[B, 64, 192]`

之後模型會：
- 加上 `cls_token`：`[B, 1, 192]`
- 與 patch token 串接成 `[B, 65, 192]`
- 加上 `pos_embed`
- 經過 `pos_drop`

### 4.2 Backbone 結構
目前 `depth` 個 block 全部都是 `StandardViTBlock`，沒有任何 `SpecialViTBlock` 類別或中間特化 block。

每層都是標準 pre-norm 殘差：
1. `x = x + Attention(LN(x))`
2. `x = x + MLP(LN(x))`

每個標準 block 都回傳 `{"aux_loss": 0}`。

### 4.3 Sequence Pair 選擇邏輯
特殊機制不是插進 block 內，而是在 backbone 跑到指定層時，取兩個 block 輸出做對齊。

`TestPicoViT.SEQUENCE_PAIR_MAP`：
- `em` -> `("early", "mid")`
- `ml` -> `("mid", "last")`
- `el` -> `("early", "last")`

對應 block index 由 `_resolve_sequence_block_indices(depth)` 決定：
- `early = 0`
- `mid = depth // 2`
- `last = depth - 1`

以預設 `depth=8` 為例：
- `early = 0`
- `mid = 4`
- `last = 7`

所以：
- `em` -> 對齊第 `0` 與第 `4` 層輸出
- `ml` -> 對齊第 `4` 與第 `7` 層輸出
- `el` -> 對齊第 `0` 與第 `7` 層輸出

模型要求 `depth >= 3`，因為 early/mid/last 必須可區分。

### 4.4 Head
分類 head 流程是：
- 取最終 `x[:, 0]` 的 cls token
- `LayerNorm`
- `Dropout`
- `Linear(embed_dim -> num_classes)`

最終 `forward()` 回傳：
- `logits`
- `stats`（一定包含 `aux_loss`；special 模式下可能額外有 sigma 統計）

---

## 5. 特殊路徑核心（`BlockSequenceAdapter`）
當 `enable_sigma_path=True` 時，模型會在 `seq_a_index` 與 `seq_b_index` 兩個時點做以下事情：

1. 在 `seq_a_index` block 跑完後，保存 `seq_a`
2. 用 `sigmanet_a(seq_a)` 估計 `sigma_a`
3. 在 `seq_b_index` block 跑完後，取得 `seq_b`
4. 用 `sigmanet_b(seq_b)` 估計 `sigma_b`
5. 對 `(seq_a, seq_b, sigma_a, sigma_b)` 計算 uDTW 損失
6. 將 `seq_b` 依 sigma 做 gated merge，直接覆蓋當前主幹的 `x`

換句話說，這條路徑是在「兩個完整 block 輸出序列之間」做對齊與重加權，不是針對 attention 分支與 MLP 分支分別處理。

### 5.1 SigmaNet
`SigmaNet` 結構：
- `Linear(dim -> hidden_dim)`
- `ReLU`
- `Linear(hidden_dim -> dim)`

輸出後會：
- reshape 回 `[B, L, D]`
- 在 channel 維度做 mean，變成 `[B, L, 1]`
- 套用 `sigmoid_ab(a, b, value)`，即 `a * sigmoid(value) + b`

因此 sigma 是每個 token 一個 scalar gate，而不是每個 channel 一個 gate。

### 5.2 Merge 邏輯
目前 `_merge_into_target()` 只支援：
- `mul`: `seq_b * sigma_b`

也就是 `seq_b` 會被 sigma 逐 token 縮放後，直接成為新的 `x`。

注意：
- 目前沒有保留 residual `seq_a + ...`
- 也沒有對 `cls token` 做特殊處理
- `merge_mode="add"` 雖然出現在 config / CLI 中，但尚未實作

### 5.3 uDTW 後端
`blocks.py` 啟動時會嘗試：
- `from src.udtw import uDTW as NativeUDTW`

若成功：
- 使用 `NativeUDTW(use_cuda=..., gamma=cfg.udtw_gamma, normalize=False)`

若失敗：
- 退回 `FallbackUDTW`

`FallbackUDTW` 的計算：
- `dist = mean(cdist(x, y, p=2)^2)`
- `sig = mean(cdist(sigma_x, sigma_y, p=1)) * beta`

回傳 `(dist, sig)`。

### 5.4 Aux loss
`apply_seq_b()` 內的輔助損失為：

`aux_loss = (mean(dtw_d) + mean(dtw_s)) / (seq_len^2)`

其中 `seq_len = seq_b.size(1)`，包含 cls token。

這個 `aux_loss` 只在 `seq_b` 被處理那一次加入整體 loss；其他所有標準 block 的 `aux_loss` 都是 0。

### 5.5 Sigma 統計
`BlockSequenceAdapter` 目前只輸出對 `sigma_b` 的統計：
- `sigma_mean`
- `sigma_max`
- `sigma_min`
- `sigma_std`

這些值在 `model.py` 中做累加/平均，但因為一次 forward 最多只觸發一次 `apply_seq_b()`，實際上就是單次值。

---

## 6. 前向與統計聚合（`model.py`）
`forward_features()` 流程：

1. patch embed + cls token + pos embed
2. 依序跑過所有 `StandardViTBlock`
3. 累加每層 `stats["aux_loss"]`（實際上標準 block 都是 0）
4. 若啟用 sigma path，於指定 `seq_a_index` 保存 `seq_a/sigma_a`
5. 於指定 `seq_b_index` 執行 `apply_seq_b()`，更新主幹 `x` 並加入 pair aux loss
6. 取最終 cls token，做 `LayerNorm`

輸出的 `stats_out`：
- 一定有 `aux_loss`
- 若 special 路徑啟用，且 sequence adapter 成功執行，則額外有 `sigma_mean/max/min/std`

---

## 7. 訓練與驗證 Pipeline（`train.py`）

## 7.1 train()
1. 選擇裝置（CUDA 優先）
2. 建立 dataloader
3. 根據 `train_mode` 決定跑 `special`、`baseline` 或兩者都跑
4. 逐 variant 呼叫 `train_one()`
5. 最後輸出各 variant 的 `best_val_acc`

啟動時會印：
- `device`
- `model`
- `dataset=cifar10`
- `train_mode`
- `sequence_pair`

## 7.2 train_one()
每個 variant 都會：
- `set_seed(cfg.seed)`
- 建立模型
- 使用 `AdamW`

每個 epoch：
- 訓練階段：
  - `logits, stats = model(images)`
  - `ce_loss = cross_entropy(logits, labels)`
  - `loss = ce_loss + cfg.aux_weight * stats["aux_loss"]`
  - backward + optimizer step
- 驗證階段：
  - 呼叫 `evaluate()` 計算同樣的 loss 公式與指標
- 輸出：
  - `train_loss`
  - `train_acc`
  - `val_loss`
  - `val_acc`
  - `val_brier`
  - `val_conf_gap`
  - `aux`
  - 若為 `special`，再加印 `sigma_mean/max/min/std`

`train.py` 目前印出的 sigma 統計，是該 epoch 最後一個 training batch 的值，不是整個 epoch 平均。

## 7.3 evaluate()
`evaluate()` 在 `model.eval()` 與 `torch.no_grad()` 下計算：

- `loss`
- `acc`
- `brier`
- `avg_conf`
- `conf_gap = avg_conf - acc`

其中：
- `brier` 使用 one-hot label 與 softmax probability 計算
- `conf_gap` 可視為簡單 calibration 誤差觀察值

---

## 8. 目前 special vs baseline 的實際差異
兩者 backbone 完全相同，都是標準 ViT。

差異只在於：
- `baseline`：不啟用 sequence adapter，沒有額外 sigma/uDTW 路徑
- `special`：在一組指定 block 輸出之間做 sequence alignment，並把對齊後的 `seq_b * sigma_b` 回灌到主幹，同時加入 `aux_loss`

因此目前實驗比較的本質是：
- 「純標準 ViT」vs
- 「標準 ViT + 一次跨層 sequence gating / uDTW regularization」

---

## 9. 預設尺寸與張量路徑（以 CIFAR-10 為例）
- 輸入影像：`[B, 3, 32, 32]`
- patch embed：`[B, 64, 192]`
- 加 cls + pos：`[B, 65, 192]`
- 經 `depth=8` 個標準 blocks 後仍為：`[B, 65, 192]`
- 若啟用 special 路徑，於指定 `seq_b` 位置將該序列改寫為 `seq_b * sigma_b`
- 最終取 `x[:, 0]` -> `LayerNorm` -> `Dropout` -> `Linear`
- logits：`[B, 10]`

---

## 10. 檔案責任分工
- `config.py`：訓練與模型設定 dataclass
- `data.py`：CIFAR-10 dataset 與 augmentation / dataloader
- `factory.py`：依 variant 建立模型
- `modules.py`：`PatchEmbed` / `TinyAttention` / `TinyMlp` / `SigmaNet`
- `blocks.py`：`StandardViTBlock`、`BlockSequenceAdapter`、uDTW fallback
- `model.py`：完整 ViT 組裝、sequence pair 選擇、統計聚合
- `train.py`：訓練、驗證、CLI 入口

---

## 11. 與舊版摘要的主要差異
如果你是對照舊版 `summary.md`，目前最重要的變更是：

- 已經沒有 `SpecialViTBlock`
- 已經沒有「中間兩層 special block」邏輯
- `special_indices` / attention 段與 MLP 段各自做 uDTW 的設計已不存在
- 現在只有一個 `BlockSequenceAdapter`，在兩個 block 輸出之間做一次 sequence-level 對齊
- sigma 統計目前只來自 `seq_b` 這一側
- 驗證指標新增了 `brier` 與 `conf_gap`
