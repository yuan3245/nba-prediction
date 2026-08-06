# 可重現性驗證報告

本文件記錄本專案在**四個不同環境**執行同一份實驗程式碼的比對結果。

結論先講：**論文結果在三個獨立環境上完整重現，唯一對不上的是開發用的 WSL2 環境。**

---

## 1. 驗證的四個環境

| # | 環境 | CPU | 微架構 | OS | 執行緒 | numpy wheel |
|---|---|---|---|---|---|---|
| E1 | 論文原始 | AMD Ryzen 7 PRO 4750U | Zen 2 | Windows | 16 | Windows (MSVC) |
| E2 | 另一台桌機 | AMD Ryzen 5 7500F | Zen 4 | Windows | 12 | Windows (MSVC) |
| E3 | AWS EC2 `c7i.2xlarge` | Intel Xeon Platinum 8488C | Sapphire Rapids | Ubuntu 24.04 | 8 | manylinux (gcc 14.2.1) |
| E4 | 本機開發環境 | AMD Ryzen 7 PRO 4750U | Zen 2 | WSL2 (Hyper-V) | 16 | manylinux (gcc 14.2.1) |

E3 由 `infra/`（Terraform）與 `ansible/`（Ansible）自動佈建，執行時間 65.1 分鐘。
E1／E2／E4 為手動建立之環境，執行時間約 90 分鐘。

套件版本在四個環境完全一致（見 `requirements.txt`）：
Python 3.14 / pandas 3.0.1 / numpy 2.4.2 / scikit-learn 1.8.0 / xgboost 3.2.0 / matplotlib 3.10.8 / scipy 1.17.1

---

## 2. 比對結果

| 比對項目 | E1 論文 | E2 桌機 | E3 EC2 | E4 WSL2 |
|---|---|---|---|---|
| dataset c 特徵數 | 28 | 28 | **28** | **33** |
| 特徵集內容 | 論文附錄表 16 | 同 E1 | **與 E1 完全相同** | 相異 |
| dataset c 七模型 Accuracy | 論文表 7 | 同 E1 | **全數相符** | 相異 |
| Exp4 ROI 2024–25 | +13.55% | +13.55% | **+13.55%** | +17.70% |
| Exp4 ROI 2025–26 | +15.53% | +15.53% | **+15.53%** | +13.50% |

E3 對 E1 的重現是**逐位**的，包含特徵集的具體成員而不只是數量。

### 2.1 E4（WSL2）的偏差細節

| 策略 | E1／E2／E3 | E4 WSL2 | 差距 |
|---|---|---|---|
| 2024–25 Exp1（僅 EV/Kelly） | −3.64% | +4.84% | 8.5 pp・**正負翻轉** |
| 2024–25 Exp2（+Cons） | +10.73% | +7.90% | 2.8 pp |
| 2024–25 Exp3（+Band） | −0.25% | +12.16% | 12.4 pp・**正負翻轉** |
| 2024–25 Exp4（+Cons+Band） | +13.55% | +17.70% | 4.2 pp |

RFECV 收斂時的 CV 最佳 AUC：E3 = 0.7111，E4 = 0.7113，**差距僅 0.0002**。

被替換的特徵多為主客場鏡像對（`elo_x` ↔ `elo_y`、`ortg_opp_5_x` ↔ `ortg_opp_5_y`、
`drtg_max_5_x` ↔ `drtg_5_y`），預測力近乎等價。

---

## 3. 分析

### 3.1 放大機制：離散決策點

模型品質本身幾乎沒有變化（AUC 差 0.0002），但特徵集換掉了 5 個成員。

原因是 **RFECV 是一個 argmax 操作**。當多組特徵子集的 CV 分數差距落在小數點第四位時，
搜尋地形近乎平坦，浮點層級的擾動就足以讓 argmax 的答案翻面。

一旦特徵集不同，下游的模型係數、預測機率、EV 篩選、Kelly 注碼全部沿著不同路徑走，
最終在 ROI 上放大成正負翻轉。誤差不是「累積」放大的，是在**離散決策點上被翻轉**放大的。

**可遷移的結論**：尋找不可重現性時，應優先檢查 pipeline 中把連續量轉換為離散決策的位置
——argmax、閾值判斷、排序、early stopping。

### 3.2 擾動來源：尚未定位

以下單變數假設均被資料推翻：

| 假設 | 反證 |
|---|---|
| CPU 微架構／SIMD 差異 | E1 與 E4 是**同一顆 CPU**，結果不同 |
| 執行緒數 | E1 與 E4 都是 16 執行緒，結果不同 |
| 作業系統（Windows vs Linux） | E3 是 Linux，結果卻與 Windows 的 E1 一致 |
| numpy build toolchain | E3 與 E4 使用**同一個 manylinux wheel**，結果不同 |

三種微架構、兩家 CPU 廠商、三種執行緒數、兩種 OS、兩套 numpy build 全部收斂到 28 個特徵；
只有 E4 是 33。因此**擾動來源尚未定位**，目前僅能確認 E4 是異常值。

尚未排除的候選：WSL2 虛擬化層對 cache topology 的回報方式（OpenBLAS 依此決定矩陣分塊參數）。

---

## 4. 已知的實驗設計缺陷

本次比對**同時改變了多個變數**，因此無法建立因果宣稱：

1. **Python patch 版本未鎖定**
   Ansible playbook 原本使用 `uv python install --python 3.14`，粒度不足。
   結果 E3 安裝 3.14.7、E4 為 3.14.6。已修正為指定完整 patch 版本。

2. **執行緒數未鎖定**
   `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` 三者皆為 `null`
   （見 `outputs/*/environment.json`），OpenBLAS 依偵測到的核心數自行決定。
   平行歸約的加總順序會因此改變。

3. **BLAS kernel 未鎖定**
   numpy wheel 內含的 OpenBLAS 以 `DYNAMIC_ARCH` 編譯，會在執行期依 CPU 特性選擇
   kernel。`OPENBLAS_CORETYPE` 未設定。

**下一步**：固定其他所有條件，一次只變動一個變數，才能逐一隔離擾動來源。

---

## 5. 對可重現性的認知修正

| 層次 | 是否已鎖定 | 鎖定方式 |
|---|---|---|
| 套件版本 | ✅ | `requirements.txt` 完整 pin |
| 直譯器 patch 版本 | ✅（本次修正） | `uv python install 3.14.6` |
| 機器規格 | ✅ | Terraform `instance_type` 固定 |
| 執行緒數 | ❌ | 需設 `OMP_NUM_THREADS` |
| BLAS kernel | ❌ | 需設 `OPENBLAS_CORETYPE` |
| CPU 微架構 | 部分 | 僅在雲端可控（鎖 instance family） |

**Container 無法解決最底下兩層。** Container 隔離的是檔案系統與行程，並未虛擬化 CPU；
同一個 image 在不同 CPU 上執行，OpenBLAS 的 runtime dispatch 仍會選到不同 kernel。

---

## 6. 如何重現本驗證

```bash
# 1. 佈建機器
cd infra
terraform init
terraform apply

# 2. 配置環境（可重複執行，第二次應為 changed=0）
cd ../ansible
ansible-playbook provision.yml

# 3. 執行完整實驗（約 65 分鐘），結果自動取回 outputs/ec2/
ansible-playbook run.yml

# 4. 比對
cd ..
diff <(grep -v '^#' outputs/dataset_c_features.txt | sort) \
     <(grep -v '^#' outputs/ec2/dataset_c_features.txt | sort)
diff outputs/betting_all_experiments.csv outputs/ec2/betting_all_experiments.csv

# 5. 銷毀
cd infra
terraform destroy
```

環境指紋（Python 版本、套件版本、執行緒環境變數、numpy build config）
會在每次執行時寫入 `outputs/environment.json`。

---

## 7. 相關檔案

| 路徑 | 說明 |
|---|---|
| `infra/` | Terraform：EC2 / Security Group / Key Pair |
| `ansible/provision.yml` | 環境配置（冪等性已驗證：changed=4 → changed=0） |
| `ansible/run.yml` | 執行實驗並取回結果 |
| `src/run_experiment.py` | 由 notebook 轉出的可執行腳本 |
| `tools/nb2py.py` | notebook → script 轉換工具 |
| `outputs/environment.json` | 本機環境指紋 |
| `outputs/ec2/environment.json` | EC2 環境指紋 |
| `docs/cpu_comparison.txt` | 兩環境 CPU 與核心數對照 |
