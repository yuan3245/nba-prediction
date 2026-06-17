# NBA Game Outcome Prediction: From Accuracy to Betting Profitability

> 基於機器學習之 NBA 勝負預測：從預測準度到投注報酬

此repository為這篇碩士論文的實驗程式碼。研究在 NBA 勝負預測場景上同時評估**預測準確度（accuracy，過往文獻主流指標）**與**投注報酬率（betting ROI，本研究擴充之維度）**，並觀察到兩者之間的**解耦現象**：在本研究的評估中，準確度最低的模型反而取得最高的投注報酬。

---

## 論文資訊

- **題目（中）**：基於機器學習之 NBA 勝負預測：從預測準度到投注報酬
- **題目（英）**：Machine Learning for NBA Game Outcome Prediction: From Accuracy to Betting Profitability
- **作者**：何承遠
- **單位**：國立陽明交通大學 資訊科學與工程研究所（NYCU, Institute of Computer Science and Engineering）
- **指導教授**：袁賢銘、張庭榕
- **年份**：2026

---

## 主要發現

> 完整的實驗設定、統計檢定與討論請見論文本文，以下僅摘錄核心數字。

- **特徵帶來的準確度提升**：加入新計算特徵後，跨七個模型的平均準確度由 `0.6334`（baseline）提升至 `0.6547`（+2.13 pp）。
- **單一最高準確度**：Stacking Classifier 搭配完整特徵（dataset a′）達 `67.62%`。
- **準確度 vs. 報酬的解耦**：XGBoost 搭配精簡特徵（dataset c）的準確度為 `63.46%`，為七個模型中最低；但在投注模擬中，XGBoost 於樣本外賽季（2025-26，至 2026/3/31）以 Exp4 策略取得 ROI `+15.53%`，為各模型最高。
- **對照 baseline**：同期「一律下注熱門方（Favorite baseline）」的 ROI 為 `−3.37%`，本研究策略領先約 `+18.90 pp`。樣本外期間 XGBoost 下注 134 場、命中 69 場、命中率 51.5%。

這顯示在投注情境下，單純追求預測準確度未必等同於追求獲利——過往多數預測文獻聚焦於 accuracy，較少系統性地延伸到投注 ROI，本研究在同一框架下同時檢視兩者並呈現此差異。

---

## 專案結構

```
nba-prediction-betting/
├── README.md                              # 本說明檔
├── requirements.txt                       # Python 套件需求
├── .gitignore                             # 忽略產生的輸出檔與環境檔
├── LICENSE                                # 程式碼授權（MIT）
├── NBA_prediction_betting_clean.ipynb     # 完整實驗 notebook（主程式）
└── data/
    ├── README.md                          # 資料字典與來源說明
    ├── nba_games_331.csv          # 比賽資料（含特徵工程後欄位）
    └── cbs_moneyline_2024_2026_331.csv  # CBS Sports moneyline 賠率
```

---

## 環境需求與安裝

建議使用 Python 3.10 以上版本。

```bash
# 1. 建立虛擬環境
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. 安裝套件
pip install -r requirements.txt
```

> **備註**：實驗結果對 `scikit-learn` 與 `xgboost` 版本敏感，不同版本可能造成小數點後的差異（例如 STACK + dataset a′ 在部分版本為 `67.25%` 而非 `67.62%`）。若需完全重現論文數字，請以論文撰寫當時的套件版本為準。

---

## 如何執行

**從專案根目錄**啟動 Jupyter（notebook 內的資料路徑以根目錄為基準，例如 `data/nba_games_331.csv`）：

```bash
jupyter notebook NBA_prediction_betting_clean.ipynb
# 或
jupyter lab
```


---

## Notebook 章節對照

| 區段 | 內容 |
|------|------|
| 1. 資料載入與清理 | 讀入比賽資料、`dropna` 處理 |
| 2. 特徵工程 | Elo 動態實力指標（賽季衰減 α=0.25）、團隊爆發力指標、賽程密集度、旅行負荷（Haversine + 時區差）、滾動平均（近 5/10 場） |
| 3. 訓練／開發／樣本外切分 | 訓練集 2016-17～2023-24；開發集 2024-25（1,170 場）；樣本外 2025-26（995 場） |
| 4. 五組資料集（a / a′ / b / c / d） | baseline、完整特徵、Pearson 過濾、RFECV 包裹、ElasticNetCV 嵌入 |
| 5. 勝負預測實驗 | 七個模型：Lasso、Logistic、Ridge、Linear SVM、Random Forest、XGBoost、Stacking Classifier |
| 6. 投注模擬實驗 | 三道篩網（共識 + 分數帶 + EV/Kelly）× 四種策略（Exp1–Exp4） |

---

## 資料來源與授權

- **程式碼**：以 [MIT License](LICENSE) 授權。
- **資料**：`data/` 內的資料衍生自第三方公開來源（比賽數據來自 Basketball Reference，賠率來自 CBS Sports），僅供本論文之學術研究與重現使用，資料的著作權與使用條款屬原始來源所有。

---

## 引用

若本程式碼或資料對你的研究有幫助，請引用本論文：

```bibtex
@mastersthesis{ho2026nba,
  author  = {Ho, Cheng-Yuan},
  title   = {Machine Learning for NBA Game Outcome Prediction: From Accuracy to Betting Profitability},
  school  = {National Yang Ming Chiao Tung University},
  year    = {2026},
  type    = {Master's Thesis}
}
```
