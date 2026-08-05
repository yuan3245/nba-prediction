#!/usr/bin/env python3
"""
將實驗 notebook 轉換為可由命令列執行的 Python 腳本。

用法：
    python tools/nb2py.py NBA_prediction_betting_clean.ipynb src/run_experiment.py

轉換規則：
  - 只取 code cell，依原順序串接（notebook 的執行順序由此固定）
  - markdown cell 的標題轉為註解，保留章節結構供閱讀
  - 於檔首插入執行環境設定（輸出目錄、BLAS 執行緒控制、matplotlib backend）
  - 於檔尾附加環境指紋輸出，記錄實際生效的數值函式庫
"""
import json
import sys
from pathlib import Path

HEADER = '''#!/usr/bin/env python3
"""
NBA 勝負預測與投注策略 — 完整實驗流程

本檔由 NBA_prediction_betting_clean.ipynb 自動轉換產生（見 tools/nb2py.py）。
請勿直接編輯本檔；實驗邏輯的修改應於 notebook 進行後重新轉換。

用法：
    python src/run_experiment.py

環境變數：
    OUTPUT_DIR           產物輸出目錄，預設 outputs
    EXPERIMENT_THREADS   限制數值函式庫執行緒數。設為 1 可提升數值確定性，
                         代價是顯著增加執行時間。未設定時使用全部核心。
    USE_CACHED_FEATURES  設為 1 時，若 dataset c 的特徵清單快取存在則直接讀取，
                         略過耗時的 RFECV。詳見 REPRODUCIBILITY.md。
"""
import os
import time

# 執行緒數必須在 numpy 匯入前設定 —— BLAS 於載入時即完成執行緒池初始化，
# 之後再修改環境變數不會生效。
_THREADS = os.environ.get("EXPERIMENT_THREADS")
if _THREADS:
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_var] = _THREADS

# 無圖形介面環境下強制使用非互動式後端，避免 matplotlib 嘗試開啟視窗
os.environ.setdefault("MPLBACKEND", "Agg")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_START_TIME = time.time()
print(f"[run_experiment] 輸出目錄：{OUTPUT_DIR}")
print(f"[run_experiment] 執行緒設定：{_THREADS or '全部核心'}")
print("=" * 70)

'''

FOOTER = '''

# ============================================================================
# 執行環境指紋
# ============================================================================
# 記錄實際生效的套件版本與數值函式庫。REPRODUCIBILITY.md 的分析顯示，
# 數值差異的根因位於 BLAS 層，而該層不受 requirements.txt 管轄，
# 因此需要獨立記錄，作為日後比對的依據。

def _write_environment_fingerprint():
    import io
    import json as _json
    import platform
    import sys as _sys

    import numpy as _np
    import pandas as _pd
    import scipy as _scipy
    import sklearn as _sklearn
    import xgboost as _xgb

    _buf = io.StringIO()
    try:
        _stdout, _sys.stdout = _sys.stdout, _buf
        _np.show_config()
    finally:
        _sys.stdout = _stdout

    fingerprint = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {
            "numpy": _np.__version__,
            "pandas": _pd.__version__,
            "scikit-learn": _sklearn.__version__,
            "xgboost": _xgb.__version__,
            "scipy": _scipy.__version__,
        },
        "threads_env": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "numpy_build_config": _buf.getvalue(),
        "elapsed_seconds": round(time.time() - _START_TIME, 1),
    }

    path = os.path.join(OUTPUT_DIR, "environment.json")
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(fingerprint, f, ensure_ascii=False, indent=2)
    return path


print("=" * 70)
_fp_path = _write_environment_fingerprint()
_elapsed = time.time() - _START_TIME
print(f"[run_experiment] 環境指紋已寫入：{_fp_path}")
print(f"[run_experiment] 完成，總耗時 {_elapsed / 60:.1f} 分鐘")
'''


def convert(nb_path: Path, out_path: Path) -> None:
    notebook = json.loads(nb_path.read_text(encoding="utf-8"))

    chunks = [HEADER]
    for cell in notebook["cells"]:
        source = "".join(cell["source"]).rstrip()
        if not source:
            continue

        if cell["cell_type"] == "markdown":
            # 只保留標題行，作為章節分隔註解
            headings = [
                line.lstrip("# ").strip()
                for line in source.split("\n")
                if line.startswith("#")
            ]
            if headings:
                chunks.append(
                    "\n# " + "=" * 74 + "\n"
                    + "\n".join(f"# {h}" for h in headings)
                    + "\n# " + "=" * 74 + "\n"
                )
        elif cell["cell_type"] == "code":
            chunks.append(source + "\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(chunks) + FOOTER, encoding="utf-8")

    line_count = out_path.read_text(encoding="utf-8").count("\n")
    print(f"已產生 {out_path}（{line_count} 行）")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
