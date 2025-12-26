# 嵌入式系統期末專題

## 專案架構

```
src/
├── anger_detector_ju.ipynb    # 主要訓練程式ResNet50
├── anger_detector.ipynb       # 別理他
├── resnet18.ipynb             # keras 沒有 ResNet18 別理他
├── eval.ipynb                 # 模型評估
├── keras_to_onnx.ipynb        # 模型格式轉換 (Keras → ONNX) 我用 colab 跑的
├── inference.py               # 以 jetson orin nano 裡面的 code 為準
└── README.md
```

---

## 技術細節

### 資料集
- **RAF-DB (Real-world Affective Faces Database)**
- 二元分類任務：
  - **正樣本 (Anger)**: 標籤 6
  - **負樣本 (Non-Anger)**: 標籤 1, 4, 7
- 訓練集：5,076 張 | 驗證集：1,269 張 | 測試集：2,356 張
- 使用下採樣（8:1）處理類別不平衡問題

### 模型架構
- **基礎模型**: ResNet50
- **遷移學習**:
  1. 凍結 ResNet50 進行初始訓練
  2. 解凍後 30 層進行 Fine-tuning（保持 BatchNormalization 凍結）
- **全連接層**: GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)

### 訓練配置
| 參數 | 值 |
|------|-----|
| 輸入尺寸 | 224 × 224 × 3 |
| Batch Size | 32 |
| 優化器 | Adam |
| 學習率 | 1e-4 (初始) → 1e-5 (Fine-tuning) |
| 損失函數 | Binary Crossentropy |
| 資料增強 | 旋轉、平移、縮放、水平翻轉 |

---

## 模型部署流程

```
Keras (.keras) → ONNX (.onnx) → TensorRT (.engine)
```

1. **Keras → ONNX**: 使用 `tf2onnx` 轉換
2. **ONNX → TensorRT**: 使用 `trtexec` 優化編譯

---

## 即時推論 (`inference.py`)

使用 TensorRT 進行 GPU 加速推論，支援即時攝影機輸入。

### 功能
- 支援 MediaPipe 人臉偵測，自動裁切臉部區域
- 可調整偵測閾值與臉部邊界框擴展比例

### 使用方式

```bash
# 基本使用（整張畫面）
python inference.py --engine model.engine --source 0

# 啟用人臉偵測
python inference.py --engine model.engine --source 0 --face-detect

# 自訂參數
python inference.py --engine model.engine \
    --source 0 \
    --face-detect \
    --threshold 0.4 \
    --face-confidence 0.5 \
    --face-margin 0.3
```

### 參數說明
| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--engine` | TensorRT 引擎檔案路徑 | 必填 |
| `--source` | 攝影機 ID 或影片路徑 | 0 |
| `--imgsz` | 模型輸入尺寸 | 224 224 |
| `--threshold` | 憤怒判斷閾值 | 0.4 |
| `--face-detect` | 啟用人臉偵測 | False |
| `--face-confidence` | 人臉偵測信心閾值 | 0.5 |
| `--face-margin` | 臉部邊界框擴展比例 | 0.3 |

### 快捷鍵
| 按鍵 | 功能 |
|------|------|
| `q` | 結束程式 |
| `f` | 切換人臉偵測模式 |
| `s` | 保存裁切的臉部圖片 |
| `+` / `-` | 調整臉部邊界框擴展比例 |