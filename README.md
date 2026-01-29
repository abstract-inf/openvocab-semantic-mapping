# Open-Vocab Semantic Mapping

This project implements a real-time semantic mapping and object detection system. It leverages **YOLO-Worldv2** for open-vocabulary detection, allowing the system to identify objects based on custom text prompts without retraining. By combining high-speed detection with depth estimation, it enables the projection of objects into 3D space for robotic mapping and navigation.

---

## 🚀 Getting Started

### Prerequisites

* **Python 3.8+**
* **CUDA-enabled GPU** (recommended for real-time performance)
* **Webcam or Orbbec Depth Camera**

### Installation

1. **Clone the repository:**
```bash

git clone https://github.com/abstract-inf/openvocab-semantic-mapping.git
cd openvocab-semantic-mapping
```


2. **Install dependencies:**
Run the following command to install all necessary libraries:
```bash
pip install torch torchvision ultralytics transformers pyyaml pandas numpy opencv-python Pillow tqdm open_clip_torch

```



### 📦 Dataset Setup

**IMPORTANT:** This repository does **not** include the validation dataset or annotations. You must manually download the **COCO val 2017** dataset to run the benchmark script.

1. **Download the following:**
* **Images:** [val2017.zip](http://images.cocodataset.org/zips/val2017.zip) (~1GB).
* **Annotations:** [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).


2. **Organize the files** to match the project tree:
* Unzip images into a folder named `val2017`.
* Unzip annotations into `annotations_trainval2017/annotations/`.



---

## 🛠 Project Components

### 1. Performance Comparison

**File:** `comparison_script.py`

* **What it does:** Benchmarks **YOLOv10n**, **RT-DETR**, **YOLO11n**, and **YOLO-Worldv2** on COCO val.
* **How to run:**
```bash
python comparison_script.py
```


* **Output:** Generates `model_comparison_results.csv` and `detailed_predictions.csv`.

### 2. Data Visualization

**File:** `data_visualization.py`

* **What it does:** Processes CSV results to create accuracy and efficiency plots.
* **How to run:**
```bash
python data_visualization.py
```

* **Output:** Saves charts to the `./paper_plots/` folder.

### 3. Real-Time Detection

**Files:** `openvocab_realtime.py` & `realtime_yolo.py`

* **What they do:** * `openvocab_realtime.py`: Finds objects via text prompts (e.g., "door") using YOLO-World.
* `realtime_yolo.py`: Runs standard YOLO11n detection.


* **How to run:**
```bash
python openvocab_realtime.py  # or realtime_yolo.py
```


* **How to close:** Press **'q'** to exit the stream.

---

## 📂 Project Structure

```text
.
├── comparison_script.py      # Benchmarking engine
├── data_visualization.py     # Plotting and visual analysis
├── openvocab_realtime.py     # Interactive open-vocab detection
├── realtime_yolo.py          # Standard real-time detection
├── requirements.txt          # Dependency list
├── paper_plots/              # Generated analysis charts
├── val2017/                  # COCO images (Install manually)
└── annotations_trainval2017/ # COCO annotations (Install manually)

```

Would you like me to generate a bash script that automatically downloads and organizes the COCO dataset for you?
