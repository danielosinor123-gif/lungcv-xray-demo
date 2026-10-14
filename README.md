# LungCV — Chest X-Ray Classification (4-Class)

![Python](https://img.shields.io/badge/python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Model](https://img.shields.io/badge/params-3.54M-green)
![Status](https://img.shields.io/badge/status-research%20prototype-red)

> **Research prototype only. This output is not a medical diagnosis.**

A chest X-ray classification system that classifies radiographs into **NORMAL,
PNEUMONIA, COVID, and TB**, built on 79,594 public images from 7 datasets, with
patient-wise evaluation, calibrated confidence scores, and a live web demo.

---

## Performance

**Validation: 91.2% accuracy · macro-F1 0.903** (1,804 held-out images)

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| PNEUMONIA | 0.958 | 0.963 | **0.960** | 562 |
| COVID | 0.936 | 0.868 | 0.901 | 569 |
| TB | 0.861 | 0.876 | 0.868 | 113 |
| NORMAL | 0.857 | 0.912 | 0.884 | 560 |
| **macro avg** | 0.903 | 0.905 | **0.903** | 1,804 |

Confidence calibration: temperature scaling (T = 0.7), fitted on validation —
reported probabilities match observed accuracy.

---

## The classes

| Class | Description | Training source |
|---|---|---|
| NORMAL | Healthy chest radiograph | NIH ChestX-ray14 (`No Finding`), Kermany, COVID-Radiography |
| PNEUMONIA | Bacterial / non-COVID viral pneumonia (pediatric + adult) | Kermany, QU-Ex `Non-COVID`, COVID-Radiography `Viral Pneumonia` |
| COVID | Confirmed COVID-19 pneumonia | QU-Ex, COVID-Radiography |
| TB | Pulmonary tuberculosis | TB Chest X-ray Database, Shenzhen + Montgomery sets |

---

## Datasets (79,594 images indexed)

| Dataset | Publisher | Images | License |
|---|---|---:|---|
| [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | paultimothymooney | 5,856 | CC BY 4.0 |
| [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data) | NIH (via nih-chest-xrays) | 20,000 sampled | CC0 |
| [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) | tawsifurrahman | 15,153 | Attribution |
| [COVID-QU-Ex](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu) | anasmohammedtahir | 33,920 | CC BY-SA 4.0 |
| [TB Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) | tawsifurrahman | 4,200 | Attribution |
| [TB Shenzhen](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen) | raddar | 327 | Public data |
| [TB Montgomery](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery) | raddar | 138 | Public data |

Data processing: exact MD5 + perceptual-hash deduplication, label validation,
class balancing (per-class cap at export), 4% burn-in border crop.

---

## Model

| Component | Detail |
|---|---|
| Architecture | EfficientNetB0 (ImageNet transfer learning) + GAP + Dropout + Dense(4, softmax) |
| Parameters | 3,543,478 (17 MB `.keras` artifact) |
| Training data | 12,009 class-balanced images (patient-wise 70/15/15 split, seed 42) |
| Augmentation | Horizontal flip, ±7° rotation, contrast/brightness jitter |
| Preprocessing | 4% burn-in border crop → 224×224 RGB (identical in training and inference) |
| Calibration | Temperature scaling on validation |

---

## Pipeline (6 self-verifying stages)

```
01 metadata (7 datasets, zip-streamed)  ->  02 validate + dedup (MD5 + pHash)
->  03 patient-wise splits (leakage-asserted)  ->  04 export + train + evaluate
+ calibrate  ->  05 package  ->  06 deploy to Hugging Face
```

Every stage asserts its own output integrity: shard member counts vs metadata,
patient-leakage checks, class-presence gates. Execution is fully automated via
the Kaggle API with dataset attachments declared in code.

---

## Demo

Live **Streamlit** app: upload an X-ray → class probabilities with calibrated
confidence bars. Example buttons for all 4 classes (verified in-distribution
samples). Images are processed in memory only — nothing is stored.

The model + config load from [the Hugging Face model repo](https://huggingface.co/danielaxv3/lungcv-xray-classifier).

`inference.py` in the package reproduces the exact preprocessing contract.

---

## Limitations

- Research prototype — not a diagnostic device, no regulatory clearance.
- COVID ↔ pneumonia is the main confusion axis (radiological overlap).
- Out-of-distribution films (unseen scanners/styles) degrade accuracy —
  external hospital validation is the next milestone.
- Public datasets carry label noise; some sources lack true patient IDs.

## Roadmap

1. Auxiliary lung-segmentation head (masks available in QU-Ex / COVID-Radiography)
2. Fine-tune stage + per-source error analysis
3. External validation on unseen hospital cohorts
4. Clinician-reviewed labels + DICOM ingestion
