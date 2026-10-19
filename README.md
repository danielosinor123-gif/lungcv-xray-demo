# LungCV — Chest X-Ray Finding Scores (multi-label v3)

![Python](https://img.shields.io/badge/python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Model](https://img.shields.io/badge/backbone-DenseNet121-green)
![Status](https://img.shields.io/badge/status-research%20prototype-red)

> **Research prototype only. This output is not a medical diagnosis.**

A chest X-ray system that scores each finding **independently** (sigmoid, multi-label)
for **NORMAL, PNEUMONIA, COVID, and TB** — instead of forcing a single class. Films
where no finding crosses the reporting threshold are flagged **uncertain**, and an
**OOD detector** flags films that do not resemble the training data at all. Built on
79,594 public images from 7 datasets with patient-wise evaluation and a live web demo.

---

## Performance (v3, validation)

**Argmax accuracy 82.9% · macro-F1 0.829** (1,800 held-out images)
per-finding sigmoid scores at a 0.5 threshold:

| Finding | Precision | Recall | F1 |
|---|---:|---:|---:|
| COVID | 0.820 | 0.707 | 0.759 |
| NORMAL | 0.832 | 0.690 | 0.754 |
| PNEUMONIA | 0.874 | 0.696 | 0.775 |
| TB | 0.961 | 0.886 | **0.922** |

OOD detector: distance threshold 16.3 (95th percentile of training distances);
5.3% of familiar validation films flagged — correctly calibrated.

Note: sigmoid scoring at a fixed threshold is a harder bar than softmax argmax —
unsure films score below 0.5 by design (that is the uncertainty behavior).

---

## Model history

| Version | Backbone | Output | Result | Status |
|---|---|---|---|---|
| v1 | EfficientNetB0 (ImageNet) | 4-class softmax | 91.2% acc, macro-F1 0.903 | superseded |
| v2 multitask | EfficientNetB0 + U-Net head | class + lung mask | 89.9% acc, TB F1 0.971 | superseded |
| **v3** | **DenseNet121 (ImageNet) + CLAHE** | **4 independent sigmoids + OOD** | **82.9% acc, macro-F1 0.829** | **production** |
| v3.1 | DenseNet121 (RadImageNet), fine-tuned | 4 independent sigmoids + OOD | 79.0% acc — lost to v3 | experiment |

The RadImageNet experiment (medical-pretrained backbone, two-stage fine-tuning,
empirical preprocessing probe) is preserved at
[danielaxv3/lungcv-xray-multilabel-v3-radimagenet](https://huggingface.co/danielaxv3/lungcv-xray-multilabel-v3-radimagenet).
Consistent with the RadImageNet paper, medical pretraining only helps on small
datasets — at our scale ImageNet features transfer better, and RadImageNet's mix
(CT/MRI/ultrasound) contains no projection radiographs.

---

## The findings

| Finding | Description | Training source |
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
| [TB Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-database) | tawsifurrahman | 4,200 | Attribution |
| [TB Shenzhen](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen) | raddar | 327 | Public data |
| [TB Montgomery](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery) | raddar | 138 | Public data |

Data processing: exact MD5 + perceptual-hash deduplication, label validation,
per-class balancing, patient-wise 70/15/15 splits (leakage-asserted).

---

## Model (v3)

| Component | Detail |
|---|---|
| Architecture | DenseNet121 (ImageNet transfer, frozen) + GAP + Dropout + Dense(4, sigmoid) |
| Parameters | ~7M (29.7 MB `.keras` artifact) |
| Training data | 12,000 class-balanced tiles (3,000/finding, patient-wise split, seed 42) |
| Harmonization | CLAHE (clipLimit 2.0, 8×8 tiles) — strips scanner intensity fingerprints |
| Preprocessing | grayscale → CLAHE → 224×224 → 3-channel → DenseNet `preprocess_input` |
| Loss | binary cross-entropy per finding (multi-label) |
| OOD | Euclidean distance to class centroids in GAP feature space |

The model + config + OOD centroids load from
[the Hugging Face model repo](https://huggingface.co/danielaxv3/lungcv-xray-multilabel-v3).

---

## Demo

Live **Streamlit** app: upload an X-ray → independent finding score bars, an
uncertainty flag when no finding crosses the threshold, and an OOD banner for
unfamiliar films. Example buttons for all 4 findings (verified in-distribution
samples). Images are processed in memory only — nothing is stored.

---

## Limitations

- Research prototype — not a diagnostic device, no regulatory clearance.
- COVID ↔ pneumonia is the main confusion axis (radiological overlap).
- Sigmoid scores are independent, not a probability distribution.
- Public datasets carry label noise; some sources lack true patient IDs.
- OOD detector uses Euclidean distance in feature space — a research-grade
  heuristic, not a certified distribution test.

## Roadmap

1. External validation on unseen hospital cohorts
2. Grounded LLM narrator (describes only what the model computed)
3. Data ladder: MIMIC-CXR (377k) + CheXpert (224k)
4. DICOM ingestion + triage framing, drift monitoring
