# LungCV - Chest X-Ray Finding Scores (v4.4, research release)

![Python](https://img.shields.io/badge/python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Status](https://img.shields.io/badge/status-research%20prototype-red)

> **Research prototype only. This output is not a medical diagnosis.**

A chest X-ray system that scores each finding **independently** and also reports
a normal/abnormal verdict from a dedicated 4-group head. Built on 71,024 public
images from 7 datasets, with patient-wise evaluation and a live web demo.

**Currently deployed:** `danielaxv3/lungcv-xray-multilabel-v44` (v4.4, seed 1 - the
best single seed). The v3.2 model remains available by setting
`MODEL_REPO=danielaxv3/lungcv-xray-multilabel-v32`.

---

## What the model does (v4.4)

Three outputs:

| Output | Shape | Meaning |
|---|---|---|
| `classifier16` | 16 findings | independent sigmoid, one probability per finding (COVID, Pneumonia, TB, Cardiomegaly, ?) |
| `classifier` | 4 groups | v3.2's own 4-group head, kept and re-trained jointly - this is where v4.4 recovers v3.2's normal-vs-disease strength |
| `segmenter` | 224?224 mask | v3.2's seg head, frozen, zero targets |

The demo reports per-finding at calibrated thresholds (not flat 0.5), shows a
normal/abnormal verdict at a calibrated threshold
(`normal_vs_disease_threshold` in the config), flags **uncertain** films where no
finding crosses its reporting threshold, and an **OOD detector** flags films that
do not resemble the training data.

---

## Measured (not promised)

### Clean-644 cohort (zero training contact, flip TTA)

| head | acc | macro-F1 |
|---|---:|---:|
| v3.2 (baseline) | 0.891 | 0.892 |
| **v4.4 seed 1 (deployed)** | **0.921** | **0.921** |
| v4.4 seed 2 | 0.905 | 0.906 |
| v4.4 seed 3 | 0.809 | 0.807 |
| **v4.4 3-seed ensemble** | **0.932** | **0.932** |

Single-seed mean is 0.878 - **below** v3.2. The deployed seed is the best single
seed, labelled as such. The ensemble is what the research numbers describe.

### 518-film radiologist-labelled benchmark (contamination-checked)

| | v3.2 | v4.4 ensemble |
|---|---:|---:|
| 4-group accuracy | 0.330 | **0.600** |
| normal-vs-disease AUC | 0.7355 | **0.8208** |
| normal-vs-disease accuracy | 0.687 | 0.581 @ 0.5 -> **0.749** @ calibrated 0.83 |

### Known, published failures

- **TB**: on 262 TB films never seen in training, the TB head scores TB at
  0.002-0.019 and calls them "No Finding" (0.46-0.62). TB's 0.99 validation F1 was
  memorisation of one oversampled source.
- **Seed variance is 11.2 points** - no single run is evidence.
- **Prevalence**: on a realistic 15%-abnormal worklist the baseline model flags
  69% of films; v4.4 flags 31%. Absolute accuracy depends on case mix - measure
  on your own prevalence before use.

---

## The findings (16)

Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis,
Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax,
"No Finding", COVID, TB - each reported independently at its calibrated
threshold. The demo will not call anything positive unless it clears its
threshold, and it says so when nothing does.

## Datasets (71,024 training tiles)

| Dataset | Publisher | Role | License |
|---|---|---|---|
| [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data) | NIH | bulk + 16-finding labels, quota lifted 800->3000/finding | CC0 |
| [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) | tawsifurrahman | COVID / viral pneumonia | Attribution |
| [COVID-QU-Ex](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu) | anasmohammedtahir | COVID / non-COVID | CC BY-SA 4.0 |
| [TB Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-database) | tawsifurrahman | TB | Attribution |
| TB Shenzhen / Montgomery | raddar | TB (**held out of v4.4 training as a generalisation test**) | Public data |
| CheXpert (official test set) | Stanford | external benchmark only | see CheXpert terms |
| Kermany pneumonia | paultimothymooney | no longer mountable (removed 2026-11) | CC BY 4.0 |

---

## Honest limits

- Research prototype. Not for diagnosis or treatment.
- Familiar-cohort accuracy ? expert-labelled accuracy; the CheXpert benchmark
  shows where the two disagree (labeler divergence, NORMAL especially).
- Normal-vs-disease at a flat 0.5 threshold underperforms; the calibrated
  threshold in the config is the correct one.
- No clinical validation; no regulatory status; no claim to be state of the art.
- The TB head generalises poorly - published deliberately rather than hidden.
