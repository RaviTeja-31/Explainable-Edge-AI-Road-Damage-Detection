# PotholeLens

**Explainable, edge-oriented AI for road-damage detection and maintenance prioritisation**  
MSc Data Science and Artificial Intelligence research project · Sheffield Hallam University  
**Researcher:** Raviteja Bondada

PotholeLens is a reproducible computer-vision research pipeline that detects and localises visible road damage in photographs. It compares three YOLO object detectors under a limited GPU budget, evaluates the selected detector on held-out images, investigates image-level damage-presence decisions, produces occlusion-sensitivity explanations, and demonstrates an interpretable maintenance-priority heuristic.

> **Research status:** Proof of concept, not a production road-inspection tool, a validated pavement-condition index, or evidence of performance on physical edge hardware. The final assessed deliverable is the executed research notebook and its analytical outputs; a web application is not part of this project's final scope.

## Research question and objectives

**Research question:** How can a lightweight and explainable computer-vision model detect common road-damage types with sufficient accuracy, transparency and computational feasibility to support practical infrastructure monitoring?

The project aims to:

1. Select and justify suitable public road-damage data and object-detection models.
2. Audit the images and annotations and prepare reproducible experimental splits.
3. Compare detector quality and inference speed under matched screening conditions.
4. Evaluate the selected model using held-out object-detection metrics.
5. Analyse a separate image-level damage-presence operating threshold.
6. Demonstrate diagnostic explainability and transparent maintenance-priority scoring.
7. Report constraints, limitations and reproducibility information.

## Dataset

**Dataset used:** RDD2022, Kaggle-prepared mirror: [aliabdelmenam/rdd-2022](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022).

The full published RDD2022 release is reported as **47,420 images**. The `RDD_SPLIT` directory actually audited by this notebook contains **38,385 images** from six countries: China, Czech Republic, India, Japan, Norway and the United States. These counts refer to different dataset packages and must not be confused.

| Accessible Kaggle split | Images |
|---|---:|
| Training | 26,869 |
| Validation | 5,758 |
| Test | 5,758 |
| **Total** | **38,385** |

The detector uses five classes:

| ID | Class |
|---:|---|
| 0 | `longitudinal_crack` |
| 1 | `transverse_crack` |
| 2 | `alligator_crack` |
| 3 | `other_corruption` |
| 4 | `pothole` |

**Data availability and rights:** Download the dataset from its source and check the current dataset record, licence, attribution and reuse conditions before using or redistributing any material. Raw RDD2022 photographs are **not included** in this repository. The dataset is third-party material; a licence for this repository's own content, if added later, would not override dataset or dependency licences.

### GPU-safe working sample

The executed `gpu_safe` experiment selected **4,500 training, 600 validation and 700 test images** from the existing source partitions. The notebook's `balanced_select()` function uses a fixed seed, cycles through damage classes, avoids selecting an image twice *within a split*, and retains background examples. It does not claim that the full source dataset was globally deduplicated.

**Important:** Subset selection makes the experiment feasible and repeatable within Kaggle limits; results are not full-dataset benchmark results. A multi-country dataset also does **not** establish generalisation to a country completely absent from training.

## Method and workflow

```text
RDD2022 Kaggle split
      |
      v
Dataset audit and exploratory analysis
      |
      v
Class-aware GPU-safe subset (4,500 / 600 / 700)
      |
      v
Matched two-epoch screening: YOLOv8s | YOLOv8m | YOLO11s
      |
      v
Accuracy / F1 / inference-latency selection
      |
      v
Selected YOLOv8s: eight further epochs, 512-pixel input
      |
      v
Held-out detection evaluation: precision, recall, F1, mAP, latency
      |
      +--> Separate image-level threshold evaluation
      +--> Occlusion-sensitivity diagnostic heatmaps
      +--> Heuristic Low / Medium / High priority examples
      |
      v
Notebook outputs, CSV/JSON evidence and saved model weights
```

The screening score implemented in the notebook is:

```text
selection_score = mAP50 + 0.25 * F1 - 0.0005 * inference_ms
```

It is an **experiment-specific selection rule**, not a universal model-quality score. The selected detector was **YOLOv8s**.

### Maintenance-priority calculation

For each detected box, the actual final notebook computes:

```python
area_ratio = box_area / image_area
priority_score = severity_weight * confidence * (1 + min(area_ratio * 30, 2.0))
priority_label = "High" if priority_score >= 6 else "Medium" if priority_score >= 3 else "Low"
```

The illustrative severity weights are 2 for longitudinal/transverse cracks, 3 for other corruption, 4 for alligator cracks and 5 for potholes. **These weights and cut-offs are a research heuristic and have not been validated by civil engineers.**

## Results from the executed notebook

### Held-out object detection (YOLOv8s)

| Metric | Test result |
|---|---:|
| Precision | 0.514 |
| Recall | 0.483 |
| F1-score | 0.498 |
| mAP@0.5 | **0.460** |
| mAP@0.5:0.95 | 0.199 |
| Inference latency | Approximately 8.7 ms/image |

### Separate image-level damage-presence analysis

The confidence threshold was chosen on a **validation** sample and then evaluated on a separate **350-image test** sample.

| Metric | Test result |
|---|---:|
| Selected confidence threshold | 0.10 |
| Accuracy | 89.7% |
| Precision | 95.2% |
| Recall | 93.4% |
| F1-score | 94.3% |
| Balanced accuracy | 72.5% |
| Specificity | 51.6% |

**Do not report 89.7% as object-detection accuracy.** It measures whether an image was flagged as containing *any* road damage, not whether individual boxes correctly localised damage. The damage-positive test sample was imbalanced, so balanced accuracy and specificity are essential context. The primary localisation result remains **mAP@0.5 = 0.460**.

The latency was measured in the notebook's GPU environment; it is **not a benchmark on a deployed edge device**. Occlusion heatmaps show sensitivity to image regions; they do not establish causal model understanding or human-validated explanation quality.

## Run the notebook on Kaggle

**Main notebook:** [`potholes.ipynb`](potholes.ipynb) (place this file in the repository root or update the link if you organise notebooks into a folder).

1. Open [Kaggle Notebooks](https://www.kaggle.com/code) and create/import a notebook from `potholes.ipynb`.
2. Choose a **GPU accelerator**. The executed final run used a **Tesla T4**. Check PyTorch/CUDA compatibility before starting; the earlier P100 environment produced an incompatible-kernel error.
3. In **Add Input**, attach [RDD2022](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022). The dataset should contain `RDD_SPLIT/train`, `RDD_SPLIT/val` and `RDD_SPLIT/test`, each with `images` and `labels` directories.
4. Enable internet access if needed for the notebook's initial package installation and pretrained-weight downloads, subject to your Kaggle account permissions.
5. In the configuration cell, keep `RUN_MODE = "gpu_safe"` to reproduce the reported run. This sets `IMG_SIZE = 512`, screening epochs to 2, final fine-tuning epochs to 8 and the sample limits described above.
6. Run the **research cells in order**: installation, imports, configuration, dataset discovery/audit, EDA, sampling, YAML, screening, model selection, fine-tuning, held-out evaluation, threshold analysis, occlusion sensitivity and priority demonstration.
7. **Skip the legacy local-application-generation cell**: it remains in the archived executed notebook but is outside the final project scope. You may then run the subsequent experiment-summary/output-export cell. The final full-`/kaggle/working` backup cell is optional, very large and unnecessary for reproducing the research results.
8. Download the selected CSV/JSON outputs, figures and model weights that you are permitted to share. A fresh run may produce different timings and slightly different values if package versions, accelerator or data availability change.

The notebook installs its own Python dependencies using a `%pip install` cell. Main packages are `ultralytics`, `torch` (provided by the Kaggle runtime), `opencv-python-headless`, `pandas`, `numpy`, `matplotlib`, `tqdm`, `pyyaml`, `scikit-learn` and `pillow`. **No exact environment lockfile is included**; for exact replication, record package versions from the runtime as well as the Kaggle accelerator and dataset version.

### Outputs

The notebook writes research files below:

```text
/kaggle/working/potholelens_final_gpu_safe/
├── outputs/
│   ├── dataset_audit.csv
│   ├── subset_manifest.csv
│   ├── quick_model_comparison.csv
│   ├── final_detection_metrics.csv
│   ├── val_image_level_scores.csv
│   ├── test_image_level_scores.csv
│   ├── test_85_target_operating_metrics.csv
│   └── experiment_summary.json
├── runs/                       # training runs and weights
├── rdd2022_subset/             # temporary copied working images/labels
└── potholelens_rdd2022_gpu_safe.yaml
```

Graphs and XAI demonstrations are displayed in the executed notebook; some supporting plots also appear in the training-run directories. Do not assume every displayed figure is independently exported as a PNG. The paths in the subset manifest refer to Kaggle's working environment, not a portable local directory. The subset and complete Kaggle working archives can be several gigabytes and should **not** be pushed to ordinary Git history.

## Suggested repository layout

This is a **suggested structure**, not a claim that every file has already been uploaded:

```text
PotholeLens/
├── README.md
├── potholes(5).ipynb
├── results/              # selected, shareable CSV/JSON outputs if uploaded
├── figures/              # selected, permitted figures if uploaded
└── docs/                 # optional project documentation
```

Do not commit raw RDD2022 images, temporary Kaggle subsets, huge ZIP archives, personally identifiable imagery, credentials or access tokens. Commit a trained model only if its size and distribution rights permit it; otherwise explain how to recreate it from the notebook. If a `requirements.txt` is later added, generate and test it from the actual working environment rather than inventing pinned versions.

## Limitations and next steps

- Moderate detection/localisation performance; the object-detection mAP@0.5 was 0.460.
- Limited GPU budget, restricted sample and short fine-tuning schedule.
- Unequal representation of road-damage types; weak image-level background specificity.
- No country-holdout or independent external-domain test.
- Inference speed was not measured on physical edge hardware.
- Occlusion sensitivity is diagnostic only; the priority heuristic has no engineering or practitioner validation.

Future work would include longer full-dataset training, improved minority-class and hard-negative sampling, country-holdout testing, calibration, edge-hardware benchmarks, systematic XAI evaluation and expert validation under appropriate ethics approval.

## Ethics and academic integrity

The approved **UREC1** design used published secondary data only; there were **no recruited participants, surveys, interviews, human user tests or newly collected field images**. Dataset attribution and reuse restrictions must be respected. Results in this README are from the executed final notebook, not independently re-run as part of this README's creation.

AI assistance was used for drafting, interpretation and debugging support; the technical work and its final interpretation must be checked against the executed notebook and submitted project report. Do not describe prototype outputs as validated road-maintenance recommendations.

## Notebook integrity

The archived executed notebook has this SHA-256 checksum:

```text
File:   potholes(5).ipynb
SHA256: 41382bc73a7e0643d7a0d4dceb32ef39f8b537a6b99158021dc851db003d0649
```

This checksum matches **the exact archived notebook examined for this README**. Editing or re-saving the notebook may change the checksum; re-compute it if you upload a modified version.

```bash
sha256sum 'potholes(5).ipynb'
```

**External dataset:** https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022  
**Software/dependency licences:** Check the licences of Ultralytics, PyTorch and other dependencies before reuse or redistribution. No separate open-source licence for the repository's original code is asserted by this README.
