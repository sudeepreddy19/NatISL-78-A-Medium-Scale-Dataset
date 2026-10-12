# NatISL-78: A Medium-Scale Dataset for Indian Sign Language Recognition

## Overview

**NatISL-78** is a medium-scale dataset and experimental framework for isolated-word Indian Sign Language (ISL) recognition. It contains 78 word-level sign classes recorded from 8 signers in natural recording conditions.

The accompanying research evaluates visual-cue-based recognition, MediaPipe Holistic landmark-based sequence learning, and image-based deep-learning architectures. The best reported NatISL-78 result is obtained using **MediaPipe Holistic with EfficientNet-B0**, achieving **96.86% Top-1 accuracy** under the paper's reported evaluation setup.

> **Task:** Isolated-word ISL recognition. This project does not claim to perform continuous sentence translation.

## Dataset

According to the paper, NatISL-78 contains:

- **78** ISL word classes
- **8** signers
- **5** recordings per word per signer
- **3,120** video samples in total
- Approximately **2–4 seconds** per video
- Natural recording conditions with variation in lighting, background, camera angle, signer appearance, age group, and signing style

Each signer contributes `78 × 5 = 390` videos, giving `8 × 390 = 3,120` videos overall.

### Dataset organization

The paper describes videos as organized by sign class and signer, but it does not provide the exact on-disk directory names. A conceptual layout is:

```text
NatISL-78/
├── <word_class_1>/
│   ├── <signer_1>/
│   │   ├── <sample_1>.mp4
│   │   └── ...
│   └── ...
├── <word_class_2>/
└── ...
```

Replace the placeholders with the actual folder and file names used by your dataset. This is an illustrative layout, not a verified directory listing.

## Methodology

The paper investigates multiple approaches to recognizing ISL signs.

### 1. Video frame sampling

Each video is normalized to **30 uniformly sampled frames**, preserving the temporal order of the gesture while providing a fixed-length input sequence.

### 2. MediaPipe Holistic landmark extraction

MediaPipe Holistic extracts landmark information from the sampled frames, including pose, left-hand, right-hand, and selected facial landmarks.

### 3. Skeleton rendering and RGB fusion

The detected landmarks are rendered as a skeleton image and fused with the original RGB frame. The paper gives the fusion equation:

`X_t = clip(α I_t + β S_t)`

where `I_t` is the RGB frame and `S_t` is the rendered skeleton frame. The reported weights are `α = 0.70` and `β = 0.90`. The fused frame is resized to **224 × 224** and normalized.

### 4. Spatial feature extraction

EfficientNet-B0 is used as a feature extractor, with its classification head removed. It produces a **1,280-dimensional feature vector per frame**. With 30 sampled frames, the video representation has shape:

`30 × 1280`

### 5. Temporal learning and attention

A bidirectional LSTM (BiLSTM) models temporal relationships between frame features. The paper specifies a hidden dimension of **256** and applies temporal attention to give more weight to informative frames.

The attention-weighted representation is passed through fully connected layers with ReLU activation and dropout, followed by a Softmax classifier for the 78 word classes.

### 6. Training configuration reported in the paper

| Setting | Value |
|---|---|
| Number of classes | 78 |
| Frames per video | 30 |
| Input image size | 224 × 224 |
| EfficientNet-B0 feature dimension | 1,280 |
| BiLSTM hidden dimension | 256 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Batch size | 16 |
| Training duration | 46 epochs |
| Loss | Categorical cross-entropy |

These are the settings described for the proposed EfficientNet-B0 framework in the paper. The paper does not provide all implementation details needed to reproduce the training exactly.

## Experiments

The paper reports results for a visual-cue-based method, a MediaPipe landmark-based BiLSTM with attention, EfficientNet-B0 and MobileNetV2, and evaluations on selected INCLUDE words and the INCLUDE-50 and INCLUDE-263 benchmarks.

### Results on NatISL-78

| Model | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| MediaPipe + EfficientNet-B0 | **96.86%** | **98.28%** | **98.92%** |
| MediaPipe + MobileNetV2 | 87.15% | 96.15% | 98.07% |

The paper also reports the following validation result for MediaPipe Holistic landmarks with BiLSTM and attention:

| Metric | Reported result |
|---|---:|
| Validation Top-1 accuracy | 81.54% |
| Training accuracy | 84.70% |
| F1-score | 0.81 |
| Precision | 0.83 |

### Evaluation on INCLUDE

| Evaluation set | Model | Top-1 | Top-3 | Top-5 |
|---|---|---:|---:|---:|
| 78 selected INCLUDE words | MediaPipe + EfficientNet-B0 | 93.77% | 98.90% | 99.62% |
| 78 selected INCLUDE words | MediaPipe + MobileNetV2 | 87.55% | 98.93% | 99.61% |
| INCLUDE-50 | MediaPipe + EfficientNet-B0 | 94.24% | 98.95% | 99.48% |
| INCLUDE-263 | MediaPipe + EfficientNet-B0 | 88.79% | 95.98% | 97.43% |

These numbers are reproduced from the paper and should be interpreted in the context of its evaluation protocol. They should not be treated as a guarantee of performance on a different dataset, split, or live webcam feed.

### Signer-independent evaluation

The paper reports a substantial performance drop when testing on signers excluded from training:

| Model | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| MediaPipe + EfficientNet-B0 | 8.20% | 13.00% | 17.17% |
| MediaPipe + MobileNetV2 | 5.64% | 14.36% | 19.74% |

This is a key limitation: high accuracy under the paper's main evaluation setup does **not** imply reliable recognition of unseen signers. The authors associate the generalization difficulty in part with the dataset's limited number of signers.

## Visual-cue experiment

The paper also explores adding external visual cues associated with target words. On NatISL-78, the best reported validation accuracy for this experiment is **61.00%**, with an F1-score of **0.60** under the reported configuration. The paper reports weaker results on the selected INCLUDE subset and concludes that this visual-cue approach was not reliable enough for the recognition task.

## Reference-similarity validation

The collected signs were compared with reference videos from the Indian Sign Language Research and Training Centre (ISLRTC). Dynamic Time Warping (DTW) was used to align temporal differences, followed by cosine similarity. The paper reports an average similarity score of **0.789**.

## Requirements

The paper identifies MediaPipe, EfficientNet-B0, BiLSTM, and attention as components of the evaluated framework, but it does **not** specify a complete software environment or exact dependency versions.

A practical implementation may require:

- Python
- OpenCV for video reading and frame processing
- MediaPipe for Holistic landmark extraction
- PyTorch or another framework matching the actual implementation
- A pretrained EfficientNet-B0 implementation
- NumPy and supporting data-processing libraries

Install only the dependencies required by your actual source code. No verified `requirements.txt`, exact package versions, or tested installation commands were included in the paper.

## Running the project

The paper does not provide source-code entry points, command-line arguments, checkpoint paths, or a complete training/inference script. Therefore, exact commands cannot be specified reliably from the paper alone.

To make this section runnable for your repository, document the actual commands and files for:

1. Preparing the dataset and labels
2. Extracting 30 frames per video
3. Extracting MediaPipe landmarks and rendering skeletons
4. Training the model
5. Evaluating Top-1, Top-3, and Top-5 accuracy
6. Running inference on a video or webcam, if implemented

Do not use example commands until they match the scripts in your repository.

## Limitations

- The dataset contains eight signers, limiting the diversity available for signer-independent evaluation.
- The reported signer-independent performance is substantially lower than the main reported results.
- The task is isolated-word recognition, not full sentence-level ISL translation.
- The paper does not include enough implementation detail to guarantee exact reproduction from the paper alone.
- Performance may vary with recording conditions, signer variation, preprocessing, data splits, and model implementation.

## Citation

If you use NatISL-78 or the methodology described in the paper, cite the research work:

```text
Sudeep Sanshi, Chetan S H, Nandeesh IB, B. Gurubasavaraj,
Umadevi F M, and Sujata C,
“NatISL-78: A Medium-Scale Dataset for Indian Sign Language Recognition,”
Department of Computer Science (AI), KLE Technological University,
Hubballi, India.
```

Add the final publication venue, year, DOI, or paper URL when those details are available.

## Acknowledgements

The paper compares the proposed approach with the INCLUDE benchmark datasets and uses ISLRTC reference videos for gesture-similarity validation.

## License and dataset access

The uploaded paper does not specify a software license, dataset license, or public download URL. Add these details only after confirming the project's actual distribution and usage permissions.
