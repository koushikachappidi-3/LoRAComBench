# LoRAComBench

## Table of Contents
- [Purpose](#purpose)
- [Tech Stack](#Tech-Stack)  
- [System Requirements](#system-requirements)  
- [Installation Guide](#installation-guide)  
- [Features and Walkthrough](#features-and-walkthrough)  
- [Design Decisions](#design-decisions)  

## Purpose
**LoRAComBench** is a research project designed to evaluate how well multiple **LoRA (Low-Rank Adaptation)** adapters—each trained on different tasks—can be merged into a single multi-task model.

The benchmark focuses on two heterogeneous NLP tasks:
- **AG News** — Text Classification  
- **GSM8K** — Mathematical Reasoning  

This project demonstrates skills in **LLM fine-tuning**, **adapter merging**, **dataset preprocessing**, **classical baselines**, and **deep learning experimentation** using **PEFT**, **GPT-2**, and **scikit-learn**.

## Tech Stack

**Languages & Core Tools**
- Python 3.9+
- Jupyter Notebook

**Machine Learning & NLP**
- Hugging Face Transformers  
- PEFT (Parameter-Efficient Fine-Tuning)  
- GPT-2 (124M)  
- scikit-learn  
- NumPy / Pandas  

**Datasets**
- AG News (Text Classification)  
- GSM8K (Math Reasoning)

**Visualization**
- Matplotlib  
- Seaborn  

**Environment**
- Google Colab (GPU: Tesla T4)  
- Virtualenv / venv  

## System Requirements

**Prerequisites:**  
Before running the project, install:

- **Python (3.8 or later)**
- **pip**  
- **Git**
- **GPU (Recommended)** – NVIDIA T4/RTX or Colab T4GPU

## Installation Guide

### 1. Clone the Repository
```bash
git clone git@github.com:koushikachappidi-3/LoRAComBench.git
cd LoRAComBench
```
### 2. Create Virtual Environment (Optional)

```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```
### 3\. Install Dependencies

`pip install -r requirements.txt`

### 4\. Prepare Datasets

AG News and GSM8K are automatically downloaded via the `datasets` library.\
No manual setup is required.

### 5. Run Experiments

```bash
python src/lora_training.py --task agnews
python src/lora_training.py --task gsm8k
python src/lora_merge.py
python src/evaluation.py
```

Features and Walkthrough
------------------------

### 1\. LoRA Training

-   Loads GPT-2 (124M)

-   Applies LoRA with rank = 16

-   Fine-tunes separately for AG News and GSM8K

-   Saves adapters under `models/`

### 2\. Dataset Exploration (EDA)

**AG News**

-   Balanced dataset (30k/class)

-   Articles are short (20--60 words)

-   Length does *not* predict news category

**GSM8K**

-   Questions: 20--80 words

-   Answers: 20--120 words

-   Moderate positive correlation (0.48)

-   Longer questions require deeper reasoning

### 3\. Adapter Merging

-   Extracts LoRA matrices

-   Performs element-wise average

-   Loads merged adapter into GPT-2
### . Result Summary

| Model | AG News | GSM8K |
| --- | --- | --- |
| LoRA_AGNews | 0.841 | 0.021 |
| LoRA_GSM8K | 0.048 | 0.190 |
| Merged Adapter | 0.316 | 0.133 |

<img width="613" height="451" alt="image" src="https://github.com/user-attachments/assets/8e416df1-8be9-445c-af73-1427c6ec6da4" />


Merging causes significant performance degradation, demonstrating strong task interference.

### 1\. Base Model Choice

GPT-2 (124M) chosen due to:

-   Fast training

-   Stable behavior

-   Lower computational cost

### 2\. LoRA Hyperparameters

-   Rank: 16

-   Learning Rate: 2e-4

-   Epochs: 3\
    Optimized for low GPU usage while maintaining performance.

### 3\. Merging Strategy

Naive weight averaging chosen as a **baseline** to highlight:

-   Knowledge conflict

-   Feature overwriting

-   Lack of task routing

Future improvements may include:

-   TIES-Merging

-   Tensorized LoRA fusion

-   LoRAHub weighted merging

-   Dynamic skill-aware routing
