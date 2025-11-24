# LoRAComBench

This repo contains all work for the **LoRAComBench** project, which evaluates how well **LoRA (Low-Rank Adaptation)** adapters—trained on different tasks—can be merged into a single multi-task model.

## SETUP

This project uses **Hugging Face Transformers** and **PEFT** for LoRA fine-tuning.  
A GPU-enabled environment (Colab or local NVIDIA GPU) is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```
Datasets (AG News, GSM8K) are automatically downloaded using the `datasets` library.

## Project Overview

The goal of this project is to analyze **LoRA adapter composition** across two very different NLP tasks:

1. **AG News** – Text Classification  
2. **GSM8K** – Math Word Problem Reasoning  

We compare:

- Individual LoRA adapters  
- Classical machine-learning baselines  
- A merged LoRA adapter using naive weight averaging  

Main evaluation metrics include:

- Accuracy  
- Macro F1  
- Performance drop after merging (task interference)  
## Implementation

All implementation code is located in the `src` directory.

Includes:

- LoRA fine-tuning  
- Classical baseline models  
- Adapter merging utilities  
- Evaluation scripts  
- Preprocessing and EDA  

Run experiments:

```bash
python src/lora_training.py --task agnews
python src/lora_training.py --task gsm8k
python src/lora_merge.py
python src/evaluation.py
```
Next Steps


-   Explore advanced adapter fusion techniques (TIES-Merging, LoRAHub, tensor fusion)

-   Add more tasks (dialogue, code, reasoning)

-   Evaluate modular routing approaches for multi-skill LoRA pipelines

## Team & Contributions (Group-1)

**1. Nisarga Vishwamanjuswamy**  
- Dataset acquisition (AG News, GSM8K)  
- Preprocessing and cleaning  
- Built input pipelines for classical ML and LoRA models  
- Wrote dataset description and preprocessing sections  

**2. Pramod Kumar Reddy Parvath Reddy**  
- Implemented LoRA pipeline using GPT-2 + PEFT  
- Adapter training and merging  
- Integrated AG News and GSM8K  
- Feature engineering (TF-IDF, tokenization)  
- Model evaluation  
- Wrote methodology and conclusion sections  

**3. Sai Teja Kusireddy**  
- Implemented classical ML baselines (Logistic Regression, Naive Bayes)  
- TF-IDF feature extraction  
- Model training and testing  
- Wrote ML implementation sections  

**4. Sreenitha Rayapuraju**  
- Performed full EDA  
- Generated class distribution plots, histograms, scatterplots, and heatmaps  
- Analyzed patterns and insights  
- Wrote EDA sections  

**5. Harshitha Reddy Mannemala**  
- Tested and debugged LoRA pipeline  
- Verified adapter behavior  
- Wrote the abstract and supported documentation  
- Created graphs, tables, and visualizations  

**6. Koushika Chappidi**  
- Evaluated classical vs LoRA models  
- Analyzed model outputs and adapter merge behavior  
- Implemented XAI (top features for LogReg and Naive Bayes)  
- Wrote introduction and results/discussion sections  

