# NLP HW03 — Text Classification: Fine-tuning vs Zero-shot LLMs

This repository contains the code and prediction outputs for a text classification assignment comparing fine-tuned transformer models, zero-shot LLM prompting, and traditional baselines on the emotion classification task.

---

## Task

**Dataset:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion)  
**Task:** Multi-class emotion classification (6 classes: sadness, joy, love, anger, fear, surprise)  
**Splits:** Train: 16,000 | Validation: 2,000 | Test: 2,000

---

## Repository Structure

```
├── NLP_HW03.ipynb                  # Main notebook (all steps)
├── predictions_baselines.csv       # Random, majority, BOW+LR predictions
├── predictions_transformers.csv    # DistilBERT and RoBERTa predictions
├── predictions_groq.csv            # Groq Llama 3.1 8B zero-shot predictions
├── predictions_ollama.csv          # Ollama Gemma 3 4B zero-shot predictions
└── README.md
```

---

## Methods

### 1. Baselines
- **Random baseline** — theoretical calculation based on class distribution
- **Majority baseline** — always predicts the most frequent class (`joy`)
- **BOW + Logistic Regression** — TF-IDF features (unigrams + bigrams, max 20k features) with scikit-learn's `LogisticRegression`

### 2. Fine-tuned Transformer Models (LoRA / PEFT)
Both models were fine-tuned using [LoRA](https://arxiv.org/abs/2106.09685) via the HuggingFace `peft` library. Only ~1% of parameters were trained.

| Model | Size | Epochs | Learning Rate |
|-------|------|--------|---------------|
| `distilbert-base-uncased` | 67M params | 3 | 2e-4 |
| `roberta-base` | 125M params | 3 | 2e-4 |

### 3. Zero-shot LLM Prompting
No training data was used. Each test example was sent to the model with a structured prompt asking it to output exactly one emotion label.

| Model | Provider | Type |
|-------|----------|------|
| Llama 3.1 8B Instant | Groq API | Cloud |
| Gemma 3 4B | Ollama (local) | Local |

Output parsing: lowercased and stripped the model's response, then checked for exact label match followed by substring match.

---

## Results

| Model | Accuracy | F1 (macro) |
|-------|----------|------------|
| Random baseline | ~0.17 | ~0.17 |
| Majority baseline | ~0.35 | ~0.06 |
| Groq Llama 3.1 8B (zero-shot) | 0.569 | 0.456 |
| Ollama Gemma 3 4B (zero-shot) | 0.562 | 0.473 |
| DistilBERT + LoRA | 0.811 | 0.730 |
| BOW + Logistic Regression | 0.831 | 0.733 |
| RoBERTa + LoRA | 0.827 | 0.751 |

---

## How to Run

### Requirements

```bash
pip install datasets transformers peft accelerate evaluate scikit-learn groq
```

### Steps

1. Open `NLP_HW03.ipynb` in Google Colab
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Mount Google Drive and set `BASE_PATH` to your desired output folder
4. Add your `GROQ_API_KEY` to Colab Secrets (🔑 panel on the left sidebar)
5. For the local Ollama model: install [Ollama](https://ollama.com), run `ollama run gemma3:4b`, then expose it via [ngrok](https://ngrok.com) and update `OLLAMA_URL` in the notebook
6. Run all cells in order

---

## AI Use Disclosure

Claude (Anthropic) and Google Gemini were used to assist with debugging, and explanation throughout this assignment.

---

## Course

NLP — University of Michigan-Flint
