# skillex: skill extraction from job postings

Fine-tunes a transformer to extract **skills** and **knowledge** spans from job ads,
then ships it: FastAPI + Docker service, Prometheus metrics, CI with an eval gate,
and a live Gradio demo on Hugging Face Spaces.

**Research question:** does domain-adapted pretraining (JobBERT) beat generic BERT on skill extraction?

```
SkillSpan (HF) ─▶ train.py ─▶ HF Hub model ─▶ serve/ (FastAPI, Docker, /metrics)
                     │                      ├▶ space/ (Gradio demo)
                     └▶ metrics/*.json      └▶ CI eval-gate (fails if F1 drops)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,serve,dev]"
pytest -q

# Train (use a GPU: Kaggle/Colab T4 is fine, ~15–25 min per run)
python -m skillex.train --config configs/bert_baseline.yaml
python -m skillex.train --config configs/jobbert.yaml

# Evaluate + quality gate
python -m skillex.evaluate --model outputs/jobbert/best --min-f1 0.50

# Serve
MODEL_ID=outputs/jobbert/best uvicorn serve.app:app --port 8000
curl -s localhost:8000/extract -H 'content-type: application/json' \
  -d '{"text":"Looking for a Python developer who can mentor juniors and work with AWS."}'

# Docker
docker build -t skillex . && docker run -p 8000:8000 -e MODEL_ID=YOUR_USER/jobbert-skill-extractor skillex
```

## Results

SkillSpan test set (3,569 sentences), exact span-match, single run (seed 42), Kaggle T4 x2, ~8 min per model.

| Model | Precision | Recall | **F1** | F1 skill | F1 knowledge |
|---|---|---|---|---|---|
| bert-base-cased | 0.521 | 0.573 | **0.546** | 0.451 | 0.626 |
| jobbert-base-cased | 0.549 | 0.629 | **0.586** | 0.494 | 0.667 |

Domain-adapted pretraining helps: **+4.0 F1** overall, and both span types improve.
Recall exceeds precision for both models (over-prediction is the main error), and skills
(avg 4.3 words) are much harder than knowledge (avg 1.8 words) under exact boundary matching.
Next: multi-seed runs to confirm the gap, and error analysis on false-positive spans.

## Roadmap (portfolio plan)

- [ ] **Week 1:** train both configs, push the best to the Hub (`hub_model_id` in config), fill in `MODEL_CARD.md`, publish
- [ ] **Week 2:** push repo to GitHub, set `MODEL_ID` repo variable for the CI eval gate, deploy `space/` to HF Spaces, add W&B (`report_to: wandb`)
- [ ] **Week 3:** Kaggle notebook "Skill extraction from job posts with JobBERT, end to end" + one NLP competition baseline notebook
- [ ] **Week 4:** Kaggle competition submission + write-up; error analysis (where does it fail?); resume bullets with real numbers

## Resume bullet template (fill with your real numbers)

> Fine-tuned JobBERT for skill/knowledge extraction on SkillSpan (span F1 **X** vs **Y** for a BERT baseline);
> served via FastAPI/Docker with Prometheus metrics and a CI eval gate that blocks regressions; live demo on HF Spaces.
