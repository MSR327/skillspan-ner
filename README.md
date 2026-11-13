# skillex: skill extraction from job postings

Fine-tunes a transformer to extract **skills** and **knowledge** spans from job ads,
then ships it. **[Live demo](https://huggingface.co/spaces/srish327/skill-extractor)** (runs fully in your browser) · [model on the Hub](https://huggingface.co/srish327/deberta-v3-skillspan-crf).
Also included: FastAPI + Docker service, Prometheus metrics, CI with an eval gate,
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

SkillSpan **test set** (3,569 sentences), exact span match, micro-F1 over both span types.
Mean ± std over seeds, Kaggle T4 (~8 min per 5-epoch run). Full logs are in the Kaggle sweep notebook.

| Model | Decoding | Seeds | **F1** | F1 skill | F1 knowledge |
|---|---|---|---|---|---|
| bert-base-cased | Viterbi | 2 | 0.558 ± 0.004 | – | – |
| jobbert-base-cased | argmax | 3 | 0.579 ± 0.007 | – | – |
| jobbert-base-cased | Viterbi | 3 | 0.598 ± 0.007 | – | – |
| deberta-v3-base | argmax | 3 | 0.611 | – | – |
| deberta-v3-base | Viterbi | 3 | 0.625 ± 0.005 | 0.540 | 0.697 |
| deberta-v3-base, 10 epochs | Viterbi | 2 | 0.632 ± 0.006 | 0.545 | 0.707 |
| **deberta-v3-base + CRF** | **CRF Viterbi** | 3 | **0.645 ± 0.005** | **0.565** | **0.713** |

**Reference:** SkillSpan paper (Zhang et al., 2022, Table 5, 5 seeds, CRF): JobBERT 0.597 combined
(skill 0.561, knowledge 0.639). Their best skill F1 is JobSpanBERT at 0.566. Their combined score follows Benikova et al. (2014),
so treat the comparison as close but not identical.

What we found:
- **Our JobBERT reproduces the paper** (0.598 vs 0.597), so the pipeline and metric are sound.
- **Constrained decoding is free F1.** Viterbi that forbids `O → I` adds about 1.5 to 2 F1 to every model, mostly through precision.
- **A learned CRF per head adds 2 more F1** over fixed-rule Viterbi (0.645 vs 0.625). It also wins on the **validation** set
  (0.648 vs 0.617 mean), so this choice was not tuned on test.
  The CRF model's argmax-only F1 collapses (0.26), as expected: with CRF loss, the emissions are only meaningful together with the transitions.
- **Training longer helps only a little** (+0.7 F1 at 10 epochs, n=2, within noise). Best epoch was still the last one.
- We beat the paper on knowledge by a wide margin (+7 F1). On skill we roughly match it (0.565 vs 0.566).
  Long skill spans (avg 4.3 words) with fuzzy boundaries are the hard part. Annotator agreement is only κ ≈ 0.70 to 0.75.

## In-browser demo (no server)

The [demo Space](https://huggingface.co/spaces/srish327/skill-extractor) runs the model client-side: transformers.js tokenizer,
onnxruntime-web, and CRF Viterbi decoding in JavaScript (`space_static/extract.js`). Getting a 736 MB DeBERTa small enough for a browser,
measured on the full test set (`scripts/export_onnx.py`):

| Browser build | Size | Test F1 (CRF) |
|---|---|---|
| fp32 ONNX (matches PyTorch to 3e-5) | 736 MB | 0.6318 |
| default dynamic int8 (all MatMuls) | 243 MB | **0.0009** (broken: attention MatMuls can't take int8) |
| int8 on dense weights + embeddings only | 244 MB | 0.584 (batch 1), 0.503 (padded batches) |
| **int8 embeddings + fp16-stored weights, fp32 compute** | **271 MB** | **0.6309** |

The JS pipeline agrees with Python on 99.4% of spans (`scripts/js_parity.mjs`). Two bugs this check caught:
transformers.js leaves out SentencePiece's word-start marker for single words, so every token ID was wrong,
and dynamic int8 activation scales get distorted by padding.

## Roadmap (portfolio plan)

- [ ] **Week 1:** train both configs, push the best to the Hub (`hub_model_id` in config), fill in `MODEL_CARD.md`, publish
- [ ] **Week 2:** push repo to GitHub, set `MODEL_ID` repo variable for the CI eval gate, deploy `space/` to HF Spaces, add W&B (`report_to: wandb`)
- [ ] **Week 3:** Kaggle notebook "Skill extraction from job posts with JobBERT, end to end" + one NLP competition baseline notebook
- [ ] **Week 4:** Kaggle competition submission + write-up; error analysis (where does it fail?); resume bullets with real numbers

## Resume bullet template (fill with your real numbers)

> Fine-tuned JobBERT for skill/knowledge extraction on SkillSpan (span F1 **X** vs **Y** for a BERT baseline);
> served via FastAPI/Docker with Prometheus metrics and a CI eval gate that blocks regressions; live demo on HF Spaces.
