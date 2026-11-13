---
language: en
license: apache-2.0
library_name: transformers
pipeline_tag: token-classification
base_model: microsoft/deberta-v3-base
datasets:
  - jjzha/skillspan
tags: [ner, skill-extraction, job-postings, crf, hr-tech]
model-index:
  - name: deberta-v3-skillspan-crf
    results:
      - task: { type: token-classification }
        dataset: { name: SkillSpan, type: jjzha/skillspan, split: test }
        metrics:
          - { type: f1, value: {{TEST_F1}}, name: span F1 (exact match, CRF decoding) }
---

# Job Posting Skill Extractor (DeBERTa-v3 + CRF)

Extracts **skills** (e.g. "mentor junior engineers") and **knowledge** (e.g. "Python", "AWS")
spans from English job postings. Skill and knowledge spans can overlap.

- **Demo:** {{SPACE_URL}} (runs in your browser, using `onnx/model_quantized.onnx`: 271 MB, test F1 0.6309 vs 0.6318 for fp32)
- **Code, training, experiments & CI:** https://github.com/MSR327/skillspan-ner

## Usage

The model has **two BIO tagging heads** (6 outputs: SKILL O/B/I + KNOWLEDGE O/B/I) and
learned CRF transitions stored in `skillex_crf.json`, so use the `skillex` decoder rather
than the stock `pipeline`:

```python
# pip install git+https://github.com/MSR327/skillspan-ner.git
from skillex.predict import SkillExtractor
ex = SkillExtractor("{{MODEL_ID}}")   # picks up the CRF automatically
ex.extract("We need a data engineer with strong SQL and stakeholder management skills.")
```

## Results (SkillSpan test set, exact span-match F1)

This checkpoint (seed 43, chosen as the best seed on **validation**; test evaluated once):
**F1 {{TEST_F1}}** (skill {{TEST_F1_SKILL}}, knowledge {{TEST_F1_KNOWLEDGE}}).

Same setup across 3 seeds, compared with alternatives (mean ± std):

| Model | Decoding | Seeds | F1 | F1 skill | F1 knowledge |
|---|---|---|---|---|---|
| bert-base-cased | Viterbi | 2 | 0.558 ± 0.004 | – | – |
| jobbert-base-cased | Viterbi | 3 | 0.598 ± 0.007 | – | – |
| deberta-v3-base | Viterbi | 3 | 0.625 ± 0.005 | 0.540 | 0.697 |
| **deberta-v3-base + CRF (this setup)** | CRF | 3 | **0.645 ± 0.005** | **0.565** | **0.713** |
| SkillSpan paper, JobBERT + CRF (Zhang et al., 2022) | CRF | 5 | 0.597 | 0.561 | 0.639 |

The paper's combined score follows Benikova et al. (2014), so the comparison is close but not
identical. Our JobBERT run reproduces the paper's number (0.598 vs 0.597).

## Training

- DeBERTa-v3-base encoder with two linear BIO heads, one for SKILL and one for KNOWLEDGE.
  Two heads are needed because 4.4% of knowledge spans sit inside a skill span.
- Each head has a linear-chain CRF over first-subword positions. `O → I` is hard-forbidden.
  Loss is the sum of the two heads' CRF negative log-likelihoods.
- 5 epochs, lr 3e-5, batch 32, warmup 10%, weight decay 0.01, max length 256, fp32 weights with fp16 autocast.
- Best epoch chosen by validation CRF F1. One Kaggle T4, about 8 minutes.

## Limitations

- Trained on English tech and house-job postings only (SkillSpan). Expect lower recall in other domains.
- Skill spans are long and fuzzy: boundaries for skills (F1 ~0.56) are much less reliable than for knowledge (~0.71).
  Human annotator agreement on this data is κ ≈ 0.70 to 0.75.
- Not intended for automated hiring decisions.

## Citation

Zhang, Jensen, Sonniks, Plank. *SkillSpan: Hard and Soft Skill Extraction from English Job Postings.* NAACL 2022.
