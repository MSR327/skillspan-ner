---
# Copy this file to the model repo on the Hub as README.md and fill in the TODOs.
language: en
license: apache-2.0
library_name: transformers
pipeline_tag: token-classification
base_model: jjzha/jobbert-base-cased
datasets:
  - jjzha/skillspan
tags: [ner, skill-extraction, job-postings, hr-tech]
model-index:
  - name: jobbert-skill-extractor
    results:
      - task: { type: token-classification }
        dataset: { name: SkillSpan, type: jjzha/skillspan, split: test }
        metrics:
          - { type: f1, value: TODO }
---

# Job Posting Skill Extractor

Extracts **skills** (e.g. "mentor junior engineers") and **knowledge** (e.g. "Python", "AWS")
spans from English job postings.

- **Demo:** TODO link to Space
- **Code, training & CI:** TODO link to GitHub repo

## Usage

The model has **two tagging heads** (6 outputs: SKILL O/B/I + KNOWLEDGE O/B/I), so use
the `skillex` decoder rather than the stock `pipeline`:

```python
# pip install git+https://github.com/MSR327/skillspan-ner.git
from skillex.predict import SkillExtractor
ex = SkillExtractor("TODO/jobbert-skill-extractor")
ex.extract("We need a data engineer with strong SQL and stakeholder management skills.")
```

## Results (SkillSpan test set, exact span-match F1)

| Model | Precision | Recall | F1 (all) | F1 skill | F1 knowledge |
|---|---|---|---|---|---|
| bert-base-cased (baseline) | TODO | TODO | TODO | TODO | TODO |
| **jobbert-base-cased (this model)** | TODO | TODO | TODO | TODO | TODO |

The checkpoint was selected on validation F1; test was evaluated once.

## Training

5 epochs, lr 5e-5, batch 16, warmup 10%, weight decay 0.01, max length 256, seed 42.
Labels: two independent BIO taggers sharing one encoder, one for SKILL and one for KNOWLEDGE,
because 4.4% of knowledge spans in SkillSpan sit inside a skill span (e.g. "QA" in
"promoting all QA relevant topics"); a single BIO sequence would discard them.
Loss = sum of the two heads' cross-entropies, computed on the first subword of each word only.
Hardware: TODO (e.g. Kaggle T4, ~N min).

## Limitations

- Trained on English tech and house-job postings only (SkillSpan); expect lower recall in other domains.
- Soft-skill spans are long and fuzzy, so their boundaries are less reliable than knowledge spans.
- Not intended for automated hiring decisions.

## Citation

Zhang, Jensen, Sonniks, Plank. *SkillSpan: Hard and Soft Skill Extraction from English Job Postings.* NAACL 2022.
