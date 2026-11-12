---
title: Job Posting Skill Extractor
emoji: 🧰
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
---

Extracts skill and knowledge spans from job postings. It uses DeBERTa-v3 with two BIO heads and a learned CRF,
fine-tuned on SkillSpan. Code: https://github.com/MSR327/skillspan-ner
