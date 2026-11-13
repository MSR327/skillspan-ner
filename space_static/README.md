---
title: Job Posting Skill Extractor
emoji: 🧰
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: apache-2.0
---

Extracts skill and knowledge spans from job postings. It runs **entirely in the browser**: the int8 ONNX model runs on onnxruntime-web,
the tokenizer is transformers.js, and Viterbi decoding with the learned CRF is written in JavaScript.
Model: https://huggingface.co/srish327/deberta-v3-skillspan-crf · Code: https://github.com/MSR327/skillspan-ner
