// JS-vs-Python parity check for the in-browser demo. Setup: npm i @huggingface/transformers@3.5.1 onnxruntime-web@1.20.1
// (run from a dir with those node_modules; copy space_static/extract.js to extract.mjs if Node treats .js as CommonJS).
import fs from "node:fs";
import * as ort from "onnxruntime-web";
import { AutoTokenizer } from "@huggingface/transformers";
import { extract } from "../space_static/extract.js";
const S = process.argv[2];  // dir with ref.json (from predict.py) and onnx_out/
const ref = JSON.parse(fs.readFileSync(`${S}/ref.json`, "utf8"));
const tok = await AutoTokenizer.from_pretrained("srish327/deberta-v3-skillspan-crf");
const crfs = await (await fetch("https://huggingface.co/srish327/deberta-v3-skillspan-crf/resolve/main/skillex_crf.json")).json();
ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(fs.readFileSync(`${S}/onnx_out/model_quantized.onnx`));
const t0 = Date.now();
const spans = await extract({ ort, tok, session, crfs }, ref.text);
const key = (s) => s.join(":");
const js = new Set(spans.map((s) => key([s.type, s.start, s.end]))), py = new Set(ref.spans.map(key));
const both = [...js].filter((k) => py.has(k)).length;
console.log(`js spans ${js.size}, python spans ${py.size}, identical ${both}  agreement F1 ${(2*both/(js.size+py.size)).toFixed(4)}  (${Date.now()-t0} ms)`);
for (const k of [...js].filter((k) => !py.has(k)).slice(0, 5)) { const [t,a,b]=k.split(":"); console.log(" js only:", t, JSON.stringify(ref.text.slice(+a,+b))); }
for (const k of [...py].filter((k) => !js.has(k)).slice(0, 5)) { const [t,a,b]=k.split(":"); console.log(" py only:", t, JSON.stringify(ref.text.slice(+a,+b))); }
