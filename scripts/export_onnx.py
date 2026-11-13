"""Export the model to ONNX for in-browser inference: fp32, plus a small browser build
(int8 word embeddings + fp16-stored weights, fp32 compute; ~same F1 as fp32).

    python scripts/export_onnx.py --model srish327/deberta-v3-skillspan-crf --out onnx_out

The graph returns raw logits only; BIO/CRF decoding runs outside the model (Python or JS),
using skillex_crf.json from the same repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForTokenClassification, AutoTokenizer


def embedding_gathers(g: onnx.GraphProto) -> list[str]:
    """The word-embedding lookup (128k x 768 = 393 MB in fp32): int8 here costs ~0 F1."""
    sizes = {i.name: int(np.prod(i.dims)) for i in g.initializer}
    return [n.name for n in g.node if n.op_type == "Gather" and sizes.get(n.input[0], 0) > 10_000_000]


def store_weights_fp16(path: Path, min_size: int = 4096) -> None:
    """Store large fp32 weights as fp16 + a Cast back to fp32: half the bytes, fp32 math.

    Unlike dynamic int8 on the dense layers (-4.8 test F1, and worse with padded batches,
    since activation scales are per tensor), this keeps predictions essentially unchanged.
    """
    m = onnx.load(str(path))
    g = m.graph
    casts = []
    for init in g.initializer:
        if init.data_type != onnx.TensorProto.FLOAT or int(np.prod(init.dims)) < min_size:
            continue
        arr = onnx.numpy_helper.to_array(init).astype(np.float16)
        orig = init.name
        init.CopyFrom(onnx.numpy_helper.from_array(arr, orig + "_fp16"))
        casts.append(onnx.helper.make_node("Cast", [orig + "_fp16"], [orig], to=onnx.TensorProto.FLOAT))
    nodes = casts + list(g.node)  # topological order: casts first
    del g.node[:]
    g.node.extend(nodes)
    onnx.save(m, str(path))
    print(f"fp16-stored weights: {len(casts)}")


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="onnx_out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).eval()
    enc = tok(["Experience with Python and SQL ."], return_tensors="pt")
    fp32 = out / "model.onnx"
    torch.onnx.export(
        LogitsOnly(model),
        (enc["input_ids"], enc["attention_mask"]),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={k: {0: "batch", 1: "seq"} for k in ("input_ids", "attention_mask", "logits")},
        opset_version=17,
        dynamo=False,
    )
    small = out / "model_quantized.onnx"
    quantize_dynamic(
        str(fp32),
        str(small),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Gather"],
        nodes_to_quantize=embedding_gathers(onnx.load(str(fp32), load_external_data=False).graph),
    )
    store_weights_fp16(small)
    for f in sorted(out.glob("*.onnx")):
        print(f"{f.name}: {f.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
