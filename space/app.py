"""Gradio demo for Hugging Face Spaces.

Uses the skillex package (installed from GitHub via requirements.txt) because the
model has two tagging heads that the stock HF pipeline can't decode.
"""

import os

import gradio as gr

from skillex.predict import SkillExtractor

MODEL_ID = os.environ.get("MODEL_ID", "YOUR_HF_USERNAME/jobbert-skill-extractor")
extractor = SkillExtractor(MODEL_ID)

EXAMPLE = (
    "We are looking for a Data Engineer with strong Python and SQL experience. "
    "You will design ETL pipelines on AWS, collaborate with stakeholders, and "
    "mentor junior engineers. Knowledge of Spark and Airflow is a plus."
)


def _highlight(text, spans, etype):
    return {
        "text": text,
        "entities": [
            {"entity": etype, "start": s["start"], "end": s["end"]} for s in spans if s["type"] == etype
        ],
    }


def run(text: str):
    r = extractor.extract(text)
    # Two views because SKILL and KNOWLEDGE spans can overlap.
    return (
        _highlight(text, r["spans"], "SKILL"),
        _highlight(text, r["spans"], "KNOWLEDGE"),
        "\n".join(r["skills"]),
        "\n".join(r["knowledge"]),
    )


demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(lines=8, label="Job posting", value=EXAMPLE),
    outputs=[
        gr.HighlightedText(label="Skills (what you can do)", color_map={"SKILL": "green"}),
        gr.HighlightedText(label="Knowledge (what you know)", color_map={"KNOWLEDGE": "blue"}),
        gr.Textbox(label="Skills"),
        gr.Textbox(label="Knowledge"),
    ],
    title="Job Posting Skill Extractor",
    description=f"`{MODEL_ID}`, fine-tuned on SkillSpan with separate SKILL and KNOWLEDGE taggers.",
    examples=[[EXAMPLE]],
)

if __name__ == "__main__":
    demo.launch()
