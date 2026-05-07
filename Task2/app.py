import gradio as gr

from hf_tasks import HuggingFaceTaskModule


tasks = HuggingFaceTaskModule()

css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

.tab-nav,
.tabs,
[role="tablist"] {
    flex-wrap: wrap !important;
    gap: 6px !important;
    overflow: visible !important;
}

.tab-nav button,
[role="tab"] {
    border-radius: 6px !important;
    padding: 6px 10px !important;
    font-size: 13px !important;
}

textarea {
    min-height: 52px !important;
}
"""


def rows_to_text(rows):
    lines = []
    for row in rows:
        lines.append(" - ".join(str(item) for item in row))
    return "\n".join(lines)


def zero_shot(text, labels):
    return rows_to_text(tasks.zero_shot_classification(text, labels))


def fill_mask(text):
    return rows_to_text(tasks.fill_mask(text))


def ner(text):
    return rows_to_text(tasks.named_entity_recognition(text))


def image_classification(image):
    return rows_to_text(tasks.image_classification(image))


def clear_text():
    return "", ""


def clear_two_inputs():
    return "", "", ""


def clear_media():
    return None, ""


def input_output_row():
    return gr.Row(equal_height=True)


def action_buttons():
    with gr.Row(equal_height=True):
        clear = gr.Button("Clear")
        submit = gr.Button("Submit", variant="primary")
    return clear, submit


with gr.Blocks(title="EE471 Task 2") as demo:
    gr.Markdown("# EE471 Week 10 - Task 2")

    with gr.Tab("Sentiment"):
        with input_output_row():
            with gr.Column(scale=1, min_width=420):
                text = gr.Textbox(label="input", value="I really want to take this EE471 course.", lines=2)
                clear, submit = action_buttons()
            with gr.Column(scale=1, min_width=420):
                output = gr.Textbox(label="output", lines=2)
        submit.click(tasks.sentiment_analysis, text, output)
        clear.click(clear_text, outputs=[text, output])

    with gr.Tab("Zero-shot"):
        with input_output_row():
            with gr.Column(scale=1, min_width=420):
                text = gr.Textbox(label="input", value="Berkshire keeps their cash reserves at an extremely high level.", lines=2)
                labels = gr.Textbox(label="labels", value="finance, education, technology, health", lines=1)
                clear, submit = action_buttons()
            with gr.Column(scale=1, min_width=420):
                output = gr.Textbox(label="output", lines=5)
        submit.click(zero_shot, [text, labels], output)
        clear.click(clear_two_inputs, outputs=[text, labels, output])

    with gr.Tab("Generation"):
        with input_output_row():
            with gr.Column(scale=1, min_width=420):
                prompt = gr.Textbox(label="input", value="If I continue to successfully complete all in-class exercises in software course,", lines=2)
                clear, submit = action_buttons()
            with gr.Column(scale=1, min_width=420):
                output = gr.Textbox(label="output", lines=5)
        submit.click(tasks.text_generation, prompt, output)
        clear.click(clear_text, outputs=[prompt, output])

    with gr.Tab("Fill mask"):
        with input_output_row():
            with gr.Column(scale=1, min_width=420):
                text = gr.Textbox(label="input", value="To understand generative AI, one must study [MASK] well.", lines=2)
                clear, submit = action_buttons()
            with gr.Column(scale=1, min_width=420):
                output = gr.Textbox(label="output", lines=6)
        submit.click(fill_mask, text, output)
        clear.click(clear_text, outputs=[text, output])

    with gr.Tab("NER"):
        with input_output_row():
            with gr.Column(scale=1, min_width=420):
                text = gr.Textbox(
                    label="input",
                    value="I am Nate, a research assistant in Izmir Institute of Technology, and currently living and working in beautiful city Izmir in Turkiye.",
                    lines=2,
                )
                clear, submit = action_buttons()
            with gr.Column(scale=1, min_width=420):
                output = gr.Textbox(label="output", lines=5)
        submit.click(ner, text, output)
        clear.click(clear_text, outputs=[text, output])

    with gr.Tab("Question answering"):
        with input_output_row():
            with gr.Column(scale=1, min_width=420):
                question = gr.Textbox(label="input", value="Where does Nate work?", lines=1)
                context = gr.Textbox(
                    label="context",
                    value="I am Nate, a research assistant in Izmir Institute of Technology, and currently living and working in beautiful city Izmir in Turkiye.",
                    lines=2,
                )
                clear, submit = action_buttons()
            with gr.Column(scale=1, min_width=420):
                output = gr.Textbox(label="output", lines=2)
        submit.click(tasks.question_answering, [question, context], output)
        clear.click(clear_two_inputs, outputs=[question, context, output])

    with gr.Tab("Summarization"):
        with input_output_row():
            with gr.Column(scale=1, min_width=420):
                text = gr.Textbox(
                    label="input",
                    value=(
                        "The 2008 Global Financial Crisis was one of the most serious economic "
                        "events of the 21st century. It started in the housing market and quickly "
                        "spread to banks, companies, and governments around the world."
                    ),
                    lines=4,
                )
                clear, submit = action_buttons()
            with gr.Column(scale=1, min_width=420):
                output = gr.Textbox(label="output", lines=4)
        submit.click(tasks.summarization, text, output)
        clear.click(clear_text, outputs=[text, output])

    with gr.Tab("Translation"):
        with input_output_row():
            with gr.Column(scale=1, min_width=420):
                text = gr.Textbox(label="input", value="Artificial intelligence is changing education.", lines=2)
                clear, submit = action_buttons()
            with gr.Column(scale=1, min_width=420):
                output = gr.Textbox(label="output", lines=2)
        submit.click(tasks.translation, text, output)
        clear.click(clear_text, outputs=[text, output])

    with gr.Tab("Image"):
        with input_output_row():
            with gr.Column(scale=1, min_width=420):
                image = gr.Image(label="input", type="filepath", height=160)
                clear, submit = action_buttons()
            with gr.Column(scale=1, min_width=420):
                output = gr.Textbox(label="output", lines=5)
        submit.click(image_classification, image, output)
        clear.click(clear_media, outputs=[image, output])

    with gr.Tab("Speech"):
        with input_output_row():
            with gr.Column(scale=1, min_width=420):
                audio = gr.Audio(label="input", type="filepath")
                clear, submit = action_buttons()
            with gr.Column(scale=1, min_width=420):
                output = gr.Textbox(label="output", lines=3)
        submit.click(tasks.speech_recognition, audio, output)
        clear.click(clear_media, outputs=[audio, output])

if __name__ == "__main__":
    demo.launch(css=css)
