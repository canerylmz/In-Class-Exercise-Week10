import argparse
from pathlib import Path
import warnings

from transformers import pipeline
from transformers.utils import logging as hf_logging


hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)


def print_section(title):
    print(f"\n{title}")
    print("-" * len(title))


def percent(score):
    return f"{score * 100:5.1f}%"


def print_ranked(results, label_key="label", score_key="score", limit=None):
    for i, result in enumerate(results[:limit], 1):
        print(f"{i}. {result[label_key]:<32} {percent(result[score_key])}")


def run_sentiment():
    print_section("1) Sentiment Analysis")
    sentiment = pipeline("sentiment-analysis")

    sentences = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate EE471 course",
    ]

    for sentence, result in zip(sentences, sentiment(sentences)):
        print(f'Text: "{sentence}"')
        print(f"Prediction: {result['label']} ({percent(result['score'])})")
        print()


def run_zero_shot():
    print_section("2) Zero-shot Classification")
    classifier = pipeline("zero-shot-classification")

    text = "Berkshire keeps their cash reserves at an extremely high level."
    labels = ["finance", "education", "technology", "health"]

    result = classifier(text, candidate_labels=labels)
    print(f'Text: "{text}"')
    print("Labels:")
    print_ranked(
        [
            {"label": label, "score": score}
            for label, score in zip(result["labels"], result["scores"])
        ]
    )


def run_generation():
    print_section("3) Text Generation")
    generator = pipeline("text-generation", model="gpt2")

    prompt = "If I continue to successfully complete all in-class exercises in EE471 course,"
    outputs = generator(
        prompt,
        max_new_tokens=35,
        num_return_sequences=2,
        do_sample=True,
    )

    for i, output in enumerate(outputs, 1):
        print(f"Alternative {i}:")
        print(output["generated_text"])
        print()


def run_fill_mask():
    print_section("4) Fill Mask")
    fill_mask = pipeline("fill-mask", model="bert-base-uncased")

    mask_text = "To understand generative AI, one must study [MASK] well."
    print(f'Text: "{mask_text}"')
    print("Top predictions:")
    for i, result in enumerate(fill_mask(mask_text), 1):
        print(f"{i}. {result['token_str']:<12} {percent(result['score'])}  {result['sequence']}")


def run_ner():
    print_section("5) Named Entity Recognition")
    ner = pipeline("ner", aggregation_strategy="simple")

    ner_text = (
        "I am Nate, a research assistant in Izmir Institute of Technology, "
        "and currently living and working in beautiful city Izmir in Turkiye."
    )

    print(f'Text: "{ner_text}"')
    print("Entities:")
    for entity in ner(ner_text):
        print(
            f"- {entity['word']:<32} {entity['entity_group']:<4} "
            f"{percent(float(entity['score']))}"
        )


def run_question_answering():
    print_section("6) Question Answering")
    qa = pipeline("question-answering")

    qa_context = (
        "I am Nate, a research assistant in Izmir Institute of Technology, "
        "and currently living and working in beautiful city Izmir in Turkiye."
    )

    questions = [
        "What is the name of the person?",
        "Which company or organization does he work in?",
        "Where does he live?",
    ]

    for question in questions:
        result = qa(question=question, context=qa_context)
        print(f"Q: {question}")
        print(f"A: {result['answer']} ({percent(result['score'])})")
        print()


def run_summarization():
    print_section("7) Summarization")
    summarizer = pipeline("summarization")

    long_text = """
The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century,
often compared to the Great Depression of the 1930s. Triggered by the bursting of the United States
housing bubble, its effects rippled across the globe, leading to the collapse of major financial
institutions and a deep international recession. The crisis began with the subprime mortgage market.
In the early 2000s, low interest rates and a push for homeownership led banks to issue high-risk
loans to borrowers with poor credit.
"""

    result = summarizer(long_text, max_length=50, min_length=20, do_sample=False)
    print(result[0]["summary_text"].strip())


def run_translation():
    print_section("8) Translation")
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-tr")

    translation_text = (
        "The 2008 Global Financial Crisis stands as the most severe economic collapse "
        "of the 21st century, often compared to the Great Depression."
    )

    result = translator(translation_text)
    print(f"English: {translation_text}")
    print(f"Turkish: {result[0]['translation_text']}")


def run_image_classification(image_path):
    print_section("9) Image Classification")
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"{image_path} not found. Provide an image with --image path\\to\\image.jpg.")
        return

    image_classifier = pipeline(
        "image-classification",
        model="google/vit-base-patch16-224",
    )
    print(f"Image: {image_path}")
    print("Top predictions:")
    print_ranked(image_classifier(str(image_path)), limit=5)


def run_speech_recognition(audio_path):
    print_section("10) Automatic Speech Recognition")
    audio_path = Path(audio_path)

    if not audio_path.exists():
        print(f"{audio_path} not found. Provide an audio file with --audio path\\to\\audio.wav.")
        return

    speech_recognizer = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
    )

    import librosa

    audio, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)
    result = speech_recognizer({"array": audio, "sampling_rate": sampling_rate})
    print(f"Audio: {audio_path}")
    print(f"Transcription: {result['text'].strip()}")


TEXT_TASKS = {
    "sentiment": run_sentiment,
    "zero-shot": run_zero_shot,
    "generation": run_generation,
    "fill-mask": run_fill_mask,
    "ner": run_ner,
    "qa": run_question_answering,
    "summarization": run_summarization,
    "translation": run_translation,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run EE471 Week 10 Transformers tasks.")
    parser.add_argument(
        "--task",
        choices=[*TEXT_TASKS.keys(), "image", "asr"],
        help="Run only one task.",
    )
    parser.add_argument("--skip-text", action="store_true", help="Skip text tasks.")
    parser.add_argument("--image", default="image.jpg", help="Image path for ViT classification.")
    parser.add_argument("--audio", default="audio.wav", help="Audio path for Whisper ASR.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.task in TEXT_TASKS:
        TEXT_TASKS[args.task]()
        return

    if args.task == "image":
        run_image_classification(args.image)
        return

    if args.task == "asr":
        run_speech_recognition(args.audio)
        return

    if not args.skip_text:
        for task in TEXT_TASKS.values():
            task()

    run_image_classification(args.image)
    run_speech_recognition(args.audio)


if __name__ == "__main__":
    main()
