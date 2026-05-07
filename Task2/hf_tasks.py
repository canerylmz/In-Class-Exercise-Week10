import librosa
from transformers import pipeline


class HuggingFaceTaskModule:
    def __init__(self):
        self.pipes = {}

    def get_pipe(self, name, task, model=None, **kwargs):
        if name not in self.pipes:
            if model is None:
                self.pipes[name] = pipeline(task, **kwargs)
            else:
                self.pipes[name] = pipeline(task, model=model, **kwargs)
        return self.pipes[name]

    def sentiment_analysis(self, text):
        pipe = self.get_pipe("sentiment", "sentiment-analysis")
        result = pipe(text)[0]
        return f"{result['label']} - {result['score']:.2f}"

    def zero_shot_classification(self, text, labels):
        pipe = self.get_pipe("zero-shot", "zero-shot-classification")
        labels = [label.strip() for label in labels.split(",")]
        result = pipe(text, candidate_labels=labels)

        rows = []
        for label, score in zip(result["labels"], result["scores"]):
            rows.append([label, f"{score:.2f}"])
        return rows

    def text_generation(self, prompt):
        pipe = self.get_pipe("generation", "text-generation", model="gpt2")
        result = pipe(prompt, max_new_tokens=40, num_return_sequences=1)
        return result[0]["generated_text"]

    def fill_mask(self, text):
        pipe = self.get_pipe("fill-mask", "fill-mask", model="bert-base-uncased")
        results = pipe(text)

        rows = []
        for result in results:
            rows.append([result["token_str"], f"{result['score']:.2f}", result["sequence"]])
        return rows

    def named_entity_recognition(self, text):
        pipe = self.get_pipe("ner", "ner", aggregation_strategy="simple")
        results = pipe(text)

        rows = []
        for result in results:
            rows.append([result["word"], result["entity_group"], f"{result['score']:.2f}"])
        return rows

    def question_answering(self, question, context):
        pipe = self.get_pipe("qa", "question-answering")
        result = pipe(question=question, context=context)
        return f"{result['answer']} - {result['score']:.2f}"

    def summarization(self, text):
        pipe = self.get_pipe("summarization", "summarization")
        result = pipe(text, max_length=70, min_length=30, do_sample=False)
        return result[0]["summary_text"]

    def translation(self, text):
        pipe = self.get_pipe(
            "translation",
            "translation",
            model="Helsinki-NLP/opus-mt-tc-big-en-tr",
        )
        result = pipe(text)
        return result[0]["translation_text"]

    def image_classification(self, image_path):
        if image_path is None:
            return []

        pipe = self.get_pipe(
            "image",
            "image-classification",
            model="google/vit-base-patch16-224",
        )
        results = pipe(image_path)

        rows = []
        for result in results:
            rows.append([result["label"], f"{result['score']:.2f}"])
        return rows

    def speech_recognition(self, audio_path):
        if audio_path is None:
            return ""

        pipe = self.get_pipe(
            "asr",
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
        )

        audio, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)
        result = pipe({"array": audio, "sampling_rate": sampling_rate})
        return result["text"]
