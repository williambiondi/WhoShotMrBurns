from transformers import pipeline


class EmotionDetector:
    def __init__(self, model_name, threshold):
        self.classifier = pipeline("text-classification", model = model_name)
        self.threshold = threshold

    def predict(self, text):
        emotions = self.classifier(text, top_k=None)
        gap = emotions[0]["score"] - emotions[1]["score"]
        if emotions[0]["label"] == "neutral" and gap < self.threshold:
            return emotions[1]
        return emotions[0]

    def get_all_labels(self):
        return [emotion["label"] for emotion in self.classifier("test", top_k=None)]

    def predict_label(self, text):
        return self.predict(text)["label"]

    def predict_score(self, text):
        return self.predict(text)["score"]
