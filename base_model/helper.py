from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def load_model():
    return TFAutoModelForSeq2SeqLM.from_pretrained("base_model/", local_files_only = True), AutoTokenizer.from_pretrained("t5-small")

def predict_summary(model, tokenizer, news):
    summarizer = pipeline("summarization", model = model, tokenizer = tokenizer, framework = "tf")
    summary = summarizer(news, min_length = 10, max_length = 150)
    return summary
