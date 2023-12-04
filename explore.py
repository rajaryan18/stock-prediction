from transformers import pipeline

specific_model = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

data = ["Shares in Reliance Power fall 17% on their first day of trading in India's biggest stock market flotation."]
print(specific_model(data))