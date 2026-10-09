from transformers import pipeline
import pandas as pd
import os

# Create dataset folder
os.makedirs("dataset", exist_ok=True)

comparison_data = []


# ==================================================
# 1. SENTIMENT ANALYSIS
# ==================================================

sentiment_analyser = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

user_input = input(
    "Enter a statement for sentiment analysis: "
)

result = sentiment_analyser(user_input)

label = result[0]["label"]
score = result[0]["score"]

sentiment_df = pd.DataFrame([{
    "Statement": user_input,
    "Sentiment": label,
    "Confidence": score
}])

sentiment_df.to_csv(
    "dataset/sentiment_analysis.csv",
    index=False
)

print("\nSentiment:", label)
print("Confidence:", score)

comparison_data.append({
    "Task": "Sentiment Analysis",
    "Input": user_input,
    "Output": label,
    "Confidence": score
})


# ==================================================
# 2. TRANSLATION
# ==================================================

translator = pipeline(
    "translation_en_to_fr",
    model="Helsinki-NLP/opus-mt-en-fr"
)

translation_input = input(
    "\nEnter an English sentence for translation: "
)

result = translator(translation_input)

translated_text = result[0]["translation_text"]

translation_df = pd.DataFrame([{
    "Original_Text": translation_input,
    "Translated_Text": translated_text
}])

translation_df.to_csv(
    "dataset/translation.csv",
    index=False
)

print("\nTranslated:", translated_text)

comparison_data.append({
    "Task": "Translation",
    "Input": translation_input,
    "Output": translated_text,
    "Confidence": ""
})


# ==================================================
# 3. TEXT SUMMARIZATION
# ==================================================

summariser = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

summary_input = input(
    "\nEnter a paragraph for summarization: "
)

result = summariser(
    summary_input,
    max_length=80,
    min_length=10,
    do_sample=False
)

summary = result[0]["summary_text"]

summary_df = pd.DataFrame([{
    "Original_Text": summary_input,
    "Summary": summary
}])

summary_df.to_csv(
    "dataset/summarization.csv",
    index=False
)

print("\nSummary:", summary)

comparison_data.append({
    "Task": "Text Summarization",
    "Input": summary_input,
    "Output": summary,
    "Confidence": ""
})


# ==================================================
# 4. QUESTION ANSWERING
# ==================================================

question_answerer = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

context = input(
    "\nEnter a context for question answering: "
)

question = input(
    "Enter your question: "
)

result = question_answerer(
    question=question,
    context=context
)

answer = result["answer"]
qa_score = result["score"]

qa_df = pd.DataFrame([{
    "Context": context,
    "Question": question,
    "Answer": answer,
    "Confidence": qa_score
}])

qa_df.to_csv(
    "dataset/question_answer.csv",
    index=False
)

print("\nAnswer:", answer)
print("Confidence:", qa_score)

comparison_data.append({
    "Task": "Question Answering",
    "Input": question,
    "Output": answer,
    "Confidence": qa_score
})


# ==================================================
# 5. PIPELINE COMPARISON CSV
# ==================================================

comparison_df = pd.DataFrame(comparison_data)

comparison_df.to_csv(
    "dataset/pipeline_comparison.csv",
    index=False
)


# ==================================================
# COMPLETION MESSAGE
# ==================================================

print("\n======================================")
print("ALL NLP TASKS COMPLETED SUCCESSFULLY!")
print("======================================")

print("\nCreated CSV files:")
print("1. sentiment_analysis.csv")
print("2. translation.csv")
print("3. summarization.csv")
print("4. question_answer.csv")
print("5. pipeline_comparison.csv")