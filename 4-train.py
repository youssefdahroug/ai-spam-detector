"""Day 2: SMS Spam Detector - training script.

Trains a TF-IDF + Multinomial Naive Bayes classifier on the UCI SMS Spam
Collection and saves the pipeline to model.joblib.
"""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_FILE = "SMSSpamCollection"
MODEL_FILE = "model.joblib"


def load_data(path=DATA_FILE):
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], encoding="utf-8")
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def main():
    df = load_data()
    print(f"Loaded {len(df)} messages: {df.label.value_counts()[0]} ham, {df.label.value_counts()[1]} spam")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
    )
    print(f"Train: {len(X_train)} messages | Test: {len(X_test)} messages")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("nb", MultinomialNB(alpha=0.1)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nClassification report (0=ham, 1=spam):")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"], digits=3))
    print("Confusion matrix [[ham->ham, ham->spam],[spam->ham, spam->spam]]:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
