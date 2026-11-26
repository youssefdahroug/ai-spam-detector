"""Day 2: SMS Spam Detector - prediction script.

Usage:
    python predict.py "Congratulations! You won a free iPhone, click here"
"""
import sys

import joblib

MODEL_FILE = "model.joblib"


def main():
    if len(sys.argv) < 2:
        print('Usage: python predict.py "your message here"')
        sys.exit(1)

    message = " ".join(sys.argv[1:])
    model = joblib.load(MODEL_FILE)
    proba = model.predict_proba([message])[0]
    spam_score = proba[1]
    verdict = "SPAM" if spam_score >= 0.5 else "HAM"
    confidence = spam_score if verdict == "SPAM" else proba[0]
    print(f'Message: "{message}"')
    print(f"Verdict: {verdict} (confidence {confidence:.1%})")


if __name__ == "__main__":
    main()
