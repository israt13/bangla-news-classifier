"""Bangla News Category Classifier - training script.

Usage:
    python train.py --data data/Bangla_news.csv

Dataset e `title`, `content`, `category` column lagbe.
"""
import argparse
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import clean_text


def tfidf():
    # token_pattern=\S+ : text already clean + space-separated. Default \w pattern
    # Bangla matra (vowel sign) e shabdo bhenge dey, tai eta use kora hoyeche.
    return TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2),
                           min_df=2, max_df=0.9, max_features=150_000, sublinear_tf=True)


def build_models():
    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.1))]),
        "Logistic Regression": Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=2000, C=10))]),
        "Linear SVM": Pipeline([("tfidf", tfidf()), ("clf", CalibratedClassifierCV(LinearSVC(C=1), cv=3))]),
    }


def top_words(model, n=10):
    """Logistic Regression theke prottek category-r sobcheye important shabdo."""
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    vocab = vec.get_feature_names_out()
    return {cls: [vocab[i] for i in clf.coef_[k].argsort()[-n:][::-1]]
            for k, cls in enumerate(clf.classes_)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Bangla_news.csv")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    n_raw = len(df)
    df = df.dropna(subset=["title", "content", "category"])
    # Title + content ek sathe use kora hocche
    df["text"] = (df["title"].astype(str) + " " + df["content"].astype(str)).apply(clean_text)
    df = df[df["text"].str.len() > 20]
    df = df.drop_duplicates(subset="text")  # duplicate article thakle train/test leak hoy
    print(f"Raw rows: {n_raw} -> after cleaning/dedup: {len(df)}")
    print(df["category"].value_counts(), "\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["category"], test_size=args.test_size,
        random_state=42, stratify=df["category"])

    results, fitted = {}, {}
    for name, model in build_models().items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        results[name] = {"accuracy": round(accuracy_score(y_te, pred), 4),
                         "macro_f1": round(f1_score(y_te, pred, average="macro"), 4)}
        fitted[name] = model
        print(f"{name:20s} accuracy={results[name]['accuracy']:.4f}  macro-F1={results[name]['macro_f1']:.4f}")

    best = max(results, key=lambda k: results[k]["macro_f1"])
    model = fitted[best]
    pred = model.predict(X_te)
    print(f"\nBest model: {best}\n")
    print(classification_report(y_te, pred, digits=3))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax, cmap="Blues", xticks_rotation=30)
    ax.set_title(f"Confusion Matrix - {best}")
    fig.savefig("models/confusion_matrix.png", dpi=150, bbox_inches="tight")

    joblib.dump(model, "models/news_model.joblib", compress=3)

    summary = {"best_model": best, "n_train": len(X_tr), "n_test": len(X_te),
               "results": results, "classes": list(model.classes_),
               "top_words": top_words(fitted["Logistic Regression"])}
    with open("models/metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Saved: models/news_model.joblib, models/metrics.json, models/confusion_matrix.png")


if __name__ == "__main__":
    main()
