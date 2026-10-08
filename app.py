import json
import joblib
import streamlit as st
from preprocess import clean_text

st.set_page_config(page_title="Bangla News Classifier", page_icon="📰")

BN = {"sports": "খেলা", "national": "জাতীয়",
      "international": "আন্তর্জাতিক", "entertainment": "বিনোদন"}

EXAMPLES = {
    "খেলা": "বাংলাদেশ ক্রিকেট দল আজ টি-টোয়েন্টি ম্যাচে ভারতকে হারিয়েছে",
    "জাতীয়": "রাজধানীর মিরপুরে সকালে সড়ক দুর্ঘটনায় দুইজন নিহত হয়েছেন",
    "আন্তর্জাতিক": "ইউক্রেনে রুশ হামলায় নিহত অন্তত ১০ জন, জাতিসংঘের নিন্দা",
    "বিনোদন": "নতুন সিনেমার শুটিং শুরু করছেন জনপ্রিয় নায়িকা",
}


@st.cache_resource
def load():
    model = joblib.load("models/news_model.joblib")
    with open("models/metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    return model, metrics


model, metrics = load()

st.title("📰 Bangla News Classifier")
st.caption("Bangla news headline ba news likhle model bolbe ta kon category-r: "
           "খেলা, জাতীয়, আন্তর্জাতিক, বিনোদন.")

if "text" not in st.session_state:
    st.session_state.text = ""

cols = st.columns(len(EXAMPLES))
for col, (name, example) in zip(cols, EXAMPLES.items()):
    if col.button(f"Example: {name}"):
        st.session_state.text = example

text = st.text_area("Bangla news text:", key="text", height=140)

if st.button("Classify", type="primary"):
    cleaned = clean_text(text)
    if len(cleaned.split()) < 3:
        st.warning("Kombesh 3 ta Bangla shobdo dao.")
    else:
        probs = model.predict_proba([cleaned])[0]
        order = probs.argsort()[::-1]
        top = model.classes_[order[0]]
        st.subheader(f"Category: {BN.get(top, top)} ({top})")
        for i in order:
            st.progress(float(probs[i]), text=f"{BN.get(model.classes_[i], model.classes_[i])}: {probs[i]:.1%}")
        if probs[order[0]] < 0.6:
            st.info("Model ekdom sure noy. Text ta ekto boro korle bhalo result pabe.")

with st.expander("Model info"):
    r = metrics["results"][metrics["best_model"]]
    st.write(f"**Model:** TF-IDF + {metrics['best_model']}  \n"
             f"**Test accuracy:** {r['accuracy']:.2%}  |  **Macro-F1:** {r['macro_f1']:.2%}")
    st.write("**Top words per category (model ja dekhe shikheche):**")
    for cls, words in metrics["top_words"].items():
        st.write(f"- {BN.get(cls, cls)}: {', '.join(words[:6])}")
