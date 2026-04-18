import joblib

model = joblib.load("models/optimal_intent_model.pkl")
vectorizer = model["vectorizer"]
clf = model["classifier"]

test_queries = [
    "What does the latest research say about Vitamin D and bone health?",
    "Summarize the mechanism of action of mRNA vaccines.",
    "What are the clinical trial results for this new drug?",
    "I have a headache and nausea, what is wrong with me?",
    "What dosage of ibuprofen should I take?",
    "Compare the outcomes of two surgical approaches in recent studies.",
    "What does this paper conclude about cardiovascular risk?",
    "Can you diagnose my condition?",
    "Is it safe to mix these two medications?",
    "Explain the methodology used in the referenced meta-analysis.",
]

print(f"{'Query':<70} {'Prediction'}")
print("=" * 90)
for q in test_queries:
    X = vectorizer.transform([q])
    pred = clf.predict(X)[0]
    print(f"{q:<70} {pred}")
