import json
import time
import joblib
import os
import random
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

# ============================================================
# 1. LARGE Synthetic Dataset — Carefully curated per class
# ============================================================
print("Generating comprehensive synthetic dataset...")

research_queries = [
    "What are the secondary endpoints of the phase 3 Pfizer trial?",
    "Summarize the mechanism of action of mRNA delivery vaccines.",
    "What is the statistical significance of Vitamin D in bone density studies?",
    "Can you provide the p-value for the control group in the Johns Hopkins study?",
    "What does the latest literature say about CRISPR gene editing for sickle cell?",
    "How do ACE inhibitors operate on a molecular level according to recent papers?",
    "Explain the methodology used in this study to assess cardiovascular risk.",
    "What were the primary conclusions of the meta-analysis on generic fluoxetine?",
    "What is the efficacy rate of the new Moderna booster in clinical trials?",
    "Describe the pharmacokinetics of remdesivir based on published research.",
    "Compare the outcomes of laparoscopic vs open surgery in colorectal cancer studies.",
    "What biomarkers are associated with early detection of Alzheimer's disease?",
    "Summarize recent findings on gut microbiome and mental health.",
    "What does peer-reviewed research say about intermittent fasting and longevity?",
    "Explain the role of T-cells in immune response according to immunology literature.",
    "What are the findings of the WHO report on air pollution and respiratory disease?",
    "How effective is cognitive behavioral therapy for anxiety according to meta-analyses?",
    "What is the current understanding of long COVID based on recent publications?",
    "How does the BRCA1 gene mutation increase cancer risk in published studies?",
    "What are the latest advances in CAR-T cell therapy for leukemia?",
    "Summarize the research on statins and cardiovascular mortality reduction.",
    "What does the literature say about vaccine efficacy in immunocompromised patients?",
    "Explain the relationship between smoking and lung cancer based on epidemiological data.",
    "What are the clinical trial results for psilocybin in treatment-resistant depression?",
    "How does metformin affect insulin sensitivity according to endocrinology research?",
    "What is the evidence for using probiotics in irritable bowel syndrome?",
    "Discuss the relationship between sleep deprivation and cognitive decline in research.",
    "Summarize research on maternal nutrition and fetal brain development.",
    "What are the known risk factors for Type 2 diabetes in recent epidemiological studies?",
    "What does current research suggest about the role of inflammation in depression?",
    "Explain how monoclonal antibodies work against COVID-19 based on clinical data.",
    "What is the research consensus on Mediterranean diet and heart disease?",
    "How does radiation therapy affect tumor regression in glioblastoma studies?",
    "What are the published side effects profiles of checkpoint inhibitors?",
    "Describe the latest gene therapy approaches for muscular dystrophy.",
    "What is the mechanism of action described in this paper?",
    "What clinical evidence supports the use of this compound?",
    "Compare the study designs used in these two randomized controlled trials.",
    "What population was studied in the referenced cohort analysis?",
    "Please summarize this document for me.",
    "How were the participants selected in this randomized trial?",
    "What was the sample size and duration of the referenced study?",
    "What do the results of this research suggest about bone density?",
    "What conclusions can be drawn from the data in this paper?",
    "Explain the significance of Figure 3 in the referenced document.",
]

diagnosis_queries = [
    "What does this rash on my arm mean?",
    "I have a severe headache and nausea, what is happening to me?",
    "Can you look at my blood test and tell me if I have diabetes?",
    "Is this mole cancerous?",
    "I feel dizzy and my chest hurts, do I have a heart attack?",
    "My child has a fever of 104, what is wrong?",
    "I have been coughing blood, what disease do I have?",
    "Do I have COVID based on these symptoms?",
    "Can you diagnose my condition from these lab results?",
    "I have sharp abdominal pain, is it appendicitis?",
    "My eyes are turning yellow, do I have jaundice?",
    "I have numbness in my left arm, am I having a stroke?",
    "What is causing my chronic fatigue?",
    "Could my back pain be a sign of kidney disease?",
    "I feel short of breath after climbing stairs, is my heart okay?",
    "My joints are swollen and painful, do I have arthritis?",
    "Why do I keep getting headaches every day?",
    "Am I depressed or is this just stress?",
    "Is this lump in my neck something serious?",
    "My blood sugar was 250 this morning, what does that mean for me?",
    "I have a persistent sore throat, could it be throat cancer?",
    "What is wrong with me based on my symptoms?",
    "Do these symptoms indicate multiple sclerosis?",
    "Diagnose me based on these test results.",
    "My stomach hurts after eating, what condition do I have?",
    "I think I might have an autoimmune disease, can you confirm?",
    "What is causing the ringing in my ears?",
    "I have been losing weight rapidly, is it cancer?",
    "My heart rate is irregular, should I be worried?",
    "Tell me what disease matches these symptoms.",
]

drug_queries = [
    "What is the proper dosage of ibuprofen for a 10 year old?",
    "Can I mix Advil with Lexapro safely?",
    "Where can I buy Ozempic online?",
    "Is it safe to drink alcohol while taking antibiotics?",
    "How much paracetamol is safe to take in one day?",
    "Prescribe me something for a sore throat.",
    "What should I take for this chronic knee pain?",
    "What is the cure for my insomnia?",
    "Should I take aspirin daily for heart health?",
    "What medicine should I take for my migraine?",
    "Can I take melatonin with my blood pressure medication?",
    "What is the recommended dose of amoxicillin for adults?",
    "Is it safe to take Tylenol during pregnancy?",
    "What over-the-counter drug is best for allergies?",
    "Can I stop taking my antidepressant cold turkey?",
    "What happens if I take too much Xanax?",
    "Should I switch from brand name to generic medication?",
    "What drug interactions should I worry about with warfarin?",
    "How long should I take this antibiotic course?",
    "Can I take two different pain relievers at the same time?",
    "What supplements should I take for iron deficiency?",
    "Is metformin safe long-term?",
    "What is the best over the counter sleep aid?",
    "Can I crush my medication tablets?",
    "What are the withdrawal symptoms if I stop this drug?",
]

# Build labeled dataset
data = []
for q in research_queries:
    data.append((q, "Research"))
for q in diagnosis_queries:
    data.append((q, "Diagnosis"))
for q in drug_queries:
    data.append((q, "Drug-related"))

# Light augmentation: lowercase variants and minor rephrasings
augmented = []
for q, label in data:
    augmented.append((q, label))
    augmented.append((q.lower(), label))
    # Add a "please" variant
    augmented.append(("Please " + q[0].lower() + q[1:], label))

random.seed(42)
random.shuffle(augmented)

X = [item[0] for item in augmented]
y = [item[1] for item in augmented]

print(f"Total training samples: {len(X)}")
print(f"  Research: {y.count('Research')}, Diagnosis: {y.count('Diagnosis')}, Drug-related: {y.count('Drug-related')}")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============================================================
# 2. TF-IDF Vectorizer (much better than HashingVectorizer for small datasets)
# ============================================================
print("\nFitting TfidfVectorizer...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

classes = np.array(["Diagnosis", "Drug-related", "Research"])
clf = SGDClassifier(loss='modified_huber', random_state=42, max_iter=1, tol=None)

# ============================================================
# 3. Epoch Loop with Checkpointing
# ============================================================
EPOCHS = 10
start_time = time.time()
best_accuracy = 0.0

for epoch in range(1, EPOCHS + 1):
    clf.partial_fit(X_train_vec, y_train, classes=classes)
    
    # Evaluate on test set each epoch
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    
    # Save Checkpoint
    checkpoint_path = f"models/checkpoint_epoch_{epoch}.pkl"
    joblib.dump({"vectorizer": vectorizer, "classifier": clf}, checkpoint_path)
    print(f"Epoch {epoch}/{EPOCHS} — Test Accuracy: {acc:.2%} — Saved {checkpoint_path}")
    
    if acc > best_accuracy:
        best_accuracy = acc

end_time = time.time()
training_time_minutes = (end_time - start_time) / 60.0

# Final evaluation
y_pred_final = clf.predict(X_test_vec)
final_accuracy = accuracy_score(y_test, y_pred_final)

print(f"\n{'='*50}")
print(f"Training Complete in {training_time_minutes:.4f} minutes.")
print(f"Final Test Accuracy: {final_accuracy:.2%}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_final))

# ============================================================
# 4. Save Final Model + Metadata
# ============================================================
best_model_path = "models/optimal_intent_model.pkl"
joblib.dump({"vectorizer": vectorizer, "classifier": clf}, best_model_path)

metadata = {
    "training_time_minutes": round(training_time_minutes, 4),
    "final_accuracy": round(final_accuracy, 4),
    "best_accuracy": round(best_accuracy, 4),
    "epochs_trained": EPOCHS,
    "samples_trained_on": len(X_train),
    "test_samples": len(X_test),
    "best_model_path": best_model_path
}

with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\nModel saved to: {best_model_path}")
print(f"Metadata saved to: models/model_metadata.json")
print("Ready for production!")
