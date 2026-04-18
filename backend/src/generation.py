import os
import yaml
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

import joblib

class GenerationPipeline:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, streaming=True)
        self.system_prompt = config["prompts"]["system_prompt"]
        self.refusal_message = config["safety"]["refusal_message"]
        
        # Load the custom ML Intent Classifier
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "optimal_intent_model.pkl")
        if os.path.exists(model_path):
            model_bundle = joblib.load(model_path)
            self.clf_vectorizer = model_bundle["vectorizer"]
            self.clf_classifier = model_bundle["classifier"]
            self.has_classifier = True
        else:
            self.has_classifier = False
            print("WARNING: ML Classifier not found. Safety guardrails are disabled.")
        
    def check_safety(self, query):
        """Returns tuple: (is_safe: bool, predicted_class: str)"""
        if not self.has_classifier:
             return True, "Research" # default
             
        X = self.clf_vectorizer.transform([query])
        pred = self.clf_classifier.predict(X)[0]
        
        if pred in ["Diagnosis", "Drug-related"]:
             return False, pred
             
        return True, pred

    def format_context(self, chunks):
        context_str = ""
        for i, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            context_str += f"--- Document [{i+1}] ---\n"
            context_str += f"Title: {meta.get('title', 'Unknown')}\n"
            context_str += f"Authors: {meta.get('authors', 'Unknown')} ({meta.get('year', 'Unknown')})\n"
            context_str += f"Content: {chunk['text']}\n\n"
        return context_str

    def stream_answer(self, query, retrieved_chunks):
        # 1. ML Safety Classification Check
        is_safe, pred_intent = self.check_safety(query)
        self.latest_intent = pred_intent  # Stored for Data Science logging in api.py
        
        if not is_safe:
            yield f"[Classified as {pred_intent}] " + self.refusal_message
            return
            
        # 2. Check Context
        if not retrieved_chunks:
            yield "No relevant medical research could be found to answer this question. Please refine your query."
            return
            
        # 3. Build Prompt
        context = self.format_context(retrieved_chunks)
        
        full_prompt = (
            f"Context:\n{context}\n\n"
            f"User Question: {query}\n\n"
            f"Answer based ONLY on the context. If you cannot answer it, say you don't know based on the context. "
            f"Cite the documents using [1], [2], etc."
        )
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=full_prompt)
        ]
        
        # 4. Stream Tokens
        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content

