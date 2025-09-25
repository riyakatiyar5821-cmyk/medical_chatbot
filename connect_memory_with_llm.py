import os

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from dotenv import load_dotenv
load_dotenv()
import os
# ...existing code...




# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")

# Try different models in order of preference (local models)
MODEL_OPTIONS = [
    "google/flan-t5-base",  # Better instruction following
    "microsoft/DialoGPT-medium",  # Better conversational model
    "distilgpt2"  # Fallback
]

HUGGINGFACE_REPO_ID = MODEL_OPTIONS[0]  # Start with the first option

def load_llm(model_id):
    try:
        print(f"Loading local model: {model_id}")
        
        # Handle T5 models differently
        if "flan-t5" in model_id.lower():
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            
            tokenizer = T5Tokenizer.from_pretrained(model_id)
            model = T5ForConditionalGeneration.from_pretrained(model_id)
            
            # Create pipeline for text2text generation
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.5,
                do_sample=True
            )
        else:
            # Load tokenizer and model for causal LM
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            
            # Create pipeline with better stopping criteria
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=200,  # Reduced to prevent repetition
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2  # Add repetition penalty
            )
        
        # Create HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=pipe)
        print(f"Successfully loaded model: {model_id}")
        return llm
        
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        # Try a simpler fallback
        try:
            print("Trying fallback with distilgpt2...")
            pipe = pipeline(
                "text-generation", 
                model="distilgpt2", 
                max_new_tokens=200, 
                temperature=0.7,
                repetition_penalty=1.2
            )
            llm = HuggingFacePipeline(pipeline=pipe)
            print("Successfully loaded fallback model: distilgpt2")
            return llm
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            raise Exception("Failed to load any model. Please check your internet connection and try again.")

print("HF_TOKEN:", HF_TOKEN)

# from langchain_huggingface import HuggingFaceEndpoint

# HF_TOKEN = os.environ.get("HF_TOKEN")  # Or hardcode for testing

# llm = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
#     temperature=0.5,
#     max_new_tokens=512,
#     huggingfacehub_api_token=HF_TOKEN  # ✅ Correct way to pass token
# )

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Based on the medical information provided below, answer the question accurately and concisely.

Context: {context}

Question: {question}

Answer:"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])