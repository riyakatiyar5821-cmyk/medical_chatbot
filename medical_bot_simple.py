import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

load_dotenv()

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create a simple text generation pipeline
print("Loading model...")
pipe = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=150,
    temperature=0.7,
    repetition_penalty=1.2,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=pipe)

# Simple prompt template
CUSTOM_PROMPT_TEMPLATE = """
Answer the medical question based on the provided context.

Context: {context}
Question: {question}

Answer:"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

print("Medical Bot is ready!")
print("Ask a medical question (type 'quit' to exit):")

while True:
    user_query = input("\nYour question: ").strip()
    
    if user_query.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if not user_query:
        continue
        
    try:
        response = qa_chain.invoke({'query': user_query})
        print(f"\nAnswer: {response['result']}")
        
        # Show source documents
        print(f"\nSources:")
        for i, doc in enumerate(response['source_documents'], 1):
            print(f"{i}. Page {doc.metadata.get('page_label', 'Unknown')} from {doc.metadata.get('source', 'Unknown')}")
            print(f"   Content: {doc.page_content[:200]}...")
            print()
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please try again with a different question.")
