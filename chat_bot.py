import streamlit as st
import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import HuggingFaceHub


# Constants
DB_FAISS_PATH = "vectorstore/db_FAISS"
HUGGINGFACE_REPO_ID = "microsoft/DialoGPT-small"

# Get HF_TOKEN from secrets or environment variable
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except (st.errors.StreamlitSecretNotFoundError, KeyError):
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        st.error("❌ HF_TOKEN not found! Please set it in .streamlit/secrets.toml or as an environment variable.")
        st.stop()

# Load FAISS vector store
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Custom prompt template
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["question", "context"])
    return prompt

# Load Hugging Face LLM
def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceHub(
        repo_id=huggingface_repo_id,
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512},
        huggingfacehub_api_token=HF_TOKEN
    )
    return llm

# Streamlit UI
def main():
    st.title("🩺 Riya Medical Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask your medical question...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Answer the medical question based on the provided context.

        Context: {context}
        Question: {question}

        Answer:
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("❌ Failed to load the vector store.")
                return

            # Try to load LLM and create chain
            try:
                llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )
                response = qa_chain.invoke({'query': prompt})
                answer = response.get("result", "🤖 Sorry, I couldn't find an answer.")
                
            except Exception as llm_error:
                # Fallback: Use simple retrieval without LLM
                st.warning("⚠️ LLM unavailable, using simple retrieval...")
                docs = vectorstore.similarity_search(prompt, k=3)
                if docs:
                    answer = f"📚 **Retrieved Information:**\n\n{docs[0].page_content[:500]}..."
                    if len(docs) > 1:
                        answer += f"\n\n**Additional sources:** {len(docs)-1} more documents found."
                else:
                    answer = "🤖 Sorry, I couldn't find relevant information in the knowledge base."

        except Exception as e:
            answer = f"⚠️ Error: {str(e)}"

        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()

