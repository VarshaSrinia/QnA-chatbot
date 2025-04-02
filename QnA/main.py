import streamlit as st
from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API with your API key
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(api_key=openai_api_key, temperature=0.1, model_name='gpt-3.5-turbo')

# Define paths with raw string notation for Windows
vectordb_file_path = r"C:\Users\Darshan\Desktop\Gen AI\langchain-main\Hands-on LLM\QnA\faiss_index"
csv_path = r"C:\Users\Darshan\Desktop\Gen AI\langchain-main\Hands-on LLM\QnA\codebasics_faqs.csv"

# Initialize embeddings model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Function to create the vector database
def create_vector_db():
    try:
        # Load data from FAQ sheet
        loader = CSVLoader(file_path=csv_path, source_column="prompt")
        data = loader.load()

        # Create a FAISS instance for vector database from 'data'
        vectordb = FAISS.from_documents(documents=data, embedding=embeddings)

        # Save vector database locally
        vectordb.save_local(vectordb_file_path)
        st.success("FAISS index created successfully.")
        st.session_state["index_created"] = True  # Set session state for index creation
    except Exception as e:
        st.error(f"Failed to create FAISS index: {e}")
        st.session_state["index_created"] = False

# Function to get the Q&A chain
def get_qa_chain():
    try:
        vectordb = FAISS.load_local(vectordb_file_path, embeddings=embeddings)
        retriever = vectordb.as_retriever(score_threshold=0.7)

        # Define prompt template
        prompt_template = """Given the following context and a question, generate an answer based on this context only.
        Try to use as much text as possible from the "response" section in the context without major changes.
        If the answer is not in the context, respond with "I don't know."

        CONTEXT: {context}

        QUESTION: {question}"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Initialize the QA chain with the prompt
        chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            retriever=retriever,
                                            input_key="query",
                                            return_source_documents=True,
                                            chain_type_kwargs={"prompt": PROMPT})

        return chain
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return None

# Streamlit app
def main():
    st.title("FAQ Question Answering 🌱")

    # Initialize session state for index creation
    if "index_created" not in st.session_state:
        st.session_state["index_created"] = os.path.exists(f"{vectordb_file_path}.index")

    # Create Knowledgebase button
    btn = st.button("Create Knowledgebase")

    # Handle button click to create knowledge base
    if btn and not st.session_state["index_created"]:
        with st.spinner("Creating vector database..."):
            create_vector_db()

    # Check if the index was created or loaded successfully
    if st.session_state["index_created"]:
        # Load the Q&A chain
        chain = get_qa_chain()

        # If chain is successfully created, show question input
        if chain:
            question = st.text_input("Enter your question:")

            # Process the question when input is given
            if question:
                with st.spinner("Fetching answer..."):
                    response = chain({"query": question})
                    answer = response['result']

                    # Display the answer
                    st.header("Answer")
                    st.write(answer)
        else:
            st.warning("The FAISS index failed to load. Please recreate the knowledge base.")
    else:
        st.info("Please create the knowledgebase to enable question-answering functionality.")

if __name__ == "__main__":
    main()
