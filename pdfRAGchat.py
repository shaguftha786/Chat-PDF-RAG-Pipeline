import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Load environment variables
load_dotenv()

# Initialize embeddings
embedding_model = SpacyEmbeddings(model_name="en_core_web_sm")

def extract_text_from_pdfs(pdf_files):
    """Extract text from uploaded PDF files."""
    extracted_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted_text += page.extract_text()
    return extracted_text

def split_text_into_chunks(extracted_text):
    """Split text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(extracted_text)

def create_vector_database(text_chunks):
    """Create a FAISS vector store from text chunks."""
    vector_db = FAISS.from_texts(text_chunks, embedding=embedding_model)
    vector_db.save_local("vector_store")

def initiate_conversation_tool(tools, query):
    """Initialize the conversational tool using Anthropic API and process the query."""
    llm = ChatAnthropic(
        model="claude-3-sonnet-20240229",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        verbose=True
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an intelligent assistant. Use the provided context to answer the query as comprehensively as possible. If the context does not contain the answer, respond with 'The answer is not available in the context.' Avoid making assumptions."""
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, [tools], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[tools], verbose=True)
    response = agent_executor.invoke({"input": query})

    st.write("Response:", response['output'])

def process_user_query(user_query):
    """Process the user's query using the retriever and LLM tool."""
    vector_db = FAISS.load_local("vector_store", embedding_model, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever()

    retrieval_tool = create_retriever_tool(
        retriever,
        name="pdf_query_tool",
        description="This tool retrieves answers based on PDF content."
    )

    initiate_conversation_tool(retrieval_tool, user_query)

def main():
    """Main application logic."""
    st.set_page_config(page_title="PDF Query Chat")
    st.header("RAG-based PDF Conversational Assistant")

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("Options")
        pdf_files = st.file_uploader(
            "Upload PDF files and click the process button:", 
            accept_multiple_files=True
        )
        if st.button("Process PDFs"):
            with st.spinner("Processing the uploaded PDFs..."):
                extracted_text = extract_text_from_pdfs(pdf_files)
                text_chunks = split_text_into_chunks(extracted_text)
                create_vector_database(text_chunks)
                st.success("PDFs have been processed successfully!")

    # Main query input
    user_query = st.text_input("Ask a question about the uploaded PDFs:")
    if user_query:
        process_user_query(user_query)

if __name__ == "__main__":
    main()
