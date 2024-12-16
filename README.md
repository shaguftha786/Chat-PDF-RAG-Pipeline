# Chat-PDF-RAG-Pipeline

This project facilitates conversational interaction with PDF documents by leveraging the Retrieval-Augmented Generation (RAG) pipeline. Users can ask questions related to the content of uploaded PDF files, and the system provides precise and informative responses generated through RAG-based techniques.

## Features

- **PDF Upload**: Enables users to upload single or multiple PDF files containing the information they want to query.
  
- **Text Extraction**: Efficiently extracts text content from uploaded PDF files for further processing.

- **Text Chunking**: Breaks down extracted text into smaller, manageable segments for effective processing and retrieval.

- **Vector Store Creation**:Uses FAISS to create a high-performance vector store, ensuring fast and accurate retrieval of relevant information.
  
- **Conversational Interface**: Integrates the RAG model to generate detailed, context-aware responses in a user-friendly conversational format.

  ## Tech Stack
- **Python**: Programming language for implementing core functionalities.
- **Streamlit**: Framework for building an interactive and dynamic web-based user interface.
- **PyPDF2**: Library for extracting and processing text from PDF files.
- **Langchain**: Framework for constructing and implementing the RAG pipeline with language models.
- **FAISS**: Toolkit for fast similarity searches and efficient vector-based information retrieval.
- **Anthropic**:  API powering conversational AI responses using the Claude-3 model.

  ## Installation

To run this project locally, follow the steps below:

1. Clone the repository:

```bash
git clone https://github.com/your-repo-name/RAG-Chat-PDF.git  
```

2. Install  required dependencies:

```bash
pip install -r requirements.txt  
```

3. Set up environment variables:

   - Obtain your Anthropic API key and add it to your environment variables as `ANTHROPIC_API_KEY`.
   
   - Ensure a `.env` file exists in the project directory, containing the necessary keys for proper configuration.

4. Launch the Streamlit application:

```bash
streamlit run app.py  
```

## Usage

1. Upload one or more PDF files that contain the information you need.
2. Click the "Process Documents" button to extract and analyze the content.
3. Enter your query in the text box, and the system will generate a contextually accurate response based on the uploaded documents.


## License

This project is licensed under the MIT License. For more information, refer to the LICENSE file.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [PyPDF2](https://github.com/mstamy2/PyPDF2)
- [langchain](https://github.com/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Anthropic](https://anthropic.com/)
- [dotenv](https://github.com/theskumar/python-dotenv)
