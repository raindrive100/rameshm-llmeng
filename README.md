# rameshm-llmeng

<div align="center">

**A comprehensive LLM engineering toolkit for conversational AI and Retrieval-Augmented Generation (RAG)**

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

[Features](#features) • [Quick Start](#quick-start) • [Architecture](#architecture) • [Deployment](#deployment) • [Contributing](#contributing)

</div>

---

## Overview

`rameshm-llmeng` is a production-ready Python toolkit for building LLM-powered applications. It provides:

- **LLM Chat Application**: A unified interface to interact with multiple LLMs (LLaMA, Gemma, Gemini, Claude, GPT) with a Gradio-based UI
- **RAG Engine**: Retrieval-Augmented Generation system for code generation with comparative analysis across LLM providers
- **Multi-Model Support**: Seamlessly switch between open-source and proprietary models
- **File Processing**: Handle multiple file formats (PDF, Word, Excel, plain text)
- **Deployment Ready**: Docker and Kubernetes configurations included

---

## Features

### 🤖 LLM Chat
- **Multi-Model Support**: Interact with LLaMA, Gemma, Gemini, Claude Sonnet, and GPT models
- **Ephemeral Conversation Context**: Session-based conversation history (no persistent storage for cost efficiency)
- **File Upload & Processing**: Support for multiple document formats
- **User-Friendly UI**: Gradio-based interface for intuitive interaction
- **Extensible Architecture**: Easy to add new LLM providers

### 📚 RAG (Retrieval-Augmented Generation)
- **Code Generation Evaluation**: Comparative analysis of code generation across multiple LLMs
- **Vector Database Integration**: ChromaDB and FAISS support for semantic search
- **Hallucination Detection**: Identify and mitigate LLM hallucinations in generated code
- **Performance Benchmarking**: Compare LLM outputs for quality and robustness

### 🔧 Additional Utilities
- **Webpage Summarization**: Convert web URLs to AI-powered brochures
- **Data Format Conversion**: CSV processing and format conversion utilities
- **Comprehensive Logging**: Built-in logging configuration for debugging
- **Environment Management**: Dotenv support for secure credential management

---

## Quick Start

### Prerequisites
- Python 3.9 or higher
- API keys for LLM services (OpenAI, Google, Anthropic)
- Optional: Docker & Docker Compose or Kubernetes cluster

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/raindrive100/rameshm-llmeng.git
   cd rameshm-llmeng
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_llmchat.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the LLM Chat application**
   ```bash
   python src/rameshm/llmeng/llmchat/gr_ui.py
   ```
   The application will be available at `http://localhost:7860`

---

## Architecture

### Project Structure

```
rameshm-llmeng/
├── src/rameshm/llmeng/
│   ├── llmchat/                 # LLM Chat Application
│   │   ├── gr_ui.py             # Gradio UI (entry point)
│   │   ├── gr_event_handler.py  # Event handling logic
│   │   ├── llm_chat.py          # Core chat logic
│   │   ├── file_handler_llm.py  # File processing (PDF, DOCX, XLSX, etc.)
│   │   └── chat_constants.py    # Configuration constants
│   │
│   ├── rag/                     # RAG Implementation
│   │   ├── rag_example_gemini*.py       # Gemini LLM examples
│   │   ├── rag_example_gpt5Thinking.py  # GPT-5 Thinking examples
│   │   ├── rag_example_sonnet*.py       # Claude Sonnet examples
│   │   ├── rag_with_faiss.py    # FAISS vector database
│   │   └── rag_constants.py     # RAG configuration
│   │
│   ├── exception/               # Custom exceptions
│   │   ├── llm_exception.py
│   │   └── llm_chat_exception.py
│   │
│   ├── utils/                   # Utility modules
│   │   ├── logging_config.py
│   │   ├── init_utils.py
│   │   └── set_environment.py
│   │
│   ├── misc/                    # Miscellaneous utilities
│   │   ├── csv_to_pipe_delimiter_converter.py
│   │   ├── create_image_sounds.py
│   │   └── get_imported_packages.py
│   │
│   ├── webpage_brochure/        # Web scraping & summarization
│   │   ├── website.py
│   │   ├── brochure_from_url.py
│   │   └── llm_website_summary.py
│   │
│   ├── examples/                # Example scripts
│   │   └── gradio_examples.py
│   │
│   └── scratch.py               # Development sandbox
│
├── docker-files/                # Local Docker deployment
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── k8sconfig/                   # Kubernetes deployment
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
│
├── data/                        # Sample data & embeddings
├── chroma_db/                   # ChromaDB vector store
├── requirements_llmchat.txt     # Python dependencies
└── README.md                    # This file
```

### Key Modules

#### **llmchat/gr_ui.py** - Recommended Entry Point
The main UI component. Start here to understand the overall application flow:
```python
python src/rameshm/llmeng/llmchat/gr_ui.py
```

#### **llmchat/gr_event_handler.py**
Handles all user interactions and event logic:
- File upload processing
- Chat message handling
- Model switching
- Conversation management

#### **llmchat/file_handler_llm.py**
File processing factory for multiple formats:
- PDF files (using PyMuPDF)
- Word documents (.docx)
- Excel spreadsheets (.xlsx)
- Plain text files
- Automatic format detection

#### **rag/** - Code Generation Analysis
Comparative analysis of LLM code generation capabilities:
- **rag_example_gemini*.py**: Gemini model evaluation
- **rag_example_gpt5Thinking.py**: GPT-5 reasoning model
- **rag_example_sonnet*.py**: Claude Sonnet performance
- **rag_with_faiss.py**: FAISS-based semantic search
- **rag_example_with_chromadb.py**: ChromaDB integration

---

## Usage

### Running the LLM Chat Application

```bash
# Start the Gradio interface
python src/rameshm/llmeng/llmchat/gr_ui.py
```

Then navigate to `http://localhost:7860` in your browser.

**Features:**
- Select an LLM model from the dropdown
- Upload documents (PDF, Word, Excel)
- Type your query
- Get responses with context from uploaded files
- Conversation history persists for the session

### Using RAG for Code Generation

```bash
# Example: Evaluate code generation with Gemini
python src/rameshm/llmeng/rag/rag_example_gemini25_with_chromadb.py

# Example: Use GPT-5 Thinking
python src/rameshm/llmeng/rag/rag_example_gpt5Thinking_with_chromadb.py

# Example: Evaluate with Claude Sonnet
python src/rameshm/llmeng/rag/rag_example_sonnet4_with_chromadb.py
```

### Website Summarization

```bash
# Convert a website to an AI-powered brochure
python src/rameshm/llmeng/webpage_brochure/brochure_from_url.py
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI API
OPENAI_API_KEY=your_openai_key

# Google Generative AI
GOOGLE_API_KEY=your_google_key

# Anthropic (Claude)
ANTHROPIC_API_KEY=your_anthropic_key

# Application Settings
LOG_LEVEL=INFO
ENABLE_PERSISTENT_STORAGE=false
MAX_FILE_SIZE_MB=50
```

### LLM Provider Configuration

Edit `src/rameshm/llmeng/llmchat/chat_constants.py`:

```python
AVAILABLE_MODELS = {
    'gpt-4': 'OpenAI GPT-4',
    'claude-sonnet-4': 'Anthropic Claude Sonnet',
    'gemini-pro': 'Google Gemini',
    'llama2': 'Open Source LLaMA 2',
    'gemma': 'Open Source Gemma'
}
```

---

## Deployment

### Local Docker Deployment

1. **Build the image**
   ```bash
   cd docker-files
   docker build -t rameshm-llmeng:latest .
   ```

2. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

3. **Access the application**
   ```
   http://localhost:7860
   ```

### Kubernetes Deployment

1. **Create namespace and secrets**
   ```bash
   kubectl create namespace llmeng
   kubectl create secret generic llm-credentials \
     --from-literal=openai_key=$OPENAI_API_KEY \
     --from-literal=google_key=$GOOGLE_API_KEY \
     -n llmeng
   ```

2. **Deploy to cluster**
   ```bash
   kubectl apply -f k8sconfig/ -n llmeng
   ```

3. **Verify deployment**
   ```bash
   kubectl get pods -n llmeng
   kubectl logs -f deployment/llmeng-app -n llmeng
   ```

4. **Access via port forwarding**
   ```bash
   kubectl port-forward svc/llmeng-service 7860:7860 -n llmeng
   ```

**Note**: Kubernetes configurations are cloud-agnostic and have been successfully tested on Google Cloud. Adapt as needed for other providers.

---

## Supported LLM Providers

| Provider | Models | Status |
|----------|--------|--------|
| **OpenAI** | GPT-4, GPT-3.5 Turbo | ✅ Supported |
| **Google** | Gemini Pro, Gemini 2.5 | ✅ Supported |
| **Anthropic** | Claude Sonnet 4, Claude 3 | ✅ Supported |
| **Open Source** | LLaMA 2, Gemma | ✅ Supported |

---

## Supported File Formats

| Format | Library | Status |
|--------|---------|--------|
| **PDF** | PyMuPDF | ✅ Supported |
| **Word** | python-docx | ✅ Supported |
| **Excel** | openpyxl (via python-pptx) | ✅ Supported |
| **Plain Text** | Built-in | ✅ Supported |
| **PowerPoint** | python-pptx | ✅ Supported |

---

## Dependencies

Key Python packages:

```
Core LLM Integration:
- anthropic==0.52.2
- openai==1.79.0
- google-genai==1.18.0
- langchain==0.3.25
- langchain-anthropic==0.3.15
- langchain-openai==0.3.17
- langchain-google-genai==2.1.5
- langchain-community==0.3.23

UI & File Handling:
- gradio==5.30.0
- python-docx==1.2.0
- python-pptx==1.0.2
- pymupdf==1.26.0
- puremagic==1.29

Utilities:
- python-dotenv==1.1.0
- requests==2.32.3
- pillow==11.1.0
- chardet==4.0.0
```

See `requirements_llmchat.txt` for the complete dependency list.

---

## Important Notes

### Conversation Persistence
- Conversation history is maintained **only for the session duration**
- Persistent storage has been intentionally omitted to:
  - Control costs (avoid long-term storage fees)
  - Minimize complexity (no database dependency)
  - Improve privacy (data not retained after session)

### File Processing
The `file_handler_llm.py` module uses a factory pattern for file handling. Future improvements could include:
- More sophisticated factory pattern implementation
- Additional file format support
- Streaming for large files

### RAG Implementation Notes
- Multiple LLMs are evaluated with identical instructions
- LLMs produce robust code structures but require human refinement
- Hallucinations must be identified and mitigated manually
- Different models show varying levels of code quality

---

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_llmchat.py -v

# Run with coverage
python -m pytest tests/ --cov=src
```

### Logging Configuration

Logging is configured in:
- `src/rameshm/llmeng/utils/logging_config.py`
- `src/rameshm/llmeng/rag/logging_config.py`

Set `LOG_LEVEL` in `.env` to control verbosity (DEBUG, INFO, WARNING, ERROR).

### Contributing Locally

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and test thoroughly
4. Commit with clear messages: `git commit -m "Add feature: description"`
5. Push to your fork: `git push origin feature/your-feature`
6. Create a Pull Request

---

## Troubleshooting

### Issue: API Key Not Found
**Solution**: Ensure all required API keys are set in `.env` file and the file is in the project root.

### Issue: Gradio Port Already in Use
**Solution**: Specify a different port:
```bash
python src/rameshm/llmeng/llmchat/gr_ui.py --server_port 7861
```

### Issue: File Upload Fails
**Solution**: 
- Check file format is supported (PDF, DOCX, XLSX, TXT)
- Verify file size is under configured limit (default: 50MB)
- Check file is not corrupted

### Issue: RAG Vector Search Returns No Results
**Solution**:
- Ensure ChromaDB/FAISS index is properly initialized
- Verify documents were successfully embedded
- Try rebuilding the vector index

---

## Performance & Optimization

### Recommended Settings for Production

```python
# chat_constants.py
MAX_CONTEXT_LENGTH = 8192  # Adjust based on model
BATCH_SIZE = 32
EMBEDDING_DIMENSION = 1536
VECTOR_DB_SIMILARITY_THRESHOLD = 0.7
```

### Caching Strategies
- Embed frequently used documents once
- Cache LLM responses for common queries
- Use connection pooling for API calls

---

## Security Considerations

- ✅ API keys stored in `.env` (never commit this file)
- ✅ Input validation on file uploads
- ✅ Rate limiting recommended for production
- ✅ HTTPS required for production deployments
- ✅ Consider implementing authentication for multi-user scenarios

---

## Roadmap

- [ ] Persistent conversation storage (optional)
- [ ] Multi-user support with authentication
- [ ] Advanced RAG with reranking
- [ ] Custom fine-tuned model support
- [ ] Streaming responses
- [ ] Batch processing API
- [ ] Analytics dashboard
- [ ] Advanced caching layer

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact & Support

For questions, issues, or suggestions:
- 📧 Open an issue on GitHub
- 🐛 Report bugs with detailed reproduction steps
- 💡 Share feature requests and ideas

---

## Acknowledgments

This project leverages the following amazing libraries and services:
- [LangChain](https://python.langchain.com/) for LLM orchestration
- [Gradio](https://www.gradio.app/) for the UI framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [OpenAI](https://openai.com/), [Google AI](https://ai.google.dev/), and [Anthropic](https://www.anthropic.com/) for LLM APIs

---

**Made with ❤️ by Ramesh M**

<div align="center">

⭐ If this project helped you, please consider giving it a star!

</div>
