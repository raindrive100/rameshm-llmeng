# rameshm-llmeng

This repository contains several modules for implementing **LLM Chat** and **RAG (Retrieval-Augmented Generation)**.

---

## Major Modules

### **1. LLM Chat**
A chat application that lets you interact with both open-source and closed-source LLMs, including **LLaMA**, **Gemma**, **Gemini**, **Sonnet**, and **GPT**.

- **Conversation Persistence:**  
  Conversation history is maintained only for the session duration.  
  Persistent storage has been intentionally omitted to control costs and minimize complexity.
- **Technology Stack:**  
  Implemented in **Python** with a **Gradio**-based UI.

---

### **2. RAG**
Demonstrates the effectiveness of LLMs in **code generation**.

- The same set of structured instructions were given to multiple LLMs to compare their code-generation capabilities.  
- **Observation:**  
  LLMs produce robust code structures, but human expertise is required to eliminate hallucinations and refine the implementation.
- **Included Materials:**  
  - LLM-generated code samples  
  - Author’s own implementations for comparison  
- The detailed description can be found in the repository’s `README.md`.

---

## Directory Hierarchy

### **docker-files/**
- Deployment scripts for **local Docker** deployment.  
- For cloud deployment, use the `k8sconfig` directory instead.

---

### **k8sconfig/**
> Deployment scripts for **Kubernetes**.

> - Successfully used for deploying the application on **Google Cloud**.  
> - Scripts are **cloud-agnostic**, suitable for adaptation to other providers.  
> - Uses images stored in **Google Cloud Artifact Registry**.

---

### **src/rameshm/llmeng/**
> Contains Python source code modules.

#### **Subdirectories:**

##### **exception/**
> - Exception class definitions.

##### **llmchat/**
> - Core implementation of the **LLM Chat Application**.  
> - Allows users to choose between different LLMs for conversational tasks.  
> - Implements an **ephemeral context cache** (non-persistent).

**Key Modules:**
> - `Chat_constants.py` – Contains shared constants.  
> - `File_handler_llm.py` – File handlers for text, PDF, Word, Excel, etc.  
>  *(A factory pattern could improve this design, but it’s sufficient for this exercise.)*  
> - `Gr_event_handler.py` – All event-handling logic for the Gradio app.  
> - `Gr_ui.py` – The Gradio user interface.  
  *Recommended entry point to understand overall application flow.*

##### **misc/**
> - Miscellaneous utility modules used for experimentation.  
> - Each module is self-explanatory.

##### **rag/**
> RAG implementation that evaluates the **code-generation performance** of:
> - **Gemini**
> - **GPT-5**
> - **Claude Sonnet 4.0**

	*For detailed information, refer to the `README.md` file under this directory.*

---

✅ *End of Document.*
