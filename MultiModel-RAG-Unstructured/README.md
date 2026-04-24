# MultiModel-RAG: Layout-Aware Document Intelligence

A professional implementation of a **Multi-Modal Retrieval-Augmented Generation (RAG)** pipeline utilizing **Unstructured.io** for layout-aware partitioning and **Gemini 1.5 Pro** for semantic synthesis.

## 🧠 Core Philosophy: Beyond OCR

Standard RAG pipelines often treat PDFs as flat text streams, losing critical context embedded in **tables**, **charts**, and **embedded images**. This project demonstrates a sophisticated architectural pattern that treats a document as a collection of structured elements rather than mere strings.

### The Problem

Traditional text extraction (OCR or basic PDF parsing) strips away:

- **Spatial context**: The relationship between a table and its descriptive text.
- **Visual evidence**: Diagrams or charts that contain data not present in text.
- **Tabular structure**: Complex nested tables often become garbled text.

### The Solution: Unstructured.io

This application leverages the **Unstructured** library to perform **Partitioning**. It breaks down the document into high-fidelity elements (Title, NarrativeText, Table, Image, etc.), allowing the system to:

1. **Extract Tables as HTML**: Preserves structural integrity for LLM reasoning.
2. **Extract Images as Base64**: Enables multi-modal vision analysis.
3. **Cross-Reference Chunks**: Groups textual narrative with nearby visual/tabular evidence.

---

## 🏗️ Architecture & Processing Pipeline

The pipeline follows a four-stage process: **Partitioning → Synthesis → Vectorization → Contextual Retrieval**.

### 1. Unified Partitioning (via Unstructured)

The system uses the `unstructured` library to decompose PDFs. Unlike native PDF parsers, it performs **Layout Analysis** to detect headers, subheaders, and specifically, `Table` and `Image` objects.

### 2. Multi-Modal Synthesis

For every chunk of text that contains associated tables or images, the system creates an **AI-Enhanced Summary**:

- **Inputs**: Raw Text + Table HTML + Image (Base64).
- **Process**: A Multi-Modal LLM analyzes all inputs simultaneously.
- **Output**: A comprehensive semantic description that internalizes numbers from tables and patterns from images into a searchable text block.

### 3. Rich Metadata Storage

Each vectorized chunk in **ChromaDB** carries a payload of `original_content` in its metadata, which includes:

- `raw_text`: The narrative content.
- `tables_html`: The structural table data.
- `images_base64`: Visual evidence for multi-modal verification.

### 4. Contextual Retrieval

During query time, the system retrieves the most relevant semantic summaries. Crucially, the "Evidence Drawer" doesn't just show text; it reconstructs the original tables and images associated with those chunks, providing the LLM (and the user) with **direct evidence**.

---

## 🛠️ Technical Stack

- **Extraction Engine**: [Unstructured.io](https://unstructured.io/)
- **Orchestration**: Python / FastAPI
- **LLM / Vision**: Google Gemini 3 Pro (Multimodal) / Gemma (Ollama)
- **Vector Store**: ChromaDB (with metadata persistence)
- **Framework**: LangChain (Core primitives)

---

## 🚀 Getting Started (Professionals Only)

### Prerequisites

- Python 3.10+
- `libmagic`, `poppler`, and `tesseract` (Unstructured system dependencies)
- Google AI / Ollama API Keys

### Installation

```bash
git clone https://github.com/your-repo/MultiModel-RAG-Unstructured.git
cd MultiModel-RAG-Unstructured
pip install -r requirements.txt
```

### Configuration

Update `src/utils/config.py` with your environment variables:

```python
API_KEY = "YOUR_GEMINI_API_KEY"
MODEL_NAME = "gemini-3-pro"
```

### Execution

```bash
uvicorn main:app --reload
```

---

## 📘 Key Learnings for Developers

- **Table Reliability**: LLMs reason significantly better on HTML-formatted tables than on Markdown or CSV, especially for merged cells.
- **Multimodal Embedding**: Instead of embedding images directly (which can be noisy), we embed an **AI-generated semantic description** of the image, which aligns better with text-based user queries.
- **Deduplication**: The system implements content-fingerprinting to ensure that overlapping retrieval chunks are deduplicated before they reached the synthesis phase.

---

_Developed as a technical demonstration of modern RAG architectures._
