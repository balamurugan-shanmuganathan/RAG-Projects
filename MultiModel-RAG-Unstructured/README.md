# 🌟 Multi-Modal RAG: Chat with Your Documents (Text, Tables, & Images!)

![MultiModal RAG Demo](demo.gif)

Welcome! This project is a beginner-friendly guide to building a **Multi-Modal Retrieval-Augmented Generation (RAG)** system. 

If that sounds like a mouthful, don't worry! It basically means building an AI that can "read" your documents and answer questions about them—even if the info is hidden inside a complex table or a picture.

---

## 🧐 What is "Multi-Modal RAG"?

Most AI systems can only read plain text. But real-world documents (like PDFs) have:
*   📊 **Tables** with important numbers.
*   📸 **Images** like charts, graphs, or diagrams.
*   🏗️ **Structure** like headers and sub-headers.

**Multi-Modal RAG** allows the AI to "see" all these elements together, giving you much smarter and more accurate answers!

---

## 🛠️ The Secret Sauce: Unstructured.io

In this project, we use a library called **Unstructured**. Think of it as a smart "document sorter." 

Instead of just grabbing all the text as one big mess, Unstructured:
1.  **Identifies** what is a title, what is a paragraph, and what is a table.
2.  **Extracts** tables as clean code (HTML) that the AI can understand perfectly.
3.  **Saves** images so the AI can "look" at them when answering your questions.

---

## 🚀 How It Works (The Simple Version)

1.  **Upload**: You give the app a PDF.
2.  **Slicing & Dicing**: Unstructured breaks the PDF into pieces (Text, Tables, Images).
3.  **Summarizing**: We use **Google Gemini** (a very smart AI) to look at each piece and write a tiny summary of what's in it.
4.  **Asking**: When you ask a question, the app finds the most relevant "pieces" and shows them to the AI.
5.  **Answering**: The AI gives you an answer based on both the text and the visual evidence!

---

## 🚦 Getting Started

Follow these steps to run the project on your own computer:

### 1. Prerequisites
*   **Python 3.10+** installed.
*   **Gemini API Key**: Get one for free from [Google AI Studio](https://aistudio.google.com/).
*   **System Tools**: You might need `poppler-utils` and `tesseract-ocr` installed on your computer for PDF processing.

### 2. Setup
Clone this folder and install the tools needed:

```bash
# Get the code
git clone https://github.com/balamurugan-shanmuganathan/RAG-Projects.git
cd MultiModel-RAG-Unstructured

# Install the dependencies
pip install -r requirements.txt
```

### 3. Configure
Open `src/utils/config.py` and add your Gemini API key:

```python
API_KEY = "YOUR_GEMINI_API_KEY"
```

### 4. Run the App!
Start the server and open your browser:

```bash
# Start the backend
uvicorn main:app --reload
```

Then, open `ui/index.html` in your browser to start chatting!

---

## 💡 What You'll Learn
*   How to handle "messy" PDFs that normal tools fail at.
*   How to combine text and images to get better AI answers.
*   How to use **ChromaDB** to store and find information quickly.

---
*Created with ❤️ to help beginners dive into the world of Modern AI.*
