import json
from langchain_core.messages import HumanMessage
from ..utils.config import gemini_llm, ollama_llm

def get_llm(model_name):
    if model_name == "ollama":
        return ollama_llm
    return gemini_llm

def stream_final_answer(chunks, query, model_name="gemini"):
    """Generator version of generate_final_answer to support streaming."""
    try:
        llm = get_llm(model_name)
        
        # Build the text prompt
        prompt_text = f"""Based on the following documents, please answer this question: {query}
        
Citations: Use [1], [2], etc. to cite your sources.

CONTENT TO ANALYZE:
"""
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"
                tables_html = original_data.get("tables_html", [])
                for j, table in enumerate(tables_html):
                    prompt_text += f"Table {j+1}:\n{table}\n\n"
            prompt_text += "\n"
        
        prompt_text += """
Please provide a clear, comprehensive answer. If the documents don't contain sufficient information, say so.

ANSWER:"""

        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add images
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                for image_base64 in original_data.get("images_base64", []):
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        message = HumanMessage(content=message_content)
        
        # Stream from LLM
        for chunk in llm.stream([message]):
            content = chunk.content
            if isinstance(content, list):
                yield "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
            else:
                yield str(content)
                 
    except Exception as e:
        yield f"❌ System Error during streaming (${model_name}): {str(e)}"

def generate_final_answer(chunks, query, model_name="gemini"):
    """Generate final answer using multimodal content"""
    try:
        llm = get_llm(model_name)
        
        # Build the text prompt
        prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"
            prompt_text += "\n"
        
        prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above. 

ANSWER:"""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add all images from all chunks
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])
                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        # Ensure response content is a string
        content = response.content
        if isinstance(content, list):
            return "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
        return str(content)
        
    except Exception as e:
        print(f"❌ Answer generation failed (${model_name}): {e}")
        return f"Sorry, I encountered an error with the {model_name} architecture."