"""
Module for processing documents using the Unstructured library.
This module provides functions to partition PDFs into atomic elements 
and group them into intelligent chunks.
"""

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from ..utils.config import MAX_CHARACTERS, NEW_AFTER_N_CHARS, COMBINE_TEXT_UNDER_N_CHARS

def partition_document(file_path: str):
    """
    Extract elements from a PDF file using the Unstructured library.

    Args:
        file_path (str): The local path to the PDF document.

    Returns:
        list: A list of Unstructured elements (Table, Image, Text, etc.).
    """
    print(f"📄 Partitioning document: {file_path}")
    
    elements = partition_pdf(
        filename=file_path,  # Path to your PDF file
        strategy="hi_res", # Use the most accurate (but slower) processing method of extraction
        infer_table_structure=True, # Keep tables as structured HTML, not jumbled text
        extract_image_block_types=["Image"], # Grab images found in the PDF
        extract_image_block_to_payload=True # Store images as base64 data you can actually use
    )
    
    print(f"✅ Extracted {len(elements)} elements")
    return elements


def create_chunks_by_title(elements):
    """
    Create intelligent chunks from Unstructured elements using a title-based strategy.

    Args:
        elements (list): A list of Unstructured elements from the partition step.

    Returns:
        list: A list of composite chunks ready for vector storage.
    """
    print("🔨 Creating smart chunks...")
    
    chunks = chunk_by_title(
        elements, # The parsed PDF elements from previous step
        max_characters=MAX_CHARACTERS, # Hard limit - never exceed 3000 characters per chunk
        new_after_n_chars=NEW_AFTER_N_CHARS, # Try to start a new chunk after 2400 characters
        combine_text_under_n_chars=COMBINE_TEXT_UNDER_N_CHARS # Merge tiny chunks under 500 chars with neighbors
    )
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks
