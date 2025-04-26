import os
import sys
import json
import time
import uuid
import base64
import logging
import requests
import subprocess
import threading
from io import BytesIO
from typing import Optional, Sequence, List, Dict, Union
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("multimodal_rag")

try:
    # Core dependencies
    import numpy as np
    import torch
    
    # ChromaDB and embedding 
    import chromadb
    from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
    from chromadb.utils.data_loaders import ImageLoader
    from chromadb.api.types import Document, Embedding, EmbeddingFunction, URI, DataLoader
    from sentence_transformers import SentenceTransformer
    
    # Image and video processing
    import cv2
    
    # LLM integration
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    # PDF processing
    from pdf2image import convert_from_bytes
    import pytesseract
    from PyPDF2 import PdfReader
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.text import partition_text
    from unstructured.partition.json import partition_json
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    raise
    
# Constants
DEFAULT_CHROMA_PATH = "data/rag_chroma_db"
REPO_METADATA_FILE = "data/repo_metadata.json"
CONVERSATIONS_FILE = "data/conversations.json"
API_KEY = ## key##

# Suppress specific warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Initialize global variables
client = None
image_loader = None
CLIP = None
device = None

def initialize_system(chroma_path=DEFAULT_CHROMA_PATH):
    """Initialize the system components"""
    global client, image_loader, CLIP, device
    
    # Create directory if it doesn't exist
    os.makedirs(chroma_path, exist_ok=True)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=chroma_path)
    
    # Initialize Image Loader and CLIP embeddings
    image_loader = ImageLoader()
    CLIP = OpenCLIPEmbeddingFunction()
    
    # Set up device for torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
  
    return {
        "client": client,
        "device": device,
        "status": "initialized"
    }

# PDF and Text Processing Functions
def extract_text_from_pdf(file_bytes):
    """Extract text from a PDF using PyPDF2."""
    try:
        reader = PdfReader(BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def fallback_pdf_ocr(file_bytes):
    """Fallback: convert PDF pages to images and extract text using OCR."""
    try:
        images = convert_from_bytes(file_bytes)
        ocr_text = ""
        for image in images:
            ocr_text += pytesseract.image_to_string(image) + "\n"
        return ocr_text.strip()
    except Exception as e:
        logger.error(f"Error performing OCR on PDF: {e}")
        return ""

def structure_pdf(file_bytes):
    """
    Try to partition the PDF using the Unstructured library.
    Fallback to PyPDF2 extraction (or OCR if necessary).
    """
    try:
        structured = partition_pdf(file=BytesIO(file_bytes))
        structured = [{"type": str(e.category), "text": e.text} for e in structured if e.text]
        if structured:
            return structured
    except Exception as e:
        logger.warning(f"Unstructured partitioning failed: {e}")
    
    raw_text = extract_text_from_pdf(file_bytes)
    if not raw_text:
        logger.info("No text via PyPDF2; falling back to OCR...")
        raw_text = fallback_pdf_ocr(file_bytes)
    
    return [{"type": "text", "text": raw_text}]

def process_pdf_file(file_path):
    """Process a PDF file given its local file path."""
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        return structure_pdf(file_bytes)
    except Exception as e:
        logger.error(f"Error processing PDF file {file_path}: {e}")
        return []

def process_text_file(file_path):
    """
    Process a text-based file (.txt, .md, .json) by reading its content.
    Returns a list with one dictionary.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [{"type": "text", "text": text}]
    except Exception as e:
        logger.error(f"Error processing text file {file_path}: {e}")
        return []

def process_ipynb_file(file_path):
    """
    Process a Jupyter Notebook file (.ipynb).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)
        text_parts = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") in ["markdown", "code"]:
                source = cell.get("source", [])
                if isinstance(source, list):
                    text_parts.append("".join(source))
                else:
                    text_parts.append(source)
        combined_text = "\n".join(text_parts)
        return [{"type": "ipynb", "text": combined_text}]
    except Exception as e:
        logger.error(f"Error processing ipynb file {file_path}: {e}")
        return []

def process_local_repo(repo_path):
    """
    Walk through a local repository directory, processing PDFs, text-based files, and ipynb files.
    Returns three lists: documents (full text), metadatas (dicts with file paths), and ids (file basenames).
    """
    documents = []
    metadatas = []
    ids = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            file_path = os.path.join(root, file)
            if file_ext == ".pdf":
                content = process_pdf_file(file_path)
            elif file_ext in [".txt", ".md", ".json"]:
                content = process_text_file(file_path)
            elif file_ext == ".ipynb":
                content = process_ipynb_file(file_path)
            else:
                continue
            
            if content:
                if isinstance(content, list):
                    combined_text = " ".join([section["text"] for section in content if section["text"]])
                else:
                    combined_text = content
                documents.append(combined_text)
                metadatas.append({"file": file_path})
                ids.append(str(uuid.uuid4()))
    
    return documents, metadatas, ids

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into chunks of approximately 'chunk_size' words with the specified overlap.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = words[start:start+chunk_size]
        chunks.append(" ".join(chunk))
        start += (chunk_size - overlap)
    return chunks

def batch_add_to_collection(collection, documents, embeddings, metadatas, ids, batch_size=100):
    """
    Add data to a collection in batches to avoid memory issues.
    """
    for i in range(0, len(documents), batch_size):
        # Slice the data into batches
        doc_batch = documents[i:i + batch_size]
        emb_batch = embeddings[i:i + batch_size]
        meta_batch = metadatas[i:i + batch_size]
        id_batch = ids[i:i + batch_size]

        # Add the batch to the collection
        collection.add(
            documents=doc_batch,
            embeddings=emb_batch,
            metadatas=meta_batch,
            ids=id_batch
        )
        logger.info(f"Batch {i // batch_size + 1} added to the collection successfully.")

def image_to_base64(image_path):
    """
    Convert an image to base64 encoding for use in LLM prompts.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def format_texts(texts, max_length=800):
    """
    Format text for the prompt
    """
    if not texts:
        return "No relevant text documents found."

    # Combine texts with source tracking
    combined_text = ""
    for i, text in enumerate(texts):
        if len(text) > max_length:
            text = text[:max_length] + "..."

        combined_text += f"Document {i+1}: {text}\n\n"

    return combined_text

# GitHub Repository Functions
def clone_github_repo(github_url):
    """
    Clone a GitHub repository to a temporary directory

    Args:
        github_url (str): URL of the GitHub repository

    Returns:
        str: Path to the cloned repository
    """
    try:
        # Ensure the URL is properly formatted
        if not github_url.startswith(("http://", "https://")):
            # If it's just a username/repo format, add the GitHub URL prefix
            if "/" in github_url and not github_url.startswith("git@"):
                github_url = f"https://github.com/{github_url}"

        # Create a unique directory name for this repository
        repo_name = github_url.split('/')[-1].replace('.git', '')
        unique_id = str(uuid.uuid4())[:8]
        repo_dir = f"/tmp/github_repos/{repo_name}_{unique_id}"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(repo_dir), exist_ok=True)

        # Clone the repository
        logger.info(f"Cloning {github_url} to {repo_dir}...")
        subprocess.run(["git", "clone", github_url, repo_dir], check=True)

        return repo_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Error cloning repository: {str(e)}")
        raise ValueError(f"Failed to clone repository: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

def process_repository(github_url, chromadb_path=DEFAULT_CHROMA_PATH, changed_files=None):
    """
    Process a GitHub repository to extract and embed multimodal data

    Args:
        github_url (str): URL of the GitHub repository
        chromadb_path (str): Path to store ChromaDB data
        changed_files (list, optional): List of specific files to process. If None, all files are processed.

    Returns:
        dict: Summary of processing results
    """
    # Initialize the system if not already initialized
    if client is None:
        initialize_system(chromadb_path)

    # Step 1: Clone the repository
    repo_path = clone_github_repo(github_url)

    logger.info(f"Processing repository: {repo_path}")
    results = {
        "text_files": 0,
        "image_files": 0,
        "video_files": 0,
        "chunks_processed": 0,
        "repo_path": repo_path
    }

    # Generate repository ID
    if "github.com/" in github_url:
        parts = github_url.split("github.com/")[1].split("/")
    else:
        parts = github_url.split("/")

    if len(parts) >= 2:
        owner = parts[0].strip()
        repo_name = parts[1].split('.git')[0].strip()
        # Create a sanitized ID - ensure it meets ChromaDB's requirements
        repo_id = f"{owner}_{repo_name}"
        # Replace any invalid characters
        repo_id = ''.join(c if c.isalnum() or c in ['_', '-', '.'] else '_' for c in repo_id)
        # Ensure it starts and ends with alphanumeric character
        if not repo_id[0].isalnum():
            repo_id = 'r' + repo_id
        if not repo_id[-1].isalnum():
            repo_id = repo_id + 'r'
        # Make sure it's not too long
        if len(repo_id) > 60:
            repo_id = repo_id[:60]
    else:
        # Fallback: Create a simple UUID
        repo_id = f"repo_{uuid.uuid4().hex[:8]}"

    results["repo_id"] = repo_id
    logger.info(f"Using repository ID: {repo_id}")

    
    # Create or get collections for this repository
    try:
        text_collection = client.get_or_create_collection(name=f"text_collection_{repo_id}")
        image_collection = client.get_or_create_collection(
            name=f"image_collection_{repo_id}",
            embedding_function=CLIP,
            data_loader=image_loader
        )
        video_collection = client.get_or_create_collection(
            name=f"video_collection_{repo_id}",
            embedding_function=CLIP,
            data_loader=image_loader
        )
        logger.info("Collections created successfully")
    except Exception as e:
        logger.error(f"Error creating collections: {str(e)}")
        return results

    # Process only specific files if changed_files is provided
    if changed_files and "*" not in changed_files:
        logger.info(f"Processing {len(changed_files)} changed files")

        # Process text files
        try:
            text_docs = []
            text_metas = []
            text_ids = []

            # Process each file individually
            for file_path in changed_files:
                abs_file_path = os.path.join(repo_path, file_path)
                if not os.path.exists(abs_file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue

                file_ext = os.path.splitext(file_path)[1].lower()

                if file_ext == ".pdf":
                    content = process_pdf_file(abs_file_path)
                elif file_ext in [".txt", ".md", ".json"]:
                    content = process_text_file(abs_file_path)
                elif file_ext == ".ipynb":
                    content = process_ipynb_file(abs_file_path)
                elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                    # Process image file
                    if os.path.exists(abs_file_path):
                        relative_path = os.path.relpath(abs_file_path, repo_path)
                        unique_id = relative_path.replace('/', '_').replace('\\', '_')
                        # Remove old embeddings if they exist
                        try:
                            image_collection.delete(ids=[unique_id])
                        except:
                            pass

                        # Add new embedding
                        image_collection.add(
                            ids=[unique_id],
                            uris=[abs_file_path]
                        )
                        results["image_files"] += 1
                    continue
                elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                    # Process video file
                    if os.path.exists(abs_file_path):
                        # Create frames directory
                        video_name = os.path.basename(abs_file_path).split('.')[0]
                        frames_dir = os.path.join(repo_path, "extracted_frames", video_name)
                        if not os.path.exists(frames_dir):
                            os.makedirs(frames_dir)

                        # Extract frames
                        video_capture = cv2.VideoCapture(abs_file_path)
                        fps = video_capture.get(cv2.CAP_PROP_FPS)
                        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

                        frame_ids = []
                        frame_uris = []
                        frame_metadatas = []

                        success, image = video_capture.read()
                        frame_number = 0
                        while success:
                            if frame_number == 0 or frame_number % int(fps * 5) == 0 or frame_number == frame_count - 1:
                                frame_time = frame_number / fps
                                output_frame_filename = os.path.join(frames_dir, f'frame_{int(frame_time)}.jpg')
                                cv2.imwrite(output_frame_filename, image)

                                relative_video_path = os.path.relpath(abs_file_path, repo_path)
                                frame_id = f"frame_{int(frame_time)}_{relative_video_path.replace('/', '_').replace(os.path.sep,'_')}"

                                frame_ids.append(frame_id)
                                frame_uris.append(output_frame_filename)
                                frame_metadatas.append({'video_uri': abs_file_path})

                            success, image = video_capture.read()
                            frame_number += 1

                        video_capture.release()

                        # Remove old frame embeddings if they exist
                        try:
                            # Query to find existing frames for this video
                            existing_frames = video_collection.query(
                                query_embeddings=None,
                                where={"video_uri": abs_file_path},
                                include=["ids"]
                            )
                            if "ids" in existing_frames and existing_frames["ids"]:
                                video_collection.delete(ids=existing_frames["ids"][0])
                        except Exception as e:
                            logger.warning(f"Error removing old video frames: {e}")

                        # Add new frame embeddings
                        if frame_ids:
                            video_collection.add(
                                ids=frame_ids,
                                uris=frame_uris,
                                metadatas=frame_metadatas
                            )
                            results["video_files"] += 1
                    continue
                else:
                    # Skip unsupported file types
                    continue

                # For text files, collect content for batch processing
                if content:
                    if isinstance(content, list):
                        combined_text = " ".join([section["text"] for section in content if section["text"]])
                    else:
                        combined_text = content

                    text_docs.append(combined_text)
                    text_metas.append({"file": abs_file_path})
                    text_ids.append(str(uuid.uuid4()))

            # Process text chunks
            if text_docs:
                model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
                all_chunks = []
                all_chunk_ids = []
                all_chunk_metadatas = []

                for i, (doc, meta, doc_id) in enumerate(zip(text_docs, text_metas, text_ids)):
                    chunks = chunk_text(doc, chunk_size=500, overlap=50)
                    for idx, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        all_chunk_ids.append(f"{doc_id}_{idx}")
                        all_chunk_metadatas.append(meta)

                results["chunks_processed"] = len(all_chunks)
                results["text_files"] = len(text_docs)

                # Generate embeddings and add to collection
                batch_size = 128
                for i in range(0, len(all_chunks), batch_size):
                    batch = all_chunks[i:i+batch_size]
                    batch_ids = all_chunk_ids[i:i+batch_size]
                    batch_metadatas = all_chunk_metadatas[i:i+batch_size]

                    # Remove old embeddings if they exist
                    try:
                        text_collection.delete(ids=batch_ids)
                    except Exception as e:
                        logger.warning(f"Error removing old text chunks: {e}")

                    batch_embeddings = model.encode(batch, convert_to_tensor=True, device=device)
                    batch_embeddings = [emb.tolist() for emb in batch_embeddings.cpu().numpy()]

                    text_collection.add(
                        documents=batch,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )

        except Exception as e:
            logger.error(f"Error processing text data: {str(e)}")

    else:
        # Process the entire repository
        # Process text files
        try:
            documents, metadatas, ids = process_local_repo(repo_path)
            logger.info(f"Found {len(documents)} text documents")
            results["text_files"] = len(documents)

            # Process text chunks
            model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
            all_chunks = []
            all_chunk_ids = []
            all_chunk_metadatas = []

            for i, (doc, meta, doc_id) in enumerate(zip(documents, metadatas, ids)):
                chunks = chunk_text(doc, chunk_size=500, overlap=50)
                for idx, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_chunk_ids.append(f"{doc_id}_{idx}")
                    all_chunk_metadatas.append(meta)
                if i % 10 == 0:
                    logger.info(f"Processed {i}/{len(documents)} documents")

            results["chunks_processed"] = len(all_chunks)
            logger.info(f"Created {len(all_chunks)} text chunks")

            # Generate embeddings and add to collection
            batch_size = 128
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i+batch_size]
                batch_ids = all_chunk_ids[i:i+batch_size]
                batch_metadatas = all_chunk_metadatas[i:i+batch_size]

                batch_embeddings = model.encode(batch, convert_to_tensor=True, device=device)
                batch_embeddings = [emb.tolist() for emb in batch_embeddings.cpu().numpy()]

                text_collection.add(
                    documents=batch,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                if i % (batch_size * 5) == 0:
                    logger.info(f"Added {i}/{len(all_chunks)} chunks to text collection")

            logger.info("Text embeddings added to collection")
        except Exception as e:
            logger.error(f"Error processing text data: {str(e)}")

        # Process image files
        try:
            image_ids = []
            image_uris = []

            for root, dirs, files in os.walk(repo_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, repo_path)
                        unique_id = relative_path.replace('/', '_').replace('\\', '_')
                        image_ids.append(unique_id)
                        image_uris.append(file_path)

            results["image_files"] = len(image_ids)
            logger.info(f"Found {len(image_ids)} images")

            if image_ids:
                # Add in batches
                batch_size = 100
                for i in range(0, len(image_ids), batch_size):
                    batch_ids = image_ids[i:i+batch_size]
                    batch_uris = image_uris[i:i+batch_size]

                    image_collection.add(
                        ids=batch_ids,
                        uris=batch_uris
                    )
                    logger.info(f"Added {i+len(batch_ids)}/{len(image_ids)} images")

        except Exception as e:
            logger.error(f"Error processing images: {str(e)}")

        # Process video files
        try:
            # Find all video files
            video_paths = []
            for root, dirs, files in os.walk(repo_path):
                for file in files:
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                        video_paths.append(os.path.join(root, file))

            results["video_files"] = len(video_paths)
            logger.info(f"Found {len(video_paths)} videos")

            if video_paths:
                # Create frames directory
                frames_dir = os.path.join(repo_path, "extracted_frames")
                if not os.path.exists(frames_dir):
                    os.makedirs(frames_dir)

                # Extract frames from each video
                for video_path in video_paths:
                    video_name = os.path.basename(video_path).split('.')[0]
                    output_subfolder = os.path.join(frames_dir, video_name)

                    if not os.path.exists(output_subfolder):
                        os.makedirs(output_subfolder)

                    # Extract frames
                    video_capture = cv2.VideoCapture(video_path)
                    fps = video_capture.get(cv2.CAP_PROP_FPS)
                    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

                    success, image = video_capture.read()
                    frame_number = 0
                    while success:
                        if frame_number == 0 or frame_number % int(fps * 5) == 0 or frame_number == frame_count - 1:
                            frame_time = frame_number / fps
                            output_frame_filename = os.path.join(output_subfolder, f'frame_{int(frame_time)}.jpg')
                            cv2.imwrite(output_frame_filename, image)

                        success, image = video_capture.read()
                        frame_number += 1

                    video_capture.release()

                # Add frames to video collection
                ids = []
                uris = []
                metadatas = []

                for video_path in video_paths:
                    video_name = os.path.basename(video_path).split('.')[0]
                    frame_folder = os.path.join(frames_dir, video_name)

                    if os.path.exists(frame_folder):
                        for frame in os.listdir(frame_folder):
                            if frame.endswith('.jpg'):
                                relative_video_path = os.path.relpath(video_path, repo_path)
                                frame_id = f"{frame[:-4]}_{relative_video_path.replace('/', '_').replace(os.path.sep,'_')}"
                                frame_path = os.path.join(frame_folder, frame)
                                ids.append(frame_id)
                                uris.append(frame_path)
                                metadatas.append({'video_uri': video_path})

                # Add frames in batches
                if ids:
                    batch_size = 100
                    for i in range(0, len(ids), batch_size):
                        end_idx = min(i + batch_size, len(ids))
                        batch_ids = ids[i:end_idx]
                        batch_uris = uris[i:end_idx]
                        batch_metadatas = metadatas[i:end_idx]

                        video_collection.add(
                            ids=batch_ids,
                            uris=batch_uris,
                            metadatas=batch_metadatas
                        )
        except Exception as e:
            logger.error(f"Error processing videos: {str(e)}")

    return {
        **results,
        "repo_id": repo_id,
        "collections": {
            "text": f"text_collection_{repo_id}",
            "image": f"image_collection_{repo_id}",
            "video": f"video_collection_{repo_id}"
        }
    }

# Retrieval Functions
def get_collections(repo_id):
    """
    Get the ChromaDB collections for a specific repository

    Args:
        repo_id (str): Repository identifier

    Returns:
        tuple: (text_collection, image_collection, video_collection)
    """
    # Initialize if not already initialized
    if client is None:
        initialize_system()
        
    text_collection = client.get_collection(name=f"text_collection_{repo_id}")
    image_collection = client.get_collection(name=f"image_collection_{repo_id}")
    video_collection = client.get_collection(name=f"video_collection_{repo_id}")

    return text_collection, image_collection, video_collection

def text_uris_for_repo(query_text, repo_id, max_distance=None, max_results=5):
    """
    Retrieve text documents for a specific repository

    Args:
        query_text (str): Query text
        repo_id (str): Repository identifier
        max_distance (float, optional): Maximum distance threshold
        max_results (int, optional): Maximum number of results

    Returns:
        list: Filtered texts
    """
    text_collection, _, _ = get_collections(repo_id)

    results = text_collection.query(
        query_texts=[query_text],
        n_results=max_results,
        include=['documents', 'distances']
    )

    filtered_texts = []
    for doc, distance in zip(results['documents'][0], results['distances'][0]):
        if max_distance is None or distance <= max_distance:
            filtered_texts.append(doc)

    return filtered_texts

def image_uris_for_repo(query_text, repo_id, max_distance=None, max_results=5):
    """
    Retrieve image URIs for a specific repository

    Args:
        query_text (str): Query text
        repo_id (str): Repository identifier
        max_distance (float, optional): Maximum distance threshold
        max_results (int, optional): Maximum number of results

    Returns:
        list: Filtered image URIs
    """
    _, image_collection, _ = get_collections(repo_id)

    results = image_collection.query(
        query_texts=[query_text],
        n_results=max_results,
        include=['uris', 'distances']
    )

    filtered_uris = []
    for uri, distance in zip(results['uris'][0], results['distances'][0]):
        if max_distance is None or distance <= max_distance:
            filtered_uris.append(uri)

    return filtered_uris

def frame_uris_for_repo(query_text, repo_id, max_distance=None, max_results=5):
    """
    Retrieve video frame URIs for a specific repository

    Args:
        query_text (str): Query text
        repo_id (str): Repository identifier
        max_distance (float, optional): Maximum distance threshold
        max_results (int, optional): Maximum number of results

    Returns:
        list: Filtered video frame URIs
    """
    _, _, video_collection = get_collections(repo_id)

    results = video_collection.query(
        query_texts=[query_text],
        n_results=max_results,
        include=['uris', 'distances']
    )

    filtered_uris = []
    seen_folders = set()

    for uri, distance in zip(results['uris'][0], results['distances'][0]):
        if max_distance is None or distance <= max_distance:
            folder = os.path.dirname(uri)
            if folder not in seen_folders:
                filtered_uris.append(uri)
                seen_folders.add(folder)

        if len(filtered_uris) == max_results:
            break

    return filtered_uris

def multimodal_retrieve_for_repo(query, repo_id, text_threshold=2.0, image_threshold=2.0, video_threshold=2.0):
    """
    Unified retrieval function that gets content from all modalities in a specific repository
    and provides source tracking information.

    Args:
        query (str): Query text
        repo_id (str): Repository identifier
        text_threshold (float, optional): Maximum distance threshold for text
        image_threshold (float, optional): Maximum distance threshold for images
        video_threshold (float, optional): Maximum distance threshold for videos

    Returns:
        dict: Retrieved content and source information
    """
    # Retrieve from each collection
    texts = text_uris_for_repo(query, repo_id, max_distance=text_threshold, max_results=3)
    images = image_uris_for_repo(query, repo_id, max_distance=image_threshold, max_results=2)
    video_frames = frame_uris_for_repo(query, repo_id, max_distance=video_threshold, max_results=1)

    # Create source tracking information
    sources = {
        "text_sources": [{"content": t[:100] + "...", "similarity": "high"} for t in texts],
        "image_sources": [{"uri": img, "filename": os.path.basename(img)} for img in images],
        "video_sources": [{"frame": frm, "video": os.path.dirname(frm).split('/')[-1]} for frm in video_frames]
    }

    return {
        "texts": texts,
        "images": images,
        "video_frames": video_frames,
        "sources": sources
    }

# LLM Integration
def build_dynamic_prompt(query, texts, image_data_1=None, image_data_2=None):
    """
    Build a dynamic prompt based on available content

    Args:
        query (str): The user's query
        texts (str): The text content
        image_data_1 (str, optional): Base64-encoded image data
        image_data_2 (str, optional): Base64-encoded video frame data

    Returns:
        ChatPromptTemplate: A dynamically constructed prompt
    """
    # Define the system message
    system_message = "You are a multimodal retrieval assistant. You are provided with extracted text, images, and video frame data relevant to the query {query}. Synthesize and explain the key information from these modalities to provide a concise and clear answer."
    
    # Start with the system message
    messages = [("system", system_message)]

    # Build user message components
    user_components = [
        {
            "type": "text",
            "text": f"Query: {query}\n\nAvailable text information:\n{texts}"
        }
    ]

    # Add first image if available
    if image_data_1:
        user_components.append({
            "type": "text",
            "text": "The following image is relevant to your query:"
        })
        user_components.append({
            "type": "image_url",
            "image_url": {'url': f"data:image/jpeg;base64,{image_data_1}"}
        })

    # Add video frame if available
    if image_data_2:
        user_components.append({
            "type": "text",
            "text": "This is a frame from a video that may be relevant:"
        })
        user_components.append({
            "type": "image_url",
            "image_url": {'url': f"data:image/jpeg;base64,{image_data_2}"}
        })

    # Add closing instruction
    user_components.append({
        "type": "text",
        "text": "Based on the above information, please provide a comprehensive answer to the query."
    })

    # Add the user message with components
    messages.append(("user", user_components))

    # Create and return the prompt template
    return ChatPromptTemplate.from_messages(messages)

def invoke_multimodal_chain(query, texts, image_data_1=None, image_data_2=None, api_key=API_KEY):
    """
    Invoke the chain with dynamic content

    Args:
        query (str): The user's query
        texts (str): The text content
        image_data_1 (str, optional): Base64-encoded image data
        image_data_2 (str, optional): Base64-encoded video frame data
        api_key (str): OpenAI API key

    Returns:
        str: The response from the LLM
    """
    # Instantiate the LLM
    gpt4o = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=api_key)
    
    # Instantiate the Output Parser
    parser = StrOutputParser()
    
    # Build the prompt
    dynamic_prompt = build_dynamic_prompt(query, texts, image_data_1, image_data_2)

    # Create the chain
    temp_chain = dynamic_prompt | gpt4o | parser

    # Invoke the chain
    try:
        response = temp_chain.invoke({"query": query})
        return response
    except Exception as e:
        logger.error(f"Error invoking LLM: {str(e)}")
        return f"Error generating response: {str(e)}"

def generate_multimodal_response_for_repo(query, repo_id, api_key=API_KEY):
    """
    Generate a comprehensive response using content from all modalities in a specific repository

    Args:
        query (str): Query text
        repo_id (str): Repository identifier
        api_key (str): OpenAI API key

    Returns:
        dict: Response and source information
    """
    retrieved_content = multimodal_retrieve_for_repo(query, repo_id)

    # Format text content for the prompt
    formatted_texts = format_texts(retrieved_content["texts"])

    # Process the first image if available
    image_data_1 = None
    if retrieved_content["images"]:
        try:
            image_path = retrieved_content["images"][0]
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
                image_data_1 = base64.b64encode(img_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Error processing image: {e}")

    # Process the first video frame if available
    image_data_2 = None
    if retrieved_content["video_frames"]:
        try:
            frame_path = retrieved_content["video_frames"][0]
            with open(frame_path, "rb") as frame_file:
                frame_data = frame_file.read()
                image_data_2 = base64.b64encode(frame_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Error processing video frame: {e}")

    # Get response using the dynamic chain
    response = invoke_multimodal_chain(query, formatted_texts, image_data_1, image_data_2, api_key)

    # Return complete response with sources
    return {
        "query": query,
        "response": response,
        "sources": retrieved_content["sources"]
    }

# Repository Management Functions
def detect_repositories_from_chromadb():
    """
    Automatically detect repositories from ChromaDB collections
    """
    try:
        # Initialize if not already initialized
        if client is None:
            initialize_system()
            
        # Get all collections from ChromaDB
        collections = client.list_collections()

        # Find repository IDs from collection names
        repo_ids = set()
        for collection in collections:
            collection_name = collection.name
            # Look for collections named like text_collection_{repo_id}
            if collection_name.startswith("text_collection_"):
                repo_id = collection_name.replace("text_collection_", "")
                repo_ids.add(repo_id)

        return list(repo_ids)
    except Exception as e:
        logger.error(f"Error detecting repositories from ChromaDB: {str(e)}")
        return []

def load_repo_metadata():
    """Load repository metadata from file or create if it doesn't exist"""
    try:
        if os.path.exists(REPO_METADATA_FILE):
            with open(REPO_METADATA_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading repository metadata: {e}")
        # Create new file if there was an error
        os.makedirs(os.path.dirname(REPO_METADATA_FILE), exist_ok=True)
        with open(REPO_METADATA_FILE, 'w') as f:
            json.dump({}, f)
        return {}

def save_repo_metadata(metadata):
    """Save repository metadata to file"""
    try:
        os.makedirs(os.path.dirname(REPO_METADATA_FILE), exist_ok=True)
        with open(REPO_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving repository metadata: {e}")
        return False

def update_metadata_from_chromadb():
    """
    Update metadata with repositories detected from ChromaDB
    """
    metadata = load_repo_metadata()
    repo_ids = detect_repositories_from_chromadb()

    # Add any missing repositories to metadata
    updated = False
    for repo_id in repo_ids:
        if repo_id not in metadata:
            repo_url = repo_id.replace('_', '/')
            repo_url = re.sub(r'github_com', 'github.com', repo_url)
            repo_url = re.sub(r'_+', '/', repo_url)

            metadata[repo_id] = {
                "repo_url": repo_url,
                "last_commit": None,
                "last_processed": "Auto-detected from ChromaDB",
                "auto_detected": True
            }
            updated = True

    if updated:
        save_repo_metadata(metadata)

    return metadata

# Conversation Management Functions
def init_conversation_store():
    """Initialize or load the conversation store"""
    try:
        if os.path.exists(CONVERSATIONS_FILE):
            with open(CONVERSATIONS_FILE, 'r') as f:
                return json.load(f)
        else:
            # Create new store
            store = {"conversations": []}
            os.makedirs(os.path.dirname(CONVERSATIONS_FILE), exist_ok=True)
            with open(CONVERSATIONS_FILE, 'w') as f:
                json.dump(store, f, indent=2)
            return store
    except Exception as e:
        logger.error(f"Error initializing conversation store: {e}")
        # Create new store on error
        store = {"conversations": []}
        try:
            os.makedirs(os.path.dirname(CONVERSATIONS_FILE), exist_ok=True)
            with open(CONVERSATIONS_FILE, 'w') as f:
                json.dump(store, f, indent=2)
        except:
            pass
        return store

def save_conversation_store(store):
    """Save the conversation store to disk"""
    try:
        os.makedirs(os.path.dirname(CONVERSATIONS_FILE), exist_ok=True)
        with open(CONVERSATIONS_FILE, 'w') as f:
            json.dump(store, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving conversation store: {e}")
        return False

def create_conversation(repo_url, repo_id):
    """Create a new conversation"""
    store = init_conversation_store()

    # Generate a unique ID
    conv_id = f"conv_{uuid.uuid4().hex[:10]}"

    # Create new conversation object
    conversation = {
        "id": conv_id,
        "repository": repo_url,
        "repo_id": repo_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "messages": []
    }

    # Add to store
    store["conversations"].append(conversation)
    save_conversation_store(store)

    return conv_id

def add_message(conv_id, role, content, sources=None):
    """Add a message to a conversation"""
    store = init_conversation_store()

    # Find the conversation
    for conv in store["conversations"]:
        if conv["id"] == conv_id:
            # Create message
            message = {
                "role": role,
                "content": content,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }

            # Add sources if provided (for system messages)
            if sources and role == "system":
                message["sources"] = sources

            # Add to conversation
            conv["messages"].append(message)
            save_conversation_store(store)
            return True

    return False

def get_conversation_history(conv_id):
    """Get all messages in a conversation"""
    store = init_conversation_store()

    for conv in store["conversations"]:
        if conv["id"] == conv_id:
            return conv["messages"]

    return []

def get_conversations_for_repo(repo_id):
    """Get all conversations for a specific repository"""
    store = init_conversation_store()

    return [conv for conv in store["conversations"] if conv["repo_id"] == repo_id]

def list_all_conversations():
    """List all conversations"""
    store = init_conversation_store()
    return store["conversations"]

# GitHub Repository Update Checking
def check_repo_for_changes(repo_url, last_checked_commit=None):
    """Check if a GitHub repository has new commits"""
    # Ensure URL is in the correct format for API calls
    if not repo_url.startswith(("http://", "https://")):
        if "/" in repo_url and not repo_url.startswith("git@"):
            formatted_repo_url = f"https://github.com/{repo_url}"
        else:
            formatted_repo_url = repo_url
    else:
        formatted_repo_url = repo_url

    # Clean up URL format to extract owner/repo
    if "github.com/" in formatted_repo_url:
        parts = formatted_repo_url.split("github.com/")[1].split("/")
    else:
        parts = formatted_repo_url.split("/")
        if len(parts) < 2:
            logger.warning(f"Invalid repository URL format: {repo_url}")
            return False, last_checked_commit, []

    try:
        username = parts[0]
        repo_name = parts[1].split('.')[0]  # Remove .git if present

        # Get the latest commit
        api_url = f"https://api.github.com/repos/{username}/{repo_name}/commits"
        response = requests.get(api_url)

        if response.status_code != 200:
            logger.error(f"Error accessing GitHub API: {response.status_code}")
            return False, last_checked_commit, []

        latest_commit = response.json()[0]["sha"]

        # If this is the first check or the commit has changed
        if last_checked_commit is None or latest_commit != last_checked_commit:
            # Get changed files
            if last_checked_commit:
                compare_url = f"https://api.github.com/repos/{username}/{repo_name}/compare/{last_checked_commit}...{latest_commit}"
                compare_response = requests.get(compare_url)
                if compare_response.status_code == 200:
                    changed_files = [file["filename"] for file in compare_response.json().get("files", [])]
                else:
                    # If comparison fails, assume all files changed
                    changed_files = ["*"]  # Wildcard indicating all files
            else:
                # First run, consider all files changed
                changed_files = ["*"]

            return True, latest_commit, changed_files

        return False, latest_commit, []
    except Exception as e:
        logger.error(f"Error checking for changes: {str(e)}")
        return False, last_checked_commit, []

def poll_repositories(metadata=None):
    """
    Poll repositories for changes and process only the changed files
    """
    if metadata is None:
        metadata = load_repo_metadata()

    poll_log = []
    poll_log.append(f"Starting polling cycle...")

    if not metadata:
        poll_log.append("No repositories found to poll.")
        return poll_log

    for repo_id, repo_data in metadata.items():
        repo_url = repo_data["repo_url"]
        last_commit = repo_data.get("last_commit")

        # Skip repositories with invalid URLs
        if not repo_url or repo_url == "Unknown" or "auto-detected" in repo_url.lower():
            poll_log.append(f"Skipping repository {repo_id}: Invalid URL format")
            continue

        poll_log.append(f"Checking repository {repo_url} for changes...")
        has_changed, latest_commit, changed_files = check_repo_for_changes(repo_url, last_commit)

        if has_changed:
            poll_log.append(f"Repository {repo_url} has changes!")
            poll_log.append(f"New commit: {latest_commit}")
            poll_log.append(f"Changed files: {len(changed_files)}")

            # Process the repository with the new changes
            poll_log.append(f"Processing repository with new changes...")
            try:
                start_time = time.time()

                # Process only the changed files
                result = process_repository(repo_url, DEFAULT_CHROMA_PATH, changed_files)

                # Calculate processing time
                processing_time = time.time() - start_time
                poll_log.append(f"Processing time: {processing_time:.2f} seconds")

                # Update metadata with new information
                metadata[repo_id]["last_commit"] = latest_commit
                metadata[repo_id]["last_processed"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                metadata[repo_id]["repo_stats"] = {
                    "text_files": result.get("text_files", 0),
                    "image_files": result.get("image_files", 0),
                    "video_files": result.get("video_files", 0),
                    "chunks_processed": result.get("chunks_processed", 0)
                }

                poll_log.append(f"Repository processed successfully!")
                poll_log.append(f"Processed {result.get('text_files', 0)} text files, " +
                              f"{result.get('image_files', 0)} images, and " +
                              f"{result.get('video_files', 0)} videos")

            except Exception as e:
                poll_log.append(f"Error processing repository: {str(e)}")
                logger.error(f"Error processing repository: {str(e)}", exc_info=True)
        else:
            poll_log.append(f"No changes detected for {repo_url}")

    # Save updated metadata
    save_repo_metadata(metadata)
    poll_log.append("Polling completed.")
    return poll_log


# Main execution for testing
if __name__ == "__main__":
    print("Initializing system...")
    initialize_system()
    
    print("Detecting repositories from ChromaDB...")
    update_metadata_from_chromadb()
    
    # Display available repositories
    metadata = load_repo_metadata()
    print(f"Available repositories: {len(metadata)}")
    for repo_id, repo_data in metadata.items():
        print(f"- {repo_data['repo_url']} (ID: {repo_id})")
    
   
    import sys
    if len(sys.argv) > 1:
        repo_url = sys.argv[1]
        print(f"\nProcessing repository: {repo_url}")
        result = process_repository(repo_url)
        print(f"Processing complete: {result}")
