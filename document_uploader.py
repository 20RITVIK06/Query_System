#!/usr/bin/env python3
"""
Document Upload System for Pinecone
Processes various document types and uploads them as vector embeddings
"""

import os
import json
import hashlib
from typing import List, Dict, Any
from pathlib import Path
import google.generativeai as genai
from config import config
from services import pinecone_service
import time
import PyPDF2
import pdfplumber
import re
import concurrent.futures
import threading

class DocumentProcessor:
    def __init__(self):
        # Configure Gemini for embeddings
        genai.configure(api_key=config.GEMINI_API_KEY)
        
        # Initialize embedding model
        self.embedding_model = "models/embedding-001"
        
        print("✅ Document processor initialized")
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert JSON to readable text
                return json.dumps(data, indent=2)
        
        elif file_path.suffix.lower() in ['.md', '.markdown']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_path.suffix.lower() == '.pdf':
            return self.extract_text_from_pdf(file_path)
        
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF files using multiple methods"""
        text = ""
        
        # Method 1: Try pdfplumber (better for complex layouts)
        try:
            print("🔄 Trying pdfplumber for PDF extraction...")
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- PAGE {page_num + 1} ---\n"
                        text += page_text + "\n"
            
            if text.strip():
                print("✅ Successfully extracted text using pdfplumber")
                return self.clean_pdf_text(text)
        
        except Exception as e:
            print(f"⚠️ pdfplumber failed: {e}")
        
        # Method 2: Fallback to PyPDF2
        try:
            print("🔄 Trying PyPDF2 for PDF extraction...")
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- PAGE {page_num + 1} ---\n"
                        text += page_text + "\n"
            
            if text.strip():
                print("✅ Successfully extracted text using PyPDF2")
                return self.clean_pdf_text(text)
        
        except Exception as e:
            print(f"❌ PyPDF2 also failed: {e}")
        
        if not text.strip():
            raise ValueError("Could not extract text from PDF using any method")
        
        return self.clean_pdf_text(text)
    
    def clean_pdf_text(self, text: str) -> str:
        """Clean and normalize text extracted from PDF"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove page headers/footers (basic cleanup)
        text = re.sub(r'--- PAGE \d+ ---\n', '', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between words
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\n\2', text)  # Add paragraph breaks
        
        return text.strip()
    
    def split_document_into_clauses(self, text: str, document_name: str) -> List[Dict[str, Any]]:
        """Split document into meaningful clauses/sections with intelligent parsing"""
        
        # First, try to identify sections by headers/titles
        sections = self.identify_document_sections(text)
        
        if sections:
            print(f"✅ Identified {len(sections)} sections with headers")
            clauses = []
            for i, section in enumerate(sections):
                if len(section['content']) < 100:  # Skip very short sections
                    continue
                
                clause_id = f"{document_name}_section_{i+1:03d}"
                
                clause = {
                    "id": clause_id,
                    "content": section['content'],
                    "document": document_name,
                    "section": i + 1,
                    "section_title": section.get('title', f"Section {i+1}"),
                    "metadata": {
                        "document_type": self.detect_document_type(document_name),
                        "word_count": len(section['content'].split()),
                        "char_count": len(section['content']),
                        "has_title": bool(section.get('title'))
                    }
                }
                clauses.append(clause)
        else:
            # Fallback to paragraph-based splitting
            print("📝 Using paragraph-based splitting")
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            clauses = []
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) < 100:  # Skip very short paragraphs
                    continue
                    
                clause_id = f"{document_name}_clause_{i+1:03d}"
                
                clause = {
                    "id": clause_id,
                    "content": paragraph,
                    "document": document_name,
                    "section": i + 1,
                    "section_title": f"Clause {i+1}",
                    "metadata": {
                        "document_type": self.detect_document_type(document_name),
                        "word_count": len(paragraph.split()),
                        "char_count": len(paragraph),
                        "has_title": False
                    }
                }
                clauses.append(clause)
        
        return clauses
    
    def identify_document_sections(self, text: str) -> List[Dict[str, str]]:
        """Identify document sections by headers and titles"""
        sections = []
        
        # Common patterns for section headers in legal/insurance documents
        section_patterns = [
            r'^(SECTION\s+\d+[:\-\.]?\s*[A-Z\s]+)',
            r'^(ARTICLE\s+\d+[:\-\.]?\s*[A-Z\s]+)',
            r'^(CHAPTER\s+\d+[:\-\.]?\s*[A-Z\s]+)',
            r'^(\d+\.\s+[A-Z][A-Z\s]+)',
            r'^([A-Z][A-Z\s]{10,})',  # All caps titles
            r'^(\d+\.\d+\s+[A-Z][A-Za-z\s]+)',
        ]
        
        lines = text.split('\n')
        current_section = {"title": None, "content": ""}
        
        for line in lines:
            line = line.strip()
            if not line:
                current_section["content"] += "\n"
                continue
            
            # Check if this line is a section header
            is_header = False
            for pattern in section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    # Save previous section if it has content
                    if current_section["content"].strip():
                        sections.append({
                            "title": current_section["title"],
                            "content": current_section["content"].strip()
                        })
                    
                    # Start new section
                    current_section = {"title": line, "content": ""}
                    is_header = True
                    break
            
            if not is_header:
                current_section["content"] += line + "\n"
        
        # Add the last section
        if current_section["content"].strip():
            sections.append({
                "title": current_section["title"],
                "content": current_section["content"].strip()
            })
        
        # Return sections only if we found meaningful headers
        if len(sections) > 1 and any(s["title"] for s in sections):
            return sections
        
        return []  # Fallback to paragraph splitting
    
    def detect_document_type(self, document_name: str) -> str:
        """Detect document type based on filename or content"""
        name_lower = document_name.lower()
        
        if any(word in name_lower for word in ['insurance', 'policy', 'coverage']):
            return "insurance"
        elif any(word in name_lower for word in ['legal', 'contract', 'agreement']):
            return "legal"
        elif any(word in name_lower for word in ['hr', 'employee', 'handbook']):
            return "hr"
        elif any(word in name_lower for word in ['compliance', 'regulation', 'audit']):
            return "compliance"
        else:
            return "general"
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Gemini"""
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            return None
    
    def upload_to_pinecone(self, clauses: List[Dict[str, Any]]) -> bool:
        """Upload clauses to Pinecone with embeddings using parallel processing"""
        if not pinecone_service or not pinecone_service.index:
            print("❌ Pinecone service not available")
            return False
        
        import concurrent.futures
        import threading
        
        print(f"🚀 Processing {len(clauses)} clauses with parallel embedding generation...")
        
        # Generate embeddings in parallel
        vectors_to_upsert = []
        lock = threading.Lock()
        
        def process_clause(clause):
            try:
                embedding = self.generate_embedding(clause['content'])
                if embedding is None:
                    return None
                
                vector = {
                    "id": clause['id'],
                    "values": embedding,
                    "metadata": {
                        "content": clause['content'],
                        "document": clause['document'],
                        "section": clause['section'],
                        "document_type": clause['metadata']['document_type'],
                        "word_count": clause['metadata']['word_count'],
                        "char_count": clause['metadata']['char_count']
                    }
                }
                return vector
            except Exception as e:
                print(f"❌ Error processing clause {clause['id']}: {e}")
                return None
        
        # Process clauses in parallel (max 5 concurrent to avoid rate limits)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_clause = {executor.submit(process_clause, clause): clause for clause in clauses}
            
            for future in concurrent.futures.as_completed(future_to_clause):
                clause = future_to_clause[future]
                try:
                    vector = future.result()
                    if vector:
                        with lock:
                            vectors_to_upsert.append(vector)
                            print(f"✅ Processed clause: {clause['id']} ({len(vectors_to_upsert)}/{len(clauses)})")
                except Exception as e:
                    print(f"❌ Clause processing failed for {clause['id']}: {e}")
        
        if not vectors_to_upsert:
            print("❌ No vectors to upload")
            return False
        
        print(f"🔄 Uploading {len(vectors_to_upsert)} vectors to Pinecone...")
        
        # Upload in batches of 100
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            try:
                pinecone_service.index.upsert(vectors=batch)
                print(f"✅ Uploaded batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1}")
                if i + batch_size < len(vectors_to_upsert):
                    time.sleep(0.5)  # Reduced rate limiting
            except Exception as e:
                print(f"❌ Batch upload failed: {e}")
                return False
        
        return True
    
    def process_document(self, file_path: str) -> bool:
        """Complete document processing pipeline"""
        try:
            print(f"📄 Processing document: {file_path}")
            
            # Extract text
            text = self.extract_text_from_file(file_path)
            print(f"✅ Extracted {len(text)} characters")
            
            # Get document name
            document_name = Path(file_path).stem
            
            # Split into clauses
            clauses = self.split_document_into_clauses(text, document_name)
            print(f"✅ Split into {len(clauses)} clauses")
            
            # Upload to Pinecone
            success = self.upload_to_pinecone(clauses)
            
            if success:
                print(f"🎉 Successfully uploaded {document_name} to Pinecone!")
                return True
            else:
                print(f"❌ Failed to upload {document_name}")
                return False
                
        except Exception as e:
            print(f"❌ Document processing failed: {e}")
            return False

def main():
    """Main function to upload documents"""
    processor = DocumentProcessor()
    
    # Create documents directory if it doesn't exist
    docs_dir = Path("documents")
    docs_dir.mkdir(exist_ok=True)
    
    print(f"📁 Looking for documents in: {docs_dir.absolute()}")
    
    # Find all supported document files
    supported_extensions = ['.txt', '.json', '.md', '.markdown', '.pdf']
    document_files = []
    
    for ext in supported_extensions:
        document_files.extend(docs_dir.glob(f"*{ext}"))
    
    if not document_files:
        print("⚠️ No documents found!")
        print("Please add documents to the 'documents' folder with these extensions:")
        print("- .txt (plain text)")
        print("- .json (JSON files)")
        print("- .md/.markdown (Markdown files)")
        print("- .pdf (PDF documents)")
        return
    
    print(f"📚 Found {len(document_files)} documents to process:")
    for doc in document_files:
        print(f"  - {doc.name}")
    
    # Process each document
    successful_uploads = 0
    for doc_file in document_files:
        if processor.process_document(str(doc_file)):
            successful_uploads += 1
        print("-" * 50)
    
    print(f"🎯 Upload Summary:")
    print(f"  Total documents: {len(document_files)}")
    print(f"  Successful uploads: {successful_uploads}")
    print(f"  Failed uploads: {len(document_files) - successful_uploads}")

if __name__ == "__main__":
    main()