#!/usr/bin/env python3
"""
Document caching system for fast retrieval of processed documents
"""

import sqlite3
import hashlib
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import threading

class DocumentCache:
    def __init__(self, db_path: str = "document_cache.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url_hash TEXT UNIQUE NOT NULL,
                    original_url TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    clause_count INTEGER NOT NULL,
                    file_size INTEGER,
                    content_hash TEXT
                )
            """)
            
            # Create clauses table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clauses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    clause_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    section INTEGER,
                    section_title TEXT,
                    word_count INTEGER,
                    char_count INTEGER,
                    document_type TEXT,
                    embedding_uploaded BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (document_id) REFERENCES documents (id),
                    UNIQUE(document_id, clause_id)
                )
            """)
            
            # Create indexes for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_url_hash ON documents(url_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_id ON clauses(document_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_clause_id ON clauses(clause_id)")
            
            conn.commit()
            print("✅ Document cache database initialized")
    
    def _generate_url_hash(self, url: str) -> str:
        """Generate a hash for the document URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate a hash for the document content"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def is_document_cached(self, url: str) -> bool:
        """Check if a document is already cached"""
        url_hash = self._generate_url_hash(url)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents WHERE url_hash = ?", (url_hash,))
            count = cursor.fetchone()[0]
            return count > 0
    
    def get_cached_document_info(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached document information"""
        url_hash = self._generate_url_hash(url)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, document_name, processed_at, clause_count, file_size
                FROM documents 
                WHERE url_hash = ?
            """, (url_hash,))
            
            row = cursor.fetchone()
            if row:
                return {
                    "document_id": row[0],
                    "document_name": row[1],
                    "processed_at": row[2],
                    "clause_count": row[3],
                    "file_size": row[4]
                }
            return None
    
    def get_cached_clauses(self, url: str) -> List[Dict[str, Any]]:
        """Get cached clauses for a document"""
        url_hash = self._generate_url_hash(url)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.clause_id, c.content, c.section, c.section_title, 
                       c.word_count, c.char_count, c.document_type
                FROM clauses c
                JOIN documents d ON c.document_id = d.id
                WHERE d.url_hash = ?
                ORDER BY c.section
            """, (url_hash,))
            
            clauses = []
            for row in cursor.fetchall():
                clause = {
                    "id": row[0],
                    "content": row[1],
                    "document": self._get_document_name_by_hash(url_hash),
                    "section": row[2],
                    "section_title": row[3],
                    "metadata": {
                        "word_count": row[4],
                        "char_count": row[5],
                        "document_type": row[6]
                    }
                }
                clauses.append(clause)
            
            return clauses
    
    def _get_document_name_by_hash(self, url_hash: str) -> str:
        """Get document name by URL hash"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT document_name FROM documents WHERE url_hash = ?", (url_hash,))
            row = cursor.fetchone()
            return row[0] if row else "unknown"
    
    def cache_document(self, url: str, document_name: str, clauses: List[Dict[str, Any]], 
                      file_size: int = None, content: str = None) -> bool:
        """Cache a processed document and its clauses"""
        url_hash = self._generate_url_hash(url)
        content_hash = self._generate_content_hash(content) if content else None
        
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Insert document record
                    cursor.execute("""
                        INSERT OR REPLACE INTO documents 
                        (url_hash, original_url, document_name, clause_count, file_size, content_hash)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (url_hash, url, document_name, len(clauses), file_size, content_hash))
                    
                    document_id = cursor.lastrowid
                    
                    # Clear existing clauses for this document
                    cursor.execute("DELETE FROM clauses WHERE document_id = ?", (document_id,))
                    
                    # Insert clauses
                    for clause in clauses:
                        cursor.execute("""
                            INSERT INTO clauses 
                            (document_id, clause_id, content, section, section_title, 
                             word_count, char_count, document_type)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            document_id,
                            clause["id"],
                            clause["content"],
                            clause["section"],
                            clause.get("section_title", ""),
                            clause["metadata"]["word_count"],
                            clause["metadata"]["char_count"],
                            clause["metadata"]["document_type"]
                        ))
                    
                    conn.commit()
                    print(f"✅ Cached document '{document_name}' with {len(clauses)} clauses")
                    return True
                    
            except Exception as e:
                print(f"❌ Error caching document: {e}")
                return False
    
    def mark_embeddings_uploaded(self, url: str) -> bool:
        """Mark that embeddings have been uploaded to Pinecone for this document"""
        url_hash = self._generate_url_hash(url)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE clauses 
                    SET embedding_uploaded = TRUE 
                    WHERE document_id = (
                        SELECT id FROM documents WHERE url_hash = ?
                    )
                """, (url_hash,))
                conn.commit()
                return True
        except Exception as e:
            print(f"❌ Error marking embeddings as uploaded: {e}")
            return False
    
    def are_embeddings_uploaded(self, url: str) -> bool:
        """Check if embeddings have been uploaded to Pinecone for this document"""
        url_hash = self._generate_url_hash(url)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as total, 
                       SUM(CASE WHEN embedding_uploaded THEN 1 ELSE 0 END) as uploaded
                FROM clauses c
                JOIN documents d ON c.document_id = d.id
                WHERE d.url_hash = ?
            """, (url_hash,))
            
            row = cursor.fetchone()
            if row and row[0] > 0:
                return row[1] == row[0]  # All clauses have embeddings uploaded
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get document count
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            # Get clause count
            cursor.execute("SELECT COUNT(*) FROM clauses")
            clause_count = cursor.fetchone()[0]
            
            # Get total file size
            cursor.execute("SELECT SUM(file_size) FROM documents WHERE file_size IS NOT NULL")
            total_size = cursor.fetchone()[0] or 0
            
            # Get recent documents
            cursor.execute("""
                SELECT document_name, processed_at, clause_count 
                FROM documents 
                ORDER BY processed_at DESC 
                LIMIT 5
            """)
            recent_docs = cursor.fetchall()
            
            return {
                "total_documents": doc_count,
                "total_clauses": clause_count,
                "total_file_size_bytes": total_size,
                "recent_documents": [
                    {
                        "name": row[0],
                        "processed_at": row[1],
                        "clause_count": row[2]
                    }
                    for row in recent_docs
                ]
            }
    
    def clear_cache(self) -> bool:
        """Clear all cached documents and clauses"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM clauses")
                cursor.execute("DELETE FROM documents")
                conn.commit()
                print("✅ Cache cleared successfully")
                return True
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
            return False

# Global cache instance
document_cache = DocumentCache()