#!/usr/bin/env python3
"""
Cache local BAJAJ.pdf document in Redis for fast retrieval
"""

import os
from pathlib import Path
from services import redis_service
from document_uploader import DocumentProcessor

def cache_local_bajaj_document():
    """Cache the local BAJAJ.pdf document in Redis"""
    
    print("🚀 CACHING LOCAL BAJAJ DOCUMENT IN REDIS")
    print("=" * 60)
    
    # Check if Redis is available
    if not redis_service or not redis_service.is_connected():
        print("❌ Redis service not available")
        return False
    
    # Path to local BAJAJ document
    bajaj_pdf_path = Path("documents/BAJAJ.pdf")
    
    if not bajaj_pdf_path.exists():
        print(f"❌ BAJAJ.pdf not found at: {bajaj_pdf_path.absolute()}")
        return False
    
    print(f"📄 Found BAJAJ.pdf at: {bajaj_pdf_path.absolute()}")
    
    # Create a pseudo-URL for the local document (for consistent caching)
    local_document_url = f"file://{bajaj_pdf_path.absolute()}"
    document_name = "BAJAJ"
    
    # Check if already cached
    if redis_service.is_document_cached(local_document_url):
        print("✅ BAJAJ document already cached in Redis!")
        cached_info = redis_service.get_cached_document_info(local_document_url)
        print(f"📋 Cached info: {cached_info}")
        return True
    
    try:
        # Initialize document processor
        processor = DocumentProcessor()
        print("✅ Document processor initialized")
        
        # Extract text from the PDF
        print("🔄 Extracting text from BAJAJ.pdf...")
        text = processor.extract_text_from_file(str(bajaj_pdf_path))
        print(f"✅ Extracted {len(text):,} characters")
        
        # Split into clauses
        print("🔄 Splitting document into clauses...")
        clauses = processor.split_document_into_clauses(text, document_name)
        print(f"✅ Split into {len(clauses)} clauses")
        
        # Get file size
        file_size = bajaj_pdf_path.stat().st_size
        print(f"📊 File size: {file_size:,} bytes")
        
        # Cache in Redis
        print("🔄 Caching document in Redis...")
        cache_success = redis_service.cache_document(
            local_document_url,
            document_name,
            clauses,
            file_size,
            text
        )
        
        if cache_success:
            print("✅ BAJAJ document successfully cached in Redis!")
            
            # Upload to Pinecone if not already there
            print("🔄 Uploading clauses to Pinecone...")
            upload_success = processor.upload_to_pinecone(clauses)
            
            if upload_success:
                # Mark embeddings as uploaded
                redis_service.mark_embeddings_uploaded(local_document_url)
                print("✅ Clauses uploaded to Pinecone and marked in Redis")
            else:
                print("⚠️ Failed to upload to Pinecone (cached in Redis anyway)")
            
            return True
        else:
            print("❌ Failed to cache document in Redis")
            return False
            
    except Exception as e:
        print(f"❌ Error caching BAJAJ document: {e}")
        return False

def test_cached_bajaj_document():
    """Test retrieval of cached BAJAJ document"""
    
    print("\n🧪 TESTING CACHED BAJAJ DOCUMENT RETRIEVAL")
    print("=" * 60)
    
    if not redis_service or not redis_service.is_connected():
        print("❌ Redis service not available")
        return
    
    local_document_url = f"file://{Path('documents/BAJAJ.pdf').absolute()}"
    
    # Test document info retrieval
    print("1. Testing document info retrieval...")
    cached_info = redis_service.get_cached_document_info(local_document_url)
    if cached_info:
        print(f"✅ Document info: {cached_info}")
    else:
        print("❌ No cached document info found")
        return
    
    # Test clauses retrieval
    print("2. Testing clauses retrieval...")
    cached_clauses = redis_service.get_cached_clauses(local_document_url)
    if cached_clauses:
        print(f"✅ Retrieved {len(cached_clauses)} clauses")
        print(f"📄 First clause preview: {cached_clauses[0]['content'][:100]}...")
    else:
        print("❌ No cached clauses found")
        return
    
    # Test embeddings status
    print("3. Testing embeddings status...")
    embeddings_uploaded = redis_service.are_embeddings_uploaded(local_document_url)
    print(f"✅ Embeddings uploaded: {embeddings_uploaded}")
    
    print("🎉 All tests passed! BAJAJ document is properly cached in Redis.")

def show_cache_comparison():
    """Show comparison between local and remote document caching"""
    
    print("\n📊 REDIS CACHE COMPARISON")
    print("=" * 60)
    
    if not redis_service or not redis_service.is_connected():
        print("❌ Redis service not available")
        return
    
    # Get cache statistics
    stats = redis_service.get_document_cache_stats()
    
    print(f"📈 Total cached documents: {stats.get('total_documents', 0)}")
    print(f"📈 Total cached clauses: {stats.get('total_clauses', 0)}")
    print(f"📈 Total cache size: {stats.get('total_file_size_bytes', 0):,} bytes")
    
    print("\n📋 Recent documents:")
    for doc in stats.get('recent_documents', []):
        doc_type = "Local" if doc['name'] == 'BAJAJ' else "Remote"
        print(f"  • {doc['name']} ({doc_type}): {doc['clause_count']} clauses, embeddings: {doc.get('embeddings_uploaded', False)}")

def main():
    """Main function to cache local document and test"""
    
    # Cache the local BAJAJ document
    success = cache_local_bajaj_document()
    
    if success:
        # Test the cached document
        test_cached_bajaj_document()
        
        # Show cache comparison
        show_cache_comparison()
        
        print("\n🎉 SUCCESS!")
        print("Your local BAJAJ.pdf document is now cached in Redis alongside the remote document.")
        print("Both documents will now have lightning-fast retrieval times!")
    else:
        print("\n❌ FAILED!")
        print("Could not cache the local BAJAJ document.")

if __name__ == "__main__":
    main()