#!/usr/bin/env python3
"""
Create a new Pinecone index with correct dimensions for Gemini embeddings
"""

from pinecone import Pinecone
from config import config

def create_index():
    print("=== CREATING PINECONE INDEX ===")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    
    # Index configuration
    index_name = "document-clauses-768"  # New index name with dimension in name
    dimension = 768  # Gemini embedding dimension
    metric = "cosine"  # Good for text similarity
    
    print(f"Index name: {index_name}")
    print(f"Dimension: {dimension}")
    print(f"Metric: {metric}")
    
    try:
        # Check if index already exists
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if index_name in existing_indexes:
            print(f"✅ Index '{index_name}' already exists!")
            
            # Get index info
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            print(f"📊 Current stats: {stats}")
            
        else:
            print(f"🔄 Creating new index '{index_name}'...")
            
            # Create the index
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
            
            print(f"✅ Index '{index_name}' created successfully!")
            print(f"🔧 Update your .env file:")
            print(f"   PINECONE_INDEX_NAME={index_name}")
            
    except Exception as e:
        print(f"❌ Failed to create index: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = create_index()
    if success:
        print("\n🎉 Index setup complete!")
        print("Next steps:")
        print("1. Update PINECONE_INDEX_NAME in your .env file")
        print("2. Run document_uploader.py to upload your documents")
    else:
        print("\n❌ Index setup failed!")