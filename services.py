import google.generativeai as genai
import pinecone
import redis
import json
import hashlib
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from config import config
from models import Clause, QueryResponse

class GeminiService:
    def __init__(self):
        # Debug: Print what we're actually getting
        print(f"🔍 Debug - config.GEMINI_MODEL: '{config.GEMINI_MODEL}'")
        print(f"🔍 Debug - os.getenv('GEMINI_MODEL'): '{os.getenv('GEMINI_MODEL')}'")
        
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Configure Gemini API
        genai.configure(api_key=config.GEMINI_API_KEY)
        
        # List available models for debugging
        try:
            available_models = [m.name for m in genai.list_models()]
            print(f"🔍 Available Gemini models: {available_models[:3]}...")  # Show first 3
        except Exception as e:
            print(f"⚠️ Could not list models: {e}")
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel(config.GEMINI_MODEL)
            print(f"✅ Gemini API key loaded: {config.GEMINI_API_KEY[:10]}...")
            print(f"✅ Using Gemini model: {config.GEMINI_MODEL}")
        except Exception as e:
            print(f"❌ Failed to initialize model {config.GEMINI_MODEL}: {e}")
            # Try fallback model
            fallback_model = "gemini-1.5-flash-latest"
            print(f"🔄 Trying fallback model: {fallback_model}")
            self.model = genai.GenerativeModel(fallback_model)
            print(f"✅ Using fallback Gemini model: {fallback_model}")
    
    def analyze_clauses(self, user_question: str, clauses: List[Clause]) -> QueryResponse:
        """Use Gemini to analyze clauses and generate structured response"""
        
        # Prepare clauses context
        clauses_context = "\n".join([
            f"Clause ID: {clause.clause_id} (Score: {clause.score})\nContent: {clause.content}\n"
            for clause in clauses
        ])
        
        prompt = f"""You are an expert AI assistant for insurance, legal, HR, and compliance documents.

Document Context:
{clauses_context}

User Question: "{user_question}"

Instructions:
1. Only use content from the provided clauses
2. Do not hallucinate - if clauses don't answer the question, say so
3. Extract conditions, eligibility, exclusions, waiting periods, limits
4. Choose the best matching clause and explain why
5. You MUST respond with ONLY a valid JSON object, no other text before or after
6. Use this EXACT format:

{{
    "answer": "Short summary answer based on the clauses",
    "conditions": ["List any conditions or limitations found"],
    "clause_id": "Most relevant clause ID from the provided clauses",
    "score": 0.95,
    "explanation": "Brief explanation of how the clause answers the question"
}}

IMPORTANT: Return ONLY the JSON object, no markdown formatting, no explanatory text, just the raw JSON."""

        try:
            print(f"🔧 Making Gemini API call with model: {config.GEMINI_MODEL}")
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            print("✅ Gemini API call successful!")
            print(f"🔍 Raw response: {response.text[:200]}...")
            
            # Clean and extract JSON from the response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith('```'):
                response_text = response_text[3:]   # Remove ```
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove trailing ```
            
            # Find JSON object in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                print(f"🔍 Extracted JSON: {json_text[:100]}...")
                result = json.loads(json_text)
                return QueryResponse(**result)
            else:
                raise json.JSONDecodeError("No JSON object found in response", response_text, 0)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {str(e)}")
            print(f"🔍 Raw response was: {response.text if 'response' in locals() else 'No response'}")
            
            # Try to create a structured response from the raw text
            if 'response' in locals() and response.text:
                best_clause = max(clauses, key=lambda x: x.score) if clauses else None
                
                # Extract basic information from the raw response
                raw_text = response.text.strip()
                
                return QueryResponse(
                    answer=raw_text[:200] + "..." if len(raw_text) > 200 else raw_text,
                    conditions=[],
                    clause_id=best_clause.clause_id if best_clause else "",
                    score=best_clause.score if best_clause else 0.0,
                    explanation="Response generated from raw AI output due to JSON parsing issue"
                )
            else:
                # Complete fallback
                best_clause = max(clauses, key=lambda x: x.score) if clauses else None
                return QueryResponse(
                    answer="Error parsing AI response - JSON format issue",
                    conditions=[],
                    clause_id=best_clause.clause_id if best_clause else "",
                    score=best_clause.score if best_clause else 0.0,
                    explanation="Fallback response due to JSON parsing error"
                )
        except Exception as e:
            print(f"❌ Gemini API call failed with error: {str(e)}")
            # Fallback response
            best_clause = max(clauses, key=lambda x: x.score) if clauses else None
            return QueryResponse(
                answer=f"Error processing with Gemini: {str(e)}",
                conditions=[],
                clause_id=best_clause.clause_id if best_clause else "",
                score=best_clause.score if best_clause else 0.0,
                explanation="Fallback response due to processing error"
            )

class PineconeService:
    def __init__(self):
        try:
            # Use modern Pinecone client (v3.0+)
            from pinecone import Pinecone
            pc = Pinecone(api_key=config.PINECONE_API_KEY)
            self.index = pc.Index(config.PINECONE_INDEX_NAME)
            print("✅ Using modern Pinecone client")
        except Exception as e:
            print(f"❌ Pinecone connection failed: {e}")
            self.index = None
    
    def search_clauses(self, query_embedding: List[float], top_k: int = 10) -> List[Clause]:
        """Search for relevant clauses using vector similarity"""
        if not self.index:
            print("Pinecone not available, returning empty results")
            return []
            
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            clauses = []
            for match in results.matches:
                clauses.append(Clause(
                    clause_id=match.id,
                    content=match.metadata.get('content', ''),
                    score=match.score
                ))
            
            return clauses
        except Exception as e:
            print(f"Pinecone search error: {e}")
            return []

class RedisService:
    def __init__(self):
        try:
            # Create Redis connection with connection pooling
            self.redis_client = redis.from_url(
                config.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            self.redis_client.ping()
            print("✅ Redis connection established successfully")
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            self.redis_client = None
    
    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    def get_cached_response(self, query_hash: str) -> Optional[QueryResponse]:
        """Get cached response for a query"""
        if not self.is_connected():
            print("Redis not connected, skipping cache lookup")
            return None
            
        try:
            cached = self.redis_client.get(f"query:{query_hash}")
            if cached:
                data = json.loads(cached)
                return QueryResponse(**data)
        except Exception as e:
            print(f"Redis get error: {e}")
        return None
    
    def cache_response(self, query_hash: str, response: QueryResponse):
        """Cache query response"""
        if not self.is_connected():
            print("Redis not connected, skipping cache storage")
            return
            
        try:
            self.redis_client.setex(
                f"query:{query_hash}",
                config.REDIS_TTL,
                response.json()
            )
            print(f"✅ Cached response for query hash: {query_hash[:8]}...")
        except Exception as e:
            print(f"Redis set error: {e}")
    
    def generate_query_hash(self, question: str, clause_ids: List[str]) -> str:
        """Generate hash for query caching"""
        content = f"{question}:{':'.join(sorted(clause_ids))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def generate_semantic_query_hash(self, question: str, document_name: str) -> str:
        """Generate semantic hash for similar questions"""
        # Normalize question for better cache hits
        normalized_question = question.lower().strip()
        # Remove common question words that don't affect meaning
        stop_words = ['what', 'is', 'the', 'how', 'when', 'where', 'why', 'does', 'do', 'can', 'will']
        words = [w for w in normalized_question.split() if w not in stop_words]
        semantic_key = f"{document_name}:{' '.join(sorted(words))}"
        return hashlib.md5(semantic_key.encode()).hexdigest()
    
    def get_semantic_cached_response(self, question: str, document_name: str) -> Optional[str]:
        """Get cached response for semantically similar questions"""
        if not self.is_connected():
            return None
        
        try:
            semantic_hash = self.generate_semantic_query_hash(question, document_name)
            cached = self.redis_client.get(f"semantic:{semantic_hash}")
            return cached
        except Exception as e:
            print(f"Redis semantic get error: {e}")
        return None
    
    def cache_semantic_response(self, question: str, document_name: str, response: str):
        """Cache response for semantic similarity"""
        if not self.is_connected():
            return
        
        try:
            semantic_hash = self.generate_semantic_query_hash(question, document_name)
            self.redis_client.setex(
                f"semantic:{semantic_hash}",
                config.REDIS_TTL * 2,  # Longer TTL for semantic cache
                response
            )
        except Exception as e:
            print(f"Redis semantic set error: {e}")
    
    def generate_document_hash(self, url: str) -> str:
        """Generate hash for document URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    # Document caching methods
    def is_document_cached(self, url: str) -> bool:
        """Check if a document is cached in Redis"""
        if not self.is_connected():
            return False
        
        try:
            doc_hash = self.generate_document_hash(url)
            return self.redis_client.exists(f"doc:{doc_hash}:info") > 0
        except Exception as e:
            print(f"Redis document check error: {e}")
            return False
    
    def cache_document(self, url: str, document_name: str, clauses: List[Dict], 
                      file_size: int = None, content: str = None) -> bool:
        """Cache document and its clauses in Redis"""
        if not self.is_connected():
            print("Redis not connected, skipping document caching")
            return False
        
        try:
            doc_hash = self.generate_document_hash(url)
            
            # Cache document info
            doc_info = {
                "url": url,
                "document_name": document_name,
                "clause_count": len(clauses),
                "file_size": file_size,
                "cached_at": json.dumps(datetime.now().isoformat()),
                "embeddings_uploaded": False
            }
            
            # Set document info with TTL (24 hours)
            self.redis_client.setex(
                f"doc:{doc_hash}:info",
                86400,  # 24 hours TTL
                json.dumps(doc_info)
            )
            
            # Cache clauses
            clauses_data = []
            for clause in clauses:
                clause_data = {
                    "id": clause["id"],
                    "content": clause["content"],
                    "document": clause["document"],
                    "section": clause["section"],
                    "section_title": clause.get("section_title", ""),
                    "metadata": clause["metadata"]
                }
                clauses_data.append(clause_data)
            
            # Set clauses with TTL (24 hours)
            self.redis_client.setex(
                f"doc:{doc_hash}:clauses",
                86400,  # 24 hours TTL
                json.dumps(clauses_data)
            )
            
            print(f"✅ Document cached in Redis: {document_name} ({len(clauses)} clauses)")
            return True
            
        except Exception as e:
            print(f"Redis document caching error: {e}")
            return False
    
    def get_cached_document_info(self, url: str) -> Optional[Dict]:
        """Get cached document info from Redis"""
        if not self.is_connected():
            return None
        
        try:
            doc_hash = self.generate_document_hash(url)
            cached_info = self.redis_client.get(f"doc:{doc_hash}:info")
            
            if cached_info:
                info = json.loads(cached_info)
                return {
                    "document_name": info["document_name"],
                    "clause_count": info["clause_count"],
                    "file_size": info.get("file_size"),
                    "cached_at": json.loads(info["cached_at"]),
                    "embeddings_uploaded": info.get("embeddings_uploaded", False)
                }
            return None
            
        except Exception as e:
            print(f"Redis document info retrieval error: {e}")
            return None
    
    def get_cached_clauses(self, url: str) -> List[Dict]:
        """Get cached clauses from Redis"""
        if not self.is_connected():
            return []
        
        try:
            doc_hash = self.generate_document_hash(url)
            cached_clauses = self.redis_client.get(f"doc:{doc_hash}:clauses")
            
            if cached_clauses:
                return json.loads(cached_clauses)
            return []
            
        except Exception as e:
            print(f"Redis clauses retrieval error: {e}")
            return []
    
    def mark_embeddings_uploaded(self, url: str) -> bool:
        """Mark that embeddings have been uploaded to Pinecone"""
        if not self.is_connected():
            return False
        
        try:
            doc_hash = self.generate_document_hash(url)
            cached_info = self.redis_client.get(f"doc:{doc_hash}:info")
            
            if cached_info:
                info = json.loads(cached_info)
                info["embeddings_uploaded"] = True
                
                # Update with same TTL
                ttl = self.redis_client.ttl(f"doc:{doc_hash}:info")
                if ttl > 0:
                    self.redis_client.setex(
                        f"doc:{doc_hash}:info",
                        ttl,
                        json.dumps(info)
                    )
                    return True
            return False
            
        except Exception as e:
            print(f"Redis embeddings marking error: {e}")
            return False
    
    def are_embeddings_uploaded(self, url: str) -> bool:
        """Check if embeddings have been uploaded to Pinecone"""
        info = self.get_cached_document_info(url)
        return info.get("embeddings_uploaded", False) if info else False
    
    def get_document_cache_stats(self) -> Dict[str, Any]:
        """Get document cache statistics from Redis"""
        if not self.is_connected():
            return {"error": "Redis not connected"}
        
        try:
            # Get all document keys
            doc_keys = self.redis_client.keys("doc:*:info")
            
            total_docs = len(doc_keys)
            total_clauses = 0
            total_size = 0
            recent_docs = []
            
            for key in doc_keys[:10]:  # Limit to first 10 for performance
                try:
                    info_str = self.redis_client.get(key)
                    if info_str:
                        info = json.loads(info_str)
                        total_clauses += info.get("clause_count", 0)
                        total_size += info.get("file_size", 0) or 0
                        
                        recent_docs.append({
                            "name": info["document_name"],
                            "cached_at": json.loads(info["cached_at"]),
                            "clause_count": info["clause_count"],
                            "embeddings_uploaded": info.get("embeddings_uploaded", False)
                        })
                except Exception as e:
                    print(f"Error processing doc key {key}: {e}")
                    continue
            
            # Sort by cached_at
            recent_docs.sort(key=lambda x: x["cached_at"], reverse=True)
            
            return {
                "total_documents": total_docs,
                "total_clauses": total_clauses,
                "total_file_size_bytes": total_size,
                "recent_documents": recent_docs[:5]
            }
            
        except Exception as e:
            print(f"Redis cache stats error: {e}")
            return {"error": str(e)}
    
    def clear_document_cache(self) -> bool:
        """Clear all document cache from Redis"""
        if not self.is_connected():
            return False
        
        try:
            # Get all document-related keys
            doc_keys = self.redis_client.keys("doc:*")
            
            if doc_keys:
                self.redis_client.delete(*doc_keys)
                print(f"✅ Cleared {len(doc_keys)} document cache entries from Redis")
            else:
                print("✅ No document cache entries to clear")
            
            return True
            
        except Exception as e:
            print(f"Redis cache clearing error: {e}")
            return False
    
    def test_connection(self) -> dict:
        """Test Redis connection and return status"""
        try:
            if not self.redis_client:
                return {"status": "error", "message": "Redis client not initialized"}
            
            # Test basic operations
            test_key = "test:connection"
            self.redis_client.set(test_key, "test_value", ex=10)
            value = self.redis_client.get(test_key)
            self.redis_client.delete(test_key)
            
            return {
                "status": "success", 
                "message": "Redis connection working",
                "test_result": value == "test_value"
            }
        except Exception as e:
            return {"status": "error", "message": f"Redis test failed: {str(e)}"}

# Service instances - Initialize with error handling
try:
    gemini_service = GeminiService()
    print("✅ Gemini service initialized")
except Exception as e:
    print(f"❌ Gemini service failed: {e}")
    gemini_service = None

try:
    pinecone_service = PineconeService()
    print("✅ Pinecone service initialized")
except Exception as e:
    print(f"❌ Pinecone service failed: {e}")
    pinecone_service = None

try:
    redis_service = RedisService()
    print("✅ Redis service initialized")
except Exception as e:
    print(f"❌ Redis service failed: {e}")
    redis_service = None