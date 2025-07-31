from fastapi import FastAPI, HTTPException
import json
import requests
import tempfile
import os
from typing import List, Dict, Any
import google.generativeai as genai

# Import models first
from models import Clause, QueryRequest, QueryResponse
from pydantic import BaseModel

# Import services after models
from services import gemini_service, redis_service, pinecone_service
from config import config
from document_uploader import DocumentProcessor

app = FastAPI(title="Document Query Assistant", version="1.0.0")

def preprocess_question(question: str) -> str:
    """Preprocess question for better accuracy"""
    # Common question patterns and their optimized versions
    question_lower = question.lower()
    
    # Map common variations to standard terms
    if any(term in question_lower for term in ['sum insured', 'coverage amount', 'maximum coverage']):
        return question.replace('coverage amount', 'sum insured').replace('maximum coverage', 'sum insured')
    elif any(term in question_lower for term in ['waiting period', 'wait time', 'waiting time']):
        return question.replace('wait time', 'waiting period').replace('waiting time', 'waiting period')
    elif any(term in question_lower for term in ['exclusion', 'not covered', 'excluded']):
        return question.replace('not covered', 'exclusions').replace('excluded', 'exclusions')
    
    return question

async def generate_hackrx_answer(question: str, clauses: List[Clause]) -> str:
    """Generate concise, direct answer for HackRX format - OPTIMIZED"""
    
    # Filter clauses by relevance score for better accuracy
    relevant_clauses = [clause for clause in clauses if clause.score > 0.7][:3]
    if not relevant_clauses:
        relevant_clauses = clauses[:3]  # Fallback to top 3
    
    # Prepare optimized clauses context
    clauses_context = "\n".join([
        f"{i+1}. {clause.content[:400]}..." if len(clause.content) > 400 else f"{i+1}. {clause.content}"
        for i, clause in enumerate(relevant_clauses)
    ])
    
    # HACKATHON-OPTIMIZED prompt for maximum scoring with structured format
    prompt = f"""You are an expert insurance policy analyst. Answer this question using ONLY the policy information provided.

POLICY INFORMATION:
{clauses_context}

QUESTION: {question}

CRITICAL INSTRUCTIONS:
- Provide a comprehensive, detailed answer based ONLY on the policy text
- Include ALL relevant details: numbers, percentages, amounts, timeframes, conditions
- Structure the answer as a complete, flowing paragraph
- Include specific policy references and conditions when available
- If multiple aspects exist, cover all of them thoroughly
- Only say "Not specified in policy" if truly no relevant information exists
- Be thorough and professional like an insurance expert

DETAILED ANSWER:"""

    try:
        # HACKATHON-OPTIMIZED generation parameters for comprehensive answers
        generation_config = {
            'temperature': 0.02,  # Ultra-low for maximum consistency
            'top_p': 0.95,
            'top_k': 5,
            'max_output_tokens': 400,  # Increased for detailed comprehensive answers
        }
        
        response = gemini_service.model.generate_content(
            prompt,
            generation_config=generation_config
        )
        answer = response.text.strip()
        
        # Clean and validate answer
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        
        # Ensure answer is not empty
        if not answer or len(answer.strip()) < 3:
            return "Not specified in policy"
        
        return answer
        
    except Exception as e:
        return "Not specified in policy"

# Hackathon-specific models
class HackRXRequest(BaseModel):
    documents: str  # URL to the PDF document
    questions: List[str]  # List of questions to ask

class HackRXResponse(BaseModel):
    answers: List[str]  # List of direct answers

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process document query and return structured response"""
    try:
        # Check if Gemini service is available
        if not gemini_service:
            raise HTTPException(status_code=503, detail="Gemini service not available")
        
        # Generate cache key if Redis is available
        cached_response = None
        if redis_service:
            clause_ids = [clause.clause_id for clause in request.top_k_clauses]
            query_hash = redis_service.generate_query_hash(request.user_question, clause_ids)
            cached_response = redis_service.get_cached_response(query_hash)
        
        # Return cached response if available
        if cached_response:
            print("✅ Returning cached response")
            return cached_response
        
        # Process with Gemini if not cached
        response = gemini_service.analyze_clauses(request.user_question, request.top_k_clauses)
        
        # Cache the response if Redis is available
        if redis_service:
            redis_service.cache_response(query_hash, response)
            print("✅ Response cached for future requests")
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Document Query Assistant"}

@app.get("/test-redis")
async def test_redis():
    """Test Redis connection endpoint"""
    if not redis_service:
        raise HTTPException(status_code=503, detail="Redis service not available")
    
    result = redis_service.test_connection()
    if result["status"] == "error":
        raise HTTPException(status_code=503, detail=result["message"])
    return result

@app.get("/cache/stats")
async def get_cache_stats():
    """Get document cache statistics"""
    if not redis_service:
        raise HTTPException(status_code=503, detail="Redis service not available")
    
    try:
        stats = redis_service.get_document_cache_stats()
        return {
            "status": "success",
            "cache_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cache stats: {str(e)}")

@app.delete("/cache/clear")
async def clear_cache():
    """Clear document cache"""
    if not redis_service:
        raise HTTPException(status_code=503, detail="Redis service not available")
    
    try:
        success = redis_service.clear_document_cache()
        if success:
            return {"status": "success", "message": "Document cache cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear document cache")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@app.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(request: HackRXRequest):
    """HackRX endpoint - Process document from URL and answer multiple questions"""
    try:
        print(f"🚀 HackRX request received with {len(request.questions)} questions")
        
        # Check services availability
        if not gemini_service:
            raise HTTPException(status_code=503, detail="Gemini service not available")
        if not pinecone_service or not pinecone_service.index:
            raise HTTPException(status_code=503, detail="Pinecone service not available")
        
        # Check if document is already cached in Redis
        document_name = "hackrx_policy"
        
        if redis_service and redis_service.is_document_cached(request.documents):
            print("🚀 Document found in Redis cache - using cached version!")
            cached_info = redis_service.get_cached_document_info(request.documents)
            print(f"📋 Cached document: {cached_info['document_name']} ({cached_info['clause_count']} clauses)")
            
            # Check if embeddings are uploaded to Pinecone
            if not redis_service.are_embeddings_uploaded(request.documents):
                print("🔄 Embeddings not in Pinecone - uploading from Redis cache...")
                cached_clauses = redis_service.get_cached_clauses(request.documents)
                
                processor = DocumentProcessor()
                success = processor.upload_to_pinecone(cached_clauses)
                if success:
                    redis_service.mark_embeddings_uploaded(request.documents)
                    print("✅ Cached clauses uploaded to Pinecone")
                else:
                    raise HTTPException(status_code=500, detail="Failed to upload cached clauses to vector database")
            else:
                print("✅ Embeddings already in Pinecone - ready to query!")
        
        else:
            print("📥 Document not cached - downloading and processing...")
            print(f"🔗 URL: {request.documents}")
            
            # Download the PDF
            response = requests.get(request.documents, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            try:
                # Process the document
                print("🔄 Processing document...")
                processor = DocumentProcessor()
                
                # Extract text and create clauses
                text = processor.extract_text_from_file(temp_file_path)
                clauses = processor.split_document_into_clauses(text, document_name)
                
                print(f"✅ Document processed into {len(clauses)} clauses")
                
                # Cache the processed document in Redis
                if redis_service:
                    file_size = len(response.content)
                    cache_success = redis_service.cache_document(
                        request.documents, 
                        document_name, 
                        clauses, 
                        file_size, 
                        text
                    )
                    
                    if cache_success:
                        print("✅ Document cached in Redis for future requests")
                    else:
                        print("⚠️ Failed to cache document in Redis (continuing anyway)")
                
                # Upload to Pinecone
                success = processor.upload_to_pinecone(clauses)
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to upload document to vector database")
                
                # Mark embeddings as uploaded in Redis
                if redis_service:
                    redis_service.mark_embeddings_uploaded(request.documents)
                print("✅ Document uploaded to vector database")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        # Process questions in parallel for maximum speed
        import asyncio
        
        async def process_single_question(question: str, question_index: int) -> str:
            """Process a single question asynchronously - OPTIMIZED"""
            
            try:
                # Preprocess question for better accuracy
                processed_question = preprocess_question(question)
                
                # Check semantic cache first for instant responses
                if redis_service:
                    cached_answer = redis_service.get_semantic_cached_response(processed_question, document_name)
                    if cached_answer:
                        return cached_answer
                
                # Generate embedding for the processed question (optimized)
                genai.configure(api_key=config.GEMINI_API_KEY)
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=processed_question,
                    task_type="retrieval_query"
                )
                query_embedding = result['embedding']
                
                # HACKATHON-OPTIMIZED search for maximum accuracy
                search_results = pinecone_service.index.query(
                    vector=query_embedding,
                    top_k=7,  # Increased for better context on unknown documents
                    include_metadata=True,
                    filter={"document": document_name}
                )
                
                if not search_results.matches:
                    return "Not specified in policy"
                
                # Convert to Clause objects (optimized)
                relevant_clauses = [
                    Clause(
                        clause_id=match.id,
                        content=match.metadata.get('content', ''),
                        score=match.score
                    )
                    for match in search_results.matches
                ]
                
                # Generate answer with optimized function
                answer = await generate_hackrx_answer(question, relevant_clauses)
                
                # Cache the answer for future requests
                if redis_service:
                    redis_service.cache_semantic_response(question, document_name, answer)
                
                return answer
                
            except Exception as e:
                return "Error processing question"
        
        # Process all questions concurrently with increased parallelism for speed
        semaphore = asyncio.Semaphore(5)  # Increased for faster processing
        
        async def process_with_semaphore(question: str, index: int) -> str:
            async with semaphore:
                return await process_single_question(question, index + 1)
        
        # Create tasks for all questions
        tasks = [
            process_with_semaphore(question, i) 
            for i, question in enumerate(request.questions)
        ]
        
        # Execute all tasks concurrently with timeout for reliability
        try:
            answers = await asyncio.wait_for(
                asyncio.gather(*tasks), 
                timeout=30.0  # 30 second timeout for all questions
            )
        except asyncio.TimeoutError:
            # Fallback: process questions sequentially if timeout
            answers = []
            for i, question in enumerate(request.questions):
                try:
                    answer = await asyncio.wait_for(
                        process_single_question(question, i + 1),
                        timeout=5.0  # 5 second timeout per question
                    )
                    answers.append(answer)
                except asyncio.TimeoutError:
                    answers.append("Response timeout - please try again")
        
        return HackRXResponse(answers=answers)
                
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
    except Exception as e:
        print(f"❌ HackRX processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
