from fastapi import FastAPI, HTTPException
import json
import httpx
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
    """Advanced question preprocessing for production-ready accuracy"""
    question_lower = question.lower()
    
    # Enhanced keyword mapping for better search accuracy
    keyword_mappings = {
        # Coverage terms
        ('sum insured', 'coverage amount', 'maximum coverage', 'coverage limit'): 'sum insured',
        ('waiting period', 'wait time', 'waiting time', 'waiting duration'): 'waiting period',
        ('exclusion', 'not covered', 'excluded', 'exclusions'): 'exclusions',
        
        # Medical terms
        ('ayush', 'ayurveda', 'yoga', 'naturopathy', 'unani', 'siddha', 'homeopathy'): 'AYUSH treatments',
        ('room rent', 'room charges', 'accommodation charges'): 'room rent',
        ('icu', 'intensive care', 'critical care'): 'ICU charges',
        ('maternity', 'pregnancy', 'childbirth', 'delivery'): 'maternity expenses',
        ('cataract', 'eye surgery', 'lens replacement'): 'cataract surgery',
        ('organ donor', 'transplant', 'organ donation'): 'organ donor coverage',
        
        # Policy terms
        ('grace period', 'premium payment period'): 'grace period',
        ('no claim discount', 'ncd', 'bonus'): 'no claim discount',
        ('health checkup', 'preventive care', 'wellness'): 'health checkup',
        ('hospital definition', 'what is hospital'): 'hospital definition',
        ('pre-existing', 'ped', 'existing disease'): 'pre-existing diseases'
    }
    
    # Apply mappings
    for terms, replacement in keyword_mappings.items():
        if any(term in question_lower for term in terms):
            for term in terms:
                question = question.replace(term, replacement)
    
    return question

def generate_search_keywords(question: str) -> List[str]:
    """Generate multiple search keywords for comprehensive document search"""
    question_lower = question.lower()
    keywords = []
    
    # Base question
    keywords.append(question)
    
    # Extract key terms based on question type
    if 'ayush' in question_lower or 'ayurveda' in question_lower:
        keywords.extend(['AYUSH', 'Ayurveda', 'Yoga', 'Naturopathy', 'Unani', 'Siddha', 'Homeopathy', 'alternative medicine'])
    
    if 'room rent' in question_lower or 'icu' in question_lower:
        keywords.extend(['room rent', 'ICU charges', 'accommodation', 'intensive care', 'daily charges', 'sub-limits'])
    
    if 'grace period' in question_lower:
        keywords.extend(['grace period', 'premium payment', 'due date', 'renewal'])
    
    if 'waiting period' in question_lower:
        keywords.extend(['waiting period', 'exclusion period', 'coverage waiting'])
    
    if 'maternity' in question_lower:
        keywords.extend(['maternity', 'pregnancy', 'childbirth', 'delivery', 'newborn'])
    
    if 'exclusion' in question_lower:
        keywords.extend(['exclusions', 'not covered', 'limitations', 'restrictions'])
    
    return keywords

async def generate_hackrx_answer(question: str, clauses: List[Clause]) -> str:
    """Generate concise, direct answer for HackRX format - OPTIMIZED"""
    
    # HACKATHON-WINNING: Use more clauses for comprehensive answers
    relevant_clauses = clauses[:5]  # Use top 5 clauses for maximum context
    
    # Prepare comprehensive clauses context with full content
    clauses_context = "\n".join([
        f"Clause {i+1}: {clause.content}"
        for i, clause in enumerate(relevant_clauses)
    ])
    
    # HACKATHON-WINNING prompt optimized for maximum accuracy and completeness
    prompt = f"""You are an expert insurance policy analyst. Analyze the policy information and provide a comprehensive answer.

POLICY INFORMATION:
{clauses_context}

QUESTION: {question}

CRITICAL INSTRUCTIONS FOR WINNING RESPONSES:
- Extract ALL relevant information from the policy text
- Include specific numbers, percentages, amounts, timeframes, and conditions
- Cover all aspects of the question comprehensively
- Mention eligibility criteria, limitations, and exceptions when applicable
- Structure as a complete, professional paragraph
- If information spans multiple clauses, synthesize them into one coherent answer
- Include procedural requirements and compliance conditions
- Only say "Not specified in policy" if absolutely no relevant information exists
- Be thorough like a senior insurance expert providing definitive guidance

COMPREHENSIVE EXPERT ANSWER:"""

    try:
        # HACKATHON-WINNING generation parameters for comprehensive answers
        generation_config = {
            'temperature': 0.01,  # Ultra-low for maximum consistency and accuracy
            'top_p': 0.98,
            'top_k': 3,
            'max_output_tokens': 500,  # Maximum tokens for comprehensive answers
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
    """PRODUCTION-GRADE: Real-time PDF analysis - Each PDF processed independently"""
    try:
        print(f"🚀 REAL-TIME REQUEST: Processing {len(request.questions)} questions")
        
        # Check services availability
        if not gemini_service:
            raise HTTPException(status_code=503, detail="Gemini service not available")
        if not pinecone_service or not pinecone_service.index:
            raise HTTPException(status_code=503, detail="Pinecone service not available")
        
        # SMART-CACHE: reuse vectors & clauses if this PDF has been processed before
        import hashlib, time, uuid

        document_name: str | None = None
        clauses: List[Clause] | None = None

        if redis_service and redis_service.is_document_cached(request.documents):
            info = redis_service.get_cached_document_info(request.documents)
            cached = redis_service.get_cached_clauses(request.documents)
            if info and cached:
                document_name = info.get("document_name")
                def _cached_to_clause(c: dict) -> Clause:
                    return Clause(
                        clause_id=c.get('clause_id') or c.get('id'),
                        content=c.get('content', ''),
                        score=c.get('score', 1.0)
                    )
                clauses = [_cached_to_clause(c) for c in cached]
                print(f"✅ Using cached document {document_name} with {len(clauses)} clauses")
                # Ensure embeddings exist in Pinecone
                if not redis_service.are_embeddings_uploaded(request.documents):
                    print("📤 Uploading cached clauses to Pinecone (first time)…")
                    DocumentProcessor().upload_to_pinecone(clauses)
                    redis_service.mark_embeddings_uploaded(request.documents)

        if clauses is None:
            # First-time processing path
            url_hash = hashlib.md5(request.documents.encode()).hexdigest()[:8]
            document_name = f"doc_{url_hash}"
            print(f"🆕 Processing new document {document_name}")
            print(f"🔗 PDF URL: {request.documents}")
            print("⚡ DOWNLOADING PDF…")
            async with httpx.AsyncClient(timeout=30) as client:
                pdf_response = await client.get(request.documents)
            pdf_response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_response.content)
                temp_file_path = temp_file.name

            try:
                processor = DocumentProcessor()
                text = processor.extract_text_from_file(temp_file_path)
                clauses = processor.split_document_into_clauses(text, document_name)
                print(f"✅ Extracted {len(clauses)} clauses from new document")

                processor.upload_to_pinecone(clauses)

                if redis_service:
                    redis_service.cache_document(request.documents, document_name,
                                                 [(c.model_dump() if hasattr(c, 'model_dump') else c) for c in clauses],
                                                 file_size=len(pdf_response.content))
                    redis_service.mark_embeddings_uploaded(request.documents)
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        # At this point, clauses for this PDF are guaranteed to be embedded and searchable in Pinecone.
        # Proceed to answer the user’s questions.
            


        
        # Process questions in parallel for maximum speed
        import asyncio
        
        async def process_single_question(question: str, question_index: int) -> str:
            """PRODUCTION-READY: Process single question with advanced accuracy and real-time performance"""
            
            try:
                # Advanced question preprocessing for better accuracy
                processed_question = preprocess_question(question)
                
                # PRODUCTION: NO CACHE LOOKUP - Always process fresh for real-time accuracy
                print(f"🔍 REAL-TIME PROCESSING: Question {question_index} - {processed_question[:50]}...")
                
                # Generate multiple search strategies for comprehensive results
                search_keywords = generate_search_keywords(processed_question)
                
                # PRODUCTION: Multi-embedding search for maximum accuracy
                genai.configure(api_key=config.GEMINI_API_KEY)
                
                all_relevant_clauses = []
                seen_content = set()
                
                # Search with multiple keyword variations
                for keyword in search_keywords[:3]:  # Limit to top 3 for performance
                    try:
                        result = genai.embed_content(
                            model="models/embedding-001",
                            content=keyword,
                            task_type="retrieval_query"
                        )
                        query_embedding = result['embedding']
                        
                        # Enhanced search with dynamic top_k based on question complexity
                        top_k = 20 if any(term in processed_question.lower() for term in ['ayush', 'room rent', 'icu', 'sub-limits']) else 15
                        
                        search_results = pinecone_service.index.query(
                            vector=query_embedding,
                            top_k=top_k,
                            include_metadata=True,
                            filter={"document": document_name}
                        )
                        
                        # Collect unique, high-quality clauses
                        for match in search_results.matches:
                            content = match.metadata.get('content', '')
                            if content and content not in seen_content and match.score > 0.6:
                                clause = Clause(
                                    clause_id=match.id,
                                    content=content,
                                    score=match.score
                                )
                                all_relevant_clauses.append(clause)
                                seen_content.add(content)
                    
                    except Exception as search_error:
                        print(f"Search error for keyword '{keyword}': {search_error}")
                        continue
                
                # If no results from multi-search, fallback to original question
                if not all_relevant_clauses:
                    try:
                        result = genai.embed_content(
                            model="models/embedding-001",
                            content=processed_question,
                            task_type="retrieval_query"
                        )
                        query_embedding = result['embedding']
                        
                        search_results = pinecone_service.index.query(
                            vector=query_embedding,
                            top_k=10,
                            include_metadata=True,
                            filter={"document": document_name}
                        )
                        
                        all_relevant_clauses = [
                            Clause(
                                clause_id=match.id,
                                content=match.metadata.get('content', ''),
                                score=match.score
                            )
                            for match in search_results.matches
                        ]
                    except Exception as fallback_error:
                        print(f"Fallback search error: {fallback_error}")
                        return "Error processing question - please try again"
                
                if not all_relevant_clauses:
                    return "Not specified in policy"
                
                # Sort by relevance score and take top clauses
                all_relevant_clauses.sort(key=lambda x: x.score, reverse=True)
                top_clauses = all_relevant_clauses[:7]  # Use top 7 for comprehensive answers
                
                # PRODUCTION: Validate answer matches question before returning
                answer = await generate_hackrx_answer(processed_question, top_clauses)
                
                # Answer validation to prevent mismatched responses
                if validate_answer_matches_question(processed_question, answer):
                    return answer
                else:
                    # If validation fails, try with different clauses
                    if len(all_relevant_clauses) > 7:
                        alternative_clauses = all_relevant_clauses[3:10]  # Try different clause set
                        alternative_answer = await generate_hackrx_answer(processed_question, alternative_clauses)
                        if validate_answer_matches_question(processed_question, alternative_answer):
                            return alternative_answer
                    
                    return "Not specified in policy"
                
            except Exception as e:
                print(f"Error processing question {question_index}: {e}")
                return "Error processing question - please try again"

        def validate_answer_matches_question(question: str, answer: str) -> bool:
            """PRODUCTION: Validate that answer is relevant to the question asked"""
            question_lower = question.lower()
            answer_lower = answer.lower()
            
            # Check for obvious mismatches
            if "grace period" in question_lower and "grace period" not in answer_lower and "premium payment" not in answer_lower:
                return False
            
            if "ayush" in question_lower and "ayush" not in answer_lower and "ayurveda" not in answer_lower:
                return False
            
            if "room rent" in question_lower and "room" not in answer_lower and "accommodation" not in answer_lower:
                return False
            
            if "icu" in question_lower and "icu" not in answer_lower and "intensive care" not in answer_lower:
                return False
            
            if "maternity" in question_lower and "maternity" not in answer_lower and "pregnancy" not in answer_lower:
                return False
            
            if "waiting period" in question_lower and "waiting" not in answer_lower and "period" not in answer_lower:
                return False
            
            # Check for generic error responses
            if answer_lower.startswith("the provided policy excerpt states that if the premium"):
                return "grace period" in question_lower or "premium" in question_lower
            
            return True

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
                
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
    except Exception as e:
        print(f"❌ HackRX processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
