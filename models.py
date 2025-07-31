from pydantic import BaseModel
from typing import List

class Clause(BaseModel):
    clause_id: str
    content: str
    score: float

class QueryRequest(BaseModel):
    user_question: str
    top_k_clauses: List[Clause]

class QueryResponse(BaseModel):
    answer: str
    conditions: List[str]
    clause_id: str
    score: float
    explanation: str