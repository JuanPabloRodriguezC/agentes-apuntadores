import time
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.evaluation import load_evaluator
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("apuntes-ia")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create vector store
vectorstore_parrafos = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",
    namespace="parrafos"
)

vectorstore_frases = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",
    namespace="frases"
)

import time

def measure_retrieval_speed(vectorstore, query, k=5, num_trials=10):
    """Measure average retrieval time"""
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        results = vectorstore.similarity_search(query, k=k)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean': sum(times) / len(times),
        'min': min(times),
        'max': max(times)
    }

# Compare
test_queries = ["tu lista de preguntas representativas"]
for query in test_queries:
    parrafos_time = measure_retrieval_speed(vectorstore_parrafos, query)
    frases_time = measure_retrieval_speed(vectorstore_frases, query)
    print(f"Query: {query}")
    print(f"Párrafos: {parrafos_time['mean']:.4f}s")
    print(f"Frases: {frases_time['mean']:.4f}s")

def evaluate_precision(vectorstore, test_cases, k=5):
    """
    test_cases: list of dicts with 'query' and 'relevant_doc_ids'
    """
    precision_scores = []
    
    for case in test_cases:
        query = case['query']
        relevant_ids = set(case['relevant_doc_ids'])
        
        # Retrieve documents
        results = vectorstore.similarity_search(query, k=k)
        retrieved_ids = set([doc.metadata.get('id') for doc in results])
        
        # Calculate precision@k
        relevant_retrieved = len(retrieved_ids.intersection(relevant_ids))
        precision = relevant_retrieved / k if k > 0 else 0
        precision_scores.append(precision)
    
    return sum(precision_scores) / len(precision_scores)

# Example test cases (you need to create these)
test_cases = [
    {
        'query': '¿Qué es una red neuronal?',
        'relevant_doc_ids': ['doc_123', 'doc_456']  # IDs you know are relevant
    },
    # ... more cases
]

precision_parrafos = evaluate_precision(vectorstore_parrafos, test_cases)
precision_frases = evaluate_precision(vectorstore_frases, test_cases)


from langchain.evaluation import load_evaluator

def evaluate_rag_quality(vectorstore, test_questions_answers):
    """
    test_questions_answers: list of dicts with 'question' and 'expected_answer'
    """
    # Context relevance evaluator
    relevance_evaluator = load_evaluator("labeled_score_string", 
                                         criteria="relevance")
    
    scores = []
    for case in test_questions_answers:
        query = case['question']
        
        # Get context from vectorstore
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Evaluate if context is relevant to answer the question
        eval_result = relevance_evaluator.evaluate_strings(
            prediction=context,
            input=query,
            reference=case.get('expected_answer', '')
        )
        scores.append(eval_result['score'])
    
    return sum(scores) / len(scores)


import pandas as pd

def comprehensive_comparison(test_queries, test_cases=None):
    results = []
    
    for query in test_queries:
        # Speed
        time_parrafos = measure_retrieval_speed(vectorstore_parrafos, query, num_trials=5)
        time_frases = measure_retrieval_speed(vectorstore_frases, query, num_trials=5)
        
        # Retrieve documents
        docs_parrafos = vectorstore_parrafos.similarity_search_with_score(query, k=5)
        docs_frases = vectorstore_frases.similarity_search_with_score(query, k=5)
        
        results.append({
            'query': query,
            'parrafos_speed': time_parrafos['mean'],
            'frases_speed': time_frases['mean'],
            'parrafos_avg_score': sum([score for _, score in docs_parrafos]) / len(docs_parrafos),
            'frases_avg_score': sum([score for _, score in docs_frases]) / len(docs_frases),
            'parrafos_top_score': docs_parrafos[0][1] if docs_parrafos else 0,
            'frases_top_score': docs_frases[0][1] if docs_frases else 0,
        })
    
    return pd.DataFrame(results)

# Run comparison
test_queries = [
    "¿Qué es el backpropagation?",
    "Explica el concepto de overfitting",
    # ... más preguntas
]

df_results = comprehensive_comparison(test_queries)
print(df_results)
df_results.to_csv('comparison_results.csv', index=False)