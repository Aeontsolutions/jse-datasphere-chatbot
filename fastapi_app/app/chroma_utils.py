import os
import logging
from typing import List, Dict, Any, Optional
from uuid import uuid4
import json
from rapidfuzz import fuzz
from google import genai
from google.genai import types

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

json_file = "companies.json"
with open(json_file, "r") as f:
    companies = json.load(f)

def init_chroma_client(persist_directory: Optional[str] = None):
    """Initialise and return a Chroma client backed by DuckDB+Parquet."""
    if persist_directory is None:
        persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/chroma_db")

    client = chromadb.PersistentClient(
        path=persist_directory, 
        # settings=settings
        )
    logger.info(f"Chroma client initialised (persist dir='{persist_directory}')")
    return client


def get_or_create_collection(
    client: "chromadb.Client", collection_name: str = "documents"
):
    """Return an existing collection or create a new one using a Google Generative AI embedding function."""
    embed_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=os.getenv("SUMMARIZER_API_KEY")
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine", "hnsw:search_ef": 100},
    )
    return collection


def add_documents(
    collection: "chromadb.Collection",
    documents: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None,
):
    """Add (or upsert) documents to a Chroma collection and return the document IDs used."""
    if ids is None:
        ids = [str(uuid4()) for _ in documents]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    # Persist changes (only the PersistentClient has `persist`, not the Collection)
    try:
        # Access the underlying client if available
        client = getattr(collection, "_client", None)
        if client and hasattr(client, "persist"):
            client.persist()
    except Exception as persist_err:
        # Log a warning but do not fail the request – data is still in memory
        logger.warning(f"Could not persist Chroma collection to disk: {persist_err}")

    logger.info(f"Added {len(ids)} documents to Chroma collection '{collection.name}'")
    return ids
    
# Create a lookup dictionary
lookup = {}
for company in companies:
    for name in [company["security_name"], company["short_name"], company["ticker_symbol"]]:
        lookup[name.lower()] = company["short_name"]  # Normalize case
        
def fuzzy_match_company(query):
    """Try to match company names using fuzzy matching."""
    matches = {}
    for name in lookup.keys():
        score = fuzz.partial_ratio(query.lower(), name)
        if score > 80:  # Tune threshold
            matches[name] = lookup[name]
    
    return list(set(matches.values()))

def extract_companies_with_llm(query):
    """Fallback method: Use an LLM to extract company names."""
    client = genai.Client(api_key=os.getenv("CHATBOT_API_KEY"))
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite-001",
        config=types.GenerateContentConfig(
            system_instruction="Your task is to retrun a comma separated list of the companies mentioned in the user's prompt.",
            temperature=0,
        ),
        contents=[query]
    )
    extracted_names = response.text
    
    # convert the extracted names to a list
    extracted_names = extracted_names.split(", ")
    return extracted_names


def get_companies_from_query(query):
    """Hybrid approach: Fuzzy matching first, then fallback to LLM."""
    companies_found = fuzzy_match_company(query)

    if not companies_found:  # If no match, try LLM
        print("No fuzzy match found, trying LLM")
        llm_extracted = extract_companies_with_llm(query)
        companies_found = [lookup.get(name.lower()) for name in llm_extracted if name.lower() in lookup]

    return companies_found

def get_doctype_from_query(query):
    """Get the document type from the query.
    Args:
        query (str): The query to get the document type from.

    Returns:
        str: The document type.
    """
    
    client = genai.Client(api_key=os.getenv("CHATBOT_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite-001",
        config=types.GenerateContentConfig(
            system_instruction="""
                You are a document classification assistant helping to identify the most relevant source type for answering user queries: financial documents, non-financial documents, or both.

                Your task is to analyze the user's query and determine **which type of document(s)** would best provide a useful and accurate response, in order to minimize irrelevant or noisy context.

                Respond in the following format:

                    Justification: <your justification>
                    Label: <financial | non-financial | both>

                Where:
                - Your **justification** explains your reasoning clearly and concisely.
                - Your **response** must be one of the following (exact match):  
                - 'financial' → for queries that rely on metrics, earnings, ratios, cash flows, or balance sheet information.  
                - 'non-financial' → for queries that depend on business strategy, ESG initiatives, leadership tone, risks, or narrative insights.  
                - 'both' → for queries that require a combination of numeric financial data *and* strategic or qualitative context.

                Think carefully and respond accurately to ensure the correct documents are used to answer the query.
                
                For example:
                Query: "What is the revenue growth of Company X in 2023?"
                Justification: "The query is about financial metrics and data."
                Label: "financial"
                
                Query: "What is the strategic direction of Company Y?"
                Justification: "The query is about business strategy and non-financial information."
                Label: "non-financial"
                
                Query: "Summarize Company Z's performance in 2023."
                Justification: "Summarizing performance typically includes both financial results (e.g., revenue, profit) and qualitative drivers (e.g., market conditions, management commentary)."
                Label: "both"
                """,
            temperature=0,
        ),
        contents=[query]
    )
    last_line = response.text.lower().strip().split("\n")[-1].strip('"')
    doctype_map = {
        "label: both": ["financial", "non-financial"],
        "label: non-financial": ["non-financial"],
        "label: financial": ["financial"]
    }
    return doctype_map.get(last_line, ["unknown"])

def query_collection(
    collection: "chromadb.Collection",
    query: str,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
):
    """Query a Chroma collection and return the raw result dict."""

    # Step 1: Get Company Matches
    company_matches = get_companies_from_query(query)
    doctype = get_doctype_from_query(query)

    # Step 2: Get Document Matches
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={
            "$and": [
                {"company_name": {"$in": company_matches}},
                {"file_type": {"$in": doctype}}
            ]
        }
    )
    
    # Step 3: Flatten Nested Metadata & Document Lists
    metadata_results = results.get('metadatas', [])
    document_results = results.get('documents', [])
    flattened_metadata = [item for sublist in metadata_results for item in sublist]
    flattened_documents = [item for sublist in document_results for item in sublist]
    
    # Step 4: Sort Metadata & Documents by Year (Descending)
    sorted_results = sorted(
        zip(flattened_metadata, flattened_documents),
        key=lambda pair: int(pair[0]["year"]),  # Convert 'year' to int for correct sorting
        reverse=True,
    )

    # Step 5: Generate Context String from Sorted Documents
    context = "\n\n".join([doc for _, doc in sorted_results])

    return sorted_results, context

# ---------------------------------------------------------------------------
# QA Bot Helper (Gemini LLM)
# ---------------------------------------------------------------------------


def qa_bot(query: str, contexts: str):
    """Answer a user question using only the supplied `contexts` string.

    Parameters
    ----------
    query : str
        The question from the user.
    contexts : str
        Concatenated context extracted from ChromaDB that should be the *only*
        source material for the answer.

    Returns
    -------
    str
        The model-generated answer.
    """

    client = genai.Client(api_key=os.getenv("CHATBOT_API_KEY"))

    qa_system_prompt = (
        """
        You are an experienced financial analyst. Your primary task is to answer user questions about financial topics based *solely* on the content of the provided financial document summaries. Your goal is to provide not just factually accurate but also insightful responses that directly address the user's query by synthesizing information and identifying key relationships within the provided documents.

        **Strict Guidelines:**

        * **Insightful Analysis Based on Facts:** You must base your insights and analysis *exclusively* on the information explicitly stated or logically implied within the provided summaries. Aim to connect different pieces of information, identify trends, and explain the significance of the data in relation to the user's question.
        * **No Fabrication or External Information:** Under no circumstances should you make up information, invent scenarios, or bring in knowledge from outside the provided financial document summaries.
        * **Handling Questions Beyond the Summaries:**
            * If the answer to the user's question requires information or analysis not explicitly present or logically derivable from the provided summaries, respond with: "Based on the provided document summaries, I cannot offer a more detailed or insightful analysis on this specific aspect." Then guide the user on how to find the answer.
            * If the question is unrelated to the financial document summaries, respond with: "This question falls outside the scope of the provided financial document summaries, and therefore I cannot offer an insightful response."
        * **Handling Unclear Questions:** If the user's question is ambiguous or lacks sufficient detail to provide an insightful response, politely ask for clarification. For example: "To provide a more insightful analysis, could you please specify which aspect of [topic] you are most interested in?" or "Could you please provide more context regarding [specific element] so I can offer a more insightful perspective based on the documents?"

        **Focus:** Provide concise yet comprehensive answers that directly address the user's query with insights derived solely from the provided financial document summaries. Aim to explain the "why" behind the numbers and trends where the information allows, without making assumptions or introducing external data.
    """
    )

    prompt = f"Question: {query}\nContext: {contexts}"

    response = client.models.generate_content(
        model="gemini-2.5-pro-preview-03-25",
        config=types.GenerateContentConfig(
            system_instruction=qa_system_prompt,
            temperature=0,
        ),
        contents=[prompt],
    )

    return response.text
