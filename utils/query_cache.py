import os
import json
import logging
from rapidfuzz import process, fuzz
from google import genai
from google.genai import types
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=os.getenv("SUMMARIZER_API_KEY"))

# Initialize the Chroma client
client = chromadb.PersistentClient()

# Get the collection
collection = client.get_collection(name="fin_doc_summaries", embedding_function=google_ef)

json_file = "companies.json"
with open(json_file, "r") as f:
    companies = json.load(f)
    
# Create a lookup dictionary
lookup = {}
for company in companies:
    for name in [company["security_name"], company["short_name"], company["ticker_symbol"]]:
        lookup[name.lower()] = company["short_name"]  # Normalize case
        
def fuzzy_match_company(query):
    """Try to match company names using fuzzy matching."""
    logger.info(f"Attempting fuzzy match for query: {query}")
    matches = {}
    for name in lookup.keys():
        score = fuzz.partial_ratio(query.lower(), name)
        if score > 80:  # Tune threshold
            matches[name] = lookup[name]
            logger.info(f"Found match: {name} with score {score}")
    
    matches_list = list(set(matches.values()))
    logger.info(f"Fuzzy matches found: {matches_list}")
    return matches_list

def extract_companies_with_llm(query):
    """Fallback method: Use an LLM to extract company names."""
    logger.info(f"Attempting LLM extraction for query: {query}")
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
    logger.info(f"LLM extracted names: {extracted_names}")
    
    # convert the extracted names to a list
    extracted_names = extracted_names.split(", ")
    return extracted_names

def get_companies_from_query(query):
    """Hybrid approach: Fuzzy matching first, then fallback to LLM."""
    logger.info(f"Getting companies from query: {query}")
    companies_found = fuzzy_match_company(query)

    if not companies_found:  # If no match, try LLM
        logger.info("No fuzzy match found, trying LLM")
        llm_extracted = extract_companies_with_llm(query)
        companies_found = [lookup.get(name.lower()) for name in llm_extracted if name.lower() in lookup]
        logger.info(f"LLM matches found: {companies_found}")

    return companies_found

def query_chromadb_sorted(collection, query, n_results=5):
    """
    Queries ChromaDB, retrieves matching documents, sorts them by year (most recent first),
    and formats the results into a context string for an LLM.

    Args:
        collection: ChromaDB collection object.
        query (str): User's query.
        n_results (int): Number of results to retrieve.

    Returns:
        Tuple[List[Tuple[Dict, str]], str]: 
            - Sorted list of (metadata, document) tuples.
            - A formatted context string with sorted document summaries.
    """
    logger.info(f"Querying ChromaDB with query: {query}")

    # Step 1: Get Company Matches
    company_matches = get_companies_from_query(query)
    logger.info(f"Company matches for query: {company_matches}")

    # Step 2: Query ChromaDB
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={
            "company_name": {"$in": company_matches}
        }
    )
    logger.info(f"ChromaDB query results count: {len(results.get('documents', [[]])[0])}")

    # Step 2: Extract Metadata & Documents
    metadata_results = results.get('metadatas', [])
    document_results = results.get('documents', [])
    logger.info(f"Metadata results count: {len(metadata_results[0]) if metadata_results else 0}")
    logger.info(f"Document results count: {len(document_results[0]) if document_results else 0}")

    # Step 3: Flatten Nested Metadata & Document Lists
    flattened_metadata = [item for sublist in metadata_results for item in sublist]
    flattened_documents = [item for sublist in document_results for item in sublist]
    logger.info(f"Flattened metadata count: {len(flattened_metadata)}")
    logger.info(f"Flattened documents count: {len(flattened_documents)}")

    # Step 4: Sort Metadata & Documents by Year (Descending)
    sorted_results = sorted(
        zip(flattened_metadata, flattened_documents),
        key=lambda pair: int(pair[0]['year']),  # Convert 'year' to int for correct sorting
        reverse=True
    )
    logger.info(f"Sorted results count: {len(sorted_results)}")

    # Step 5: Generate Context String from Sorted Documents
    context = "\n\n".join([doc for _, doc in sorted_results])
    logger.info(f"Generated context length: {len(context)} characters")

    return sorted_results, context

def qa_bot(query: str, context: str):
    """
    A function to answer questions about the financial document summaries.
    
    Args:
        query (str): The question to answer.
        context (str): The context of the financial document summaries.

    Returns:
        str: The answer to the question.
    """
    logger.info(f"QA Bot processing query: {query}")
    logger.info(f"Context length: {len(context)} characters")
    
    client = genai.Client(api_key=os.getenv("CHATBOT_API_KEY"))
    
    # QA System Prompt
    qa_system_prompt = """
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
    
    # Create a prompt for the question and context
    prompt = f"Question: {query}\nContext: {context}"
    logger.info(f"Generated prompt length: {len(prompt)} characters")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",
            config=types.GenerateContentConfig(
                system_instruction=qa_system_prompt,
                temperature=0,
            ),
            contents=[prompt]
        )
        logger.info("Successfully generated response from Gemini")
        return response.text
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise

def qa_workflow(query: str):
    """
    A workflow to answer questions about the financial document summaries.
    """
    logger.info(f"Starting QA workflow for query: {query}")
    
    try:
        # Step 1: Query ChromaDB
        sorted_results, context = query_chromadb_sorted(collection, query)
        logger.info("Successfully retrieved and sorted results from ChromaDB")
        
        # Step 2: Answer the question
        answer = qa_bot(query, context)
        logger.info("Successfully generated answer")
        
        return answer
    except Exception as e:
        logger.error(f"Error in QA workflow: {str(e)}")
        raise


if __name__ == "__main__":
    query = "How well did the EduFocal perform in 20243"
    answer = qa_workflow(query)
    print(answer)