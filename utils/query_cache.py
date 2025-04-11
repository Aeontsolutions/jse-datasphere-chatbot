import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz
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

# Load environment variables
dotenv.load_dotenv()

@dataclass
class Config:
    """Configuration class for the query cache."""
    SUMMARIZER_API_KEY: str = os.getenv("SUMMARIZER_API_KEY")
    CHATBOT_API_KEY: str = os.getenv("CHATBOT_API_KEY")
    COMPANIES_JSON_PATH: str = "companies.json"
    CHROMA_COLLECTION_NAME: str = "doc_summaries"
    FUZZY_MATCH_THRESHOLD: int = 80
    DEFAULT_N_RESULTS: int = 5
    GEMINI_MODEL: str = "gemini-2.0-flash-lite-001"
    GEMINI_PRO_MODEL: str = "gemini-2.0-flash-001"

class CompanyMatcher:
    """Handles company name matching and lookup."""
    
    def __init__(self, config: Config):
        self.config = config
        self.lookup: Dict[str, str] = {}
        self._load_companies()
    
    def _load_companies(self) -> None:
        """Load company data from JSON file."""
        try:
            with open(self.config.COMPANIES_JSON_PATH, "r") as f:
                companies = json.load(f)
            
            for company in companies:
                for name in [company["security_name"], company["short_name"], company["ticker_symbol"]]:
                    self.lookup[name.lower()] = company["short_name"]
        except Exception as e:
            logger.error(f"Error loading companies: {str(e)}")
            raise
    
    def fuzzy_match_company(self, query: str) -> List[str]:
        """Try to match company names using fuzzy matching."""
        logger.info(f"Attempting fuzzy match for query: {query}")
        matches = {}
        
        for name in self.lookup.keys():
            score = fuzz.partial_ratio(query.lower(), name)
            if score > self.config.FUZZY_MATCH_THRESHOLD:
                matches[name] = self.lookup[name]
                logger.info(f"Found match: {name} with score {score}")
        
        matches_list = list(set(matches.values()))
        logger.info(f"Fuzzy matches found: {matches_list}")
        return matches_list
    
    def extract_companies_with_llm(self, query: str) -> List[str]:
        """Fallback method: Use an LLM to extract company names."""
        logger.info(f"Attempting LLM extraction for query: {query}")
        
        try:
            client = genai.Client(api_key=self.config.CHATBOT_API_KEY)
            response = client.models.generate_content(
                model=self.config.GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction="Your task is to return a comma separated list of the companies mentioned in the user's prompt.",
                    temperature=0,
                ),
                contents=[query]
            )
            extracted_names = response.text
            logger.info(f"LLM extracted names: {extracted_names}")
            return extracted_names.split(", ")
        except Exception as e:
            logger.error(f"Error in LLM extraction: {str(e)}")
            return []
    
    def get_companies_from_query(self, query: str) -> List[str]:
        """Hybrid approach: Fuzzy matching first, then fallback to LLM."""
        logger.info(f"Getting companies from query: {query}")
        companies_found = self.fuzzy_match_company(query)

        if not companies_found:
            logger.info("No fuzzy match found, trying LLM")
            llm_extracted = self.extract_companies_with_llm(query)
            companies_found = [self.lookup.get(name.lower()) for name in llm_extracted if name.lower() in self.lookup]
            logger.info(f"LLM matches found: {companies_found}")

        return companies_found

class DocumentClassifier:
    """Handles document type classification."""
    
    def __init__(self, config: Config):
        self.config = config
        self.doctype_map = {
            "label: both": ["financial", "non-financial"],
            "label: non-financial": ["non-financial"],
            "label: financial": ["financial"]
        }
    
    def get_doctype_from_query(self, query: str) -> List[str]:
        """Get the document type from the query."""
        try:
            client = genai.Client(api_key=self.config.CHATBOT_API_KEY)
            response = client.models.generate_content(
                model=self.config.GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=self._get_classification_prompt(),
                    temperature=0,
                ),
                contents=[query]
            )
            last_line = response.text.lower().strip().split("\n")[-1].strip('"')
            return self.doctype_map.get(last_line, ["unknown"])
        except Exception as e:
            logger.error(f"Error in document classification: {str(e)}")
            return ["unknown"]
    
    def _get_classification_prompt(self) -> str:
        """Get the system prompt for document classification."""
        return """
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
        """

class ChromaDBManager:
    """Manages ChromaDB operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=self.config.SUMMARIZER_API_KEY
        )
        self.client = chromadb.PersistentClient()
        self.collection = self.client.get_collection(
            name=self.config.CHROMA_COLLECTION_NAME,
            embedding_function=self.google_ef
        )
    
    def query_chromadb_sorted(
        self,
        query: str,
        n_results: int = None
    ) -> Tuple[List[Tuple[Dict, str]], str]:
        """
        Queries ChromaDB, retrieves matching documents, sorts them by year (most recent first),
        and formats the results into a context string for an LLM.
        """
        if n_results is None:
            n_results = self.config.DEFAULT_N_RESULTS
            
        logger.info(f"Querying ChromaDB with query: {query}")
        
        # Get company matches and document type
        company_matcher = CompanyMatcher(self.config)
        classifier = DocumentClassifier(self.config)
        
        company_matches = company_matcher.get_companies_from_query(query)
        doc_type = classifier.get_doctype_from_query(query)
        
        logger.info(f"Company matches: {company_matches}")
        logger.info(f"Document type: {doc_type}")

        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where={
                "$and": [
                    {"company_name": {"$in": company_matches}},
                    {"file_type": {"$in": doc_type}}
                ]
            }
        )

        # Process results
        metadata_results = results.get('metadatas', [])
        document_results = results.get('documents', [])
        
        # Flatten and sort results
        flattened_metadata = [item for sublist in metadata_results for item in sublist]
        flattened_documents = [item for sublist in document_results for item in sublist]
        
        sorted_results = sorted(
            zip(flattened_metadata, flattened_documents),
            key=lambda pair: int(pair[0]['year']),
            reverse=True
        )
        
        # Generate context string
        context = "\n\n".join([doc for _, doc in sorted_results])
        
        return sorted_results, context

class QABot:
    """Handles question answering using LLM."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def qa_bot(self, query: str, context: str) -> str:
        """Answer questions about financial document summaries."""
        logger.info(f"QA Bot processing query: {query}")
        logger.info(f"Context length: {len(context)} characters")
        
        try:
            client = genai.Client(api_key=self.config.CHATBOT_API_KEY)
            
            prompt = f"Question: {query}\nContext: {context}"
            logger.info(f"Generated prompt length: {len(prompt)} characters")

            response = client.models.generate_content(
                model=self.config.GEMINI_PRO_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=self._get_qa_prompt(),
                    temperature=0,
                ),
                contents=[prompt]
            )
            logger.info("Successfully generated response from Gemini")
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _get_qa_prompt(self) -> str:
        """Get the system prompt for QA."""
        return """
            You are an experienced financial analyst. Your primary task is to answer user questions about financial topics based *solely* on the content of the provided financial document summaries. Your goal is to provide not just factually accurate but also insightful responses that directly address the user's query by synthesizing information and identifying key relationships within the provided documents.

            **Strict Guidelines:**

            * **Insightful Analysis Based on Facts:** You must base your insights and analysis *exclusively* on the information explicitly stated or logically implied within the provided summaries. Aim to connect different pieces of information, identify trends, and explain the significance of the data in relation to the user's question.
            * **No Fabrication or External Information:** Under no circumstances should you make up information, invent scenarios, or bring in knowledge from outside the provided financial document summaries.
            * **Most Recent Information:** If the user's question is about the most recent information, you should use the most recent document summaries.
            * **Handling Questions Beyond the Summaries:**
                * If the answer to the user's question requires information or analysis not explicitly present or logically derivable from the provided summaries, respond with: "Based on the provided document summaries, I cannot offer a more detailed or insightful analysis on this specific aspect." Then guide the user on how to find the answer.
                * If the question is unrelated to the financial document summaries, respond with: "This question falls outside the scope of the provided financial document summaries, and therefore I cannot offer an insightful response."
            * **Handling Unclear Questions:** If the user's question is ambiguous or lacks sufficient detail to provide an insightful response, politely ask for clarification.

            **Focus:** Provide concise yet comprehensive answers that directly address the user's query with insights derived solely from the provided financial document summaries.
        """

class QueryEnhancer:
    """Enhances queries with context and conversation history."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def enhance_query_with_context(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Enhance the user query using conversation history and context."""
        logger.info("Enhancing query with context...")
        
        try:
            client = genai.Client(api_key=self.config.CHATBOT_API_KEY)
            
            conversation_context = ""
            if conversation_history and len(conversation_history) > 0:
                recent_history = conversation_history[-10:]
                conversation_context = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in recent_history]
                )
            
            enhancement_prompt = f"""
            You are a financial analyst assistant. Your task is to enhance the user's query to make it more effective for retrieving and analyzing financial and non-financial document information.
            
            Consider the following:
            1. The query will be used to search through financial and non-financial statements and reports
            2. The response should maintain the original intent while being more specific about financial metrics, time periods, and non-financial information
            3. Include relevant financial terminology when appropriate
            4. If the query is about company performance, specify which metrics would be most relevant
            5. If the query is about non-financial information, such as ESG initiatives, leadership tone, risks, or narrative insights, specify which non-financial information would be most relevant
            
            Previous conversation context:
            {conversation_context}
            
            Original query: {query}
            
            Please provide an enhanced version of this query that would be more effective for financial and non-financial document analysis.
            Focus on making it more specific and relevant to financial and non-financial statement analysis.
            """
            
            response = client.models.generate_content(
                model=self.config.GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    temperature=0,
                ),
                contents=[enhancement_prompt]
            )
            
            enhanced_query = response.text.strip()
            logger.info(f"Query enhanced from '{query}' to '{enhanced_query}'")
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Error enhancing query: {str(e)}")
            return query

class AnswerValidator:
    """Validates answers from the QA system."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def answer_found(self, answer: str, query: str) -> bool:
        """Check if the answer to the query was found."""
        try:
            client = genai.Client(api_key=self.config.CHATBOT_API_KEY)
            response = client.models.generate_content(
                model=self.config.GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=self._get_validation_prompt(),
                    temperature=0,
                ),
                contents=[query, answer]
            )
            return "true" in response.text.lower()
        except Exception as e:
            logger.error(f"Error validating answer: {str(e)}")
            return False
    
    def _get_validation_prompt(self) -> str:
        """Get the system prompt for answer validation."""
        return """
            Your job is to check if the answer to the query was found in the context.
            You will be given the query and the answer.
            Analyze the answer and the query and respond with "True" if the answer was found in the context, and "False" otherwise.
            
            Example:
            Query: "Who is the CEO of EduFocal?"
            Answer: "The CEO of EduFocal is John Doe."
            Response: "True"
            
            Example:
            Query: "Who is the CEO of EduFocal?"
            Answer: "Based on the provided document summaries, I cannot offer a more detailed or insightful analysis on this specific aspect."
            Response: "False"
        """

class QAWorkflow:
    """Main workflow for question answering."""
    
    def __init__(self):
        self.config = Config()
        self.chroma_db = ChromaDBManager(self.config)
        self.qa_bot = QABot(self.config)
        self.query_enhancer = QueryEnhancer(self.config)
        self.answer_validator = AnswerValidator(self.config)
    
    def qa_workflow(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Main workflow to answer questions about financial document summaries."""
        logger.info(f"Starting QA workflow for query: {query}")
        
        try:
            # Step 1: Enhance the query with context
            if conversation_history:    
                enhanced_query = self.query_enhancer.enhance_query_with_context(
                    query,
                    conversation_history
                )
            else:
                enhanced_query = query
            logger.info(f"Using enhanced query: {enhanced_query}")
            
            # Step 2: Query ChromaDB
            sorted_results, context = self.chroma_db.query_chromadb_sorted(enhanced_query)
            logger.info("Successfully retrieved and sorted results from ChromaDB")
            
            # Step 3: Answer the question
            answer = self.qa_bot.qa_bot(enhanced_query, context)
            logger.info("Successfully generated answer")
            
            return answer
        except Exception as e:
            logger.error(f"Error in QA workflow: {str(e)}")
            raise

# Initialize the workflow
qa_workflow = QAWorkflow().qa_workflow
answer_found = AnswerValidator(Config()).answer_found

if __name__ == "__main__":
    query = "Compare Edufocal and One on One."
    answer = qa_workflow(query)
    print(answer)