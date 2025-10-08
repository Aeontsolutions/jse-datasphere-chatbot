import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from google.cloud import bigquery
from google.oauth2 import service_account
from app.models import FinancialDataFilters, FinancialDataRecord
import google.generativeai as genai

logger = logging.getLogger(__name__)

def get_row_attr(row, attr):
    # Try exact match first
    if hasattr(row, attr):
        return getattr(row, attr)
    # Fallback: try case-insensitive match
    for candidate in dir(row):
        if candidate.lower() == attr.lower():
            return getattr(row, candidate)
    raise AttributeError(f"Row has no attribute '{attr}' (case-insensitive search failed)")

def safe_float(val):
    try:
        if val is None or val == '' or (isinstance(val, str) and val.strip().lower() in ['nan', 'null', 'none']):
            return None
        return float(val)
    except Exception:
        return None

class FinancialDataManager:
    """
    Manager class for handling financial data queries and processing (BigQuery version)
    """
    def __init__(self):
        # Load BigQuery config from environment
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.dataset = os.getenv("BIGQUERY_DATASET")
        self.table = os.getenv("BIGQUERY_TABLE")
        self.location = os.getenv("BIGQUERY_LOCATION", "US")
        self.metadata: Optional[Dict] = None
        self.model = None
        self.bq_client = None
        
        # Validate required environment variables
        if not all([self.project_id, self.dataset, self.table]):
            missing_vars = []
            if not self.project_id:
                missing_vars.append("GCP_PROJECT_ID")
            if not self.dataset:
                missing_vars.append("BIGQUERY_DATASET")
            if not self.table:
                missing_vars.append("BIGQUERY_TABLE")
            raise ValueError(f"Missing required BigQuery environment variables: {', '.join(missing_vars)}")
        
        self._initialize_ai_model()
        try:
            self._initialize_bigquery_client()
            self.load_metadata_from_bigquery()
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client or load metadata: {e}")
            self.bq_client = None
            self.metadata = None
            # Re-raise the exception so the main app knows initialization failed
            raise

    def _initialize_bigquery_client(self):
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        service_account_info = os.getenv("GCP_SERVICE_ACCOUNT_INFO")
        if credentials_path and os.path.exists(credentials_path):
            self.bq_client = bigquery.Client.from_service_account_json(credentials_path, project=self.project_id, location=self.location)
            logger.info(f"BigQuery client initialized with service account file: {credentials_path}")
        elif service_account_info:
            info = json.loads(service_account_info)
            credentials = service_account.Credentials.from_service_account_info(info)
            self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id, location=self.location)
            logger.info("BigQuery client initialized with service account info from env var.")
        else:
            self.bq_client = bigquery.Client(project=self.project_id, location=self.location)
            logger.info("BigQuery client initialized with default credentials.")

    def _initialize_ai_model(self):
        """Initialize the Gemini AI model"""
        try:
            # Use the same API key configuration as the main app
            api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY') or os.getenv('CHATBOT_API_KEY')
            if api_key:
                logger.info(f"Found API key (length: {len(api_key)}), initializing Gemini AI model...")
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                logger.info("✅ Gemini AI model initialized successfully for financial data queries")
            else:
                logger.warning("❌ GOOGLE_API_KEY/GEMINI_API_KEY/CHATBOT_API_KEY not found, falling back to basic parsing")
                # Log available environment variables for debugging
                env_vars = [key for key in os.environ.keys() if 'API' in key or 'KEY' in key]
                logger.info(f"Available API-related env vars: {env_vars}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini AI model: {e}")
            self.model = None

    def load_metadata_from_bigquery(self):
        """Load metadata (companies, symbols, years, standard_items, associations) from BigQuery."""
        try:
            # Companies
            companies_query = f"""
                SELECT DISTINCT Company FROM `{self.project_id}.{self.dataset}.{self.table}`
            """
            companies = [get_row_attr(row, 'Company') for row in self.bq_client.query(companies_query).result()]
            logger.info(f"Loaded {len(companies)} companies from BigQuery")
            
            # Symbols
            symbols_query = f"""
                SELECT DISTINCT Symbol FROM `{self.project_id}.{self.dataset}.{self.table}`
            """
            symbols = [get_row_attr(row, 'Symbol') for row in self.bq_client.query(symbols_query).result()]
            logger.info(f"Loaded {len(symbols)} symbols from BigQuery")
            
            # Years
            years_query = f"""
                SELECT DISTINCT CAST(Year AS STRING) as Year FROM `{self.project_id}.{self.dataset}.{self.table}`
            """
            years = [get_row_attr(row, 'Year') for row in self.bq_client.query(years_query).result()]
            logger.info(f"Loaded {len(years)} years from BigQuery")
            
            # Standard Items
            items_query = f"""
                SELECT DISTINCT standard_item FROM `{self.project_id}.{self.dataset}.{self.table}`
            """
            standard_items = [get_row_attr(row, 'standard_item') for row in self.bq_client.query(items_query).result()]
            logger.info(f"Loaded {len(standard_items)} standard items from BigQuery")
            # Associations
            # company_to_symbol
            c2s_query = f"""
                SELECT Company, ARRAY_AGG(DISTINCT Symbol IGNORE NULLS) as symbols FROM `{self.project_id}.{self.dataset}.{self.table}` GROUP BY Company
            """
            company_to_symbol = {get_row_attr(row, 'Company'): get_row_attr(row, 'symbols') for row in self.bq_client.query(c2s_query).result()}
            # symbol_to_company
            s2c_query = f"""
                SELECT Symbol, ARRAY_AGG(DISTINCT Company IGNORE NULLS) as companies FROM `{self.project_id}.{self.dataset}.{self.table}` GROUP BY Symbol
            """
            symbol_to_company = {get_row_attr(row, 'Symbol'): get_row_attr(row, 'companies') for row in self.bq_client.query(s2c_query).result()}
            # company_to_years
            c2y_query = f"""
                SELECT Company, ARRAY_AGG(DISTINCT CAST(Year AS STRING) IGNORE NULLS) as years FROM `{self.project_id}.{self.dataset}.{self.table}` GROUP BY Company
            """
            company_to_years = {get_row_attr(row, 'Company'): get_row_attr(row, 'years') for row in self.bq_client.query(c2y_query).result()}
            # company_to_items
            c2i_query = f"""
                SELECT Company, ARRAY_AGG(DISTINCT standard_item IGNORE NULLS) as items FROM `{self.project_id}.{self.dataset}.{self.table}` GROUP BY Company
            """
            company_to_items = {get_row_attr(row, 'Company'): get_row_attr(row, 'items') for row in self.bq_client.query(c2i_query).result()}
            # year_to_companies
            y2c_query = f"""
                SELECT CAST(Year AS STRING) as Year, ARRAY_AGG(DISTINCT Company IGNORE NULLS) as companies FROM `{self.project_id}.{self.dataset}.{self.table}` GROUP BY Year
            """
            year_to_companies = {get_row_attr(row, 'Year'): get_row_attr(row, 'companies') for row in self.bq_client.query(y2c_query).result()}
            # item_to_companies
            i2c_query = f"""
                SELECT standard_item, ARRAY_AGG(DISTINCT Company IGNORE NULLS) as companies FROM `{self.project_id}.{self.dataset}.{self.table}` GROUP BY standard_item
            """
            item_to_companies = {get_row_attr(row, 'standard_item'): get_row_attr(row, 'companies') for row in self.bq_client.query(i2c_query).result()}
            # company_year_to_items
            cy2i_query = f"""
                SELECT Company, CAST(Year AS STRING) as Year, ARRAY_AGG(DISTINCT standard_item IGNORE NULLS) as items FROM `{self.project_id}.{self.dataset}.{self.table}` GROUP BY Company, Year
            """
            company_year_to_items = {}
            for row in self.bq_client.query(cy2i_query).result():
                company = get_row_attr(row, 'Company')
                year = get_row_attr(row, 'Year')
                items = get_row_attr(row, 'items')
                company_year_to_items.setdefault(company, {})[year] = items
            # symbol_year_to_items
            sy2i_query = f"""
                SELECT Symbol, CAST(Year AS STRING) as Year, ARRAY_AGG(DISTINCT standard_item IGNORE NULLS) as items FROM `{self.project_id}.{self.dataset}.{self.table}` GROUP BY Symbol, Year
            """
            symbol_year_to_items = {}
            for row in self.bq_client.query(sy2i_query).result():
                symbol = get_row_attr(row, 'Symbol')
                year = get_row_attr(row, 'Year')
                items = get_row_attr(row, 'items')
                symbol_year_to_items.setdefault(symbol, {})[year] = items
            # year_to_items
            y2i_query = f"""
                SELECT CAST(Year AS STRING) as Year, ARRAY_AGG(DISTINCT standard_item IGNORE NULLS) as items FROM `{self.project_id}.{self.dataset}.{self.table}` GROUP BY Year
            """
            year_to_items = {get_row_attr(row, 'Year'): get_row_attr(row, 'items') for row in self.bq_client.query(y2i_query).result()}
            # Filter out None values and log some sample data
            companies_filtered = [c for c in companies if c is not None]
            symbols_filtered = [s for s in symbols if s is not None]
            years_filtered = [y for y in years if y is not None]
            standard_items_filtered = [i for i in standard_items if i is not None]
            
            logger.info(f"Filtered data - companies: {len(companies_filtered)}/{len(companies)}, symbols: {len(symbols_filtered)}/{len(symbols)}, years: {len(years_filtered)}/{len(years)}, items: {len(standard_items_filtered)}/{len(standard_items)}")
            
            # Compose metadata
            self.metadata = {
                "companies": sorted(companies_filtered),
                "symbols": sorted(symbols_filtered),
                "years": sorted(years_filtered),
                "standard_items": sorted(standard_items_filtered),
                "associations": {
                    "company_to_symbol": company_to_symbol,
                    "symbol_to_company": symbol_to_company,
                    "company_to_years": company_to_years,
                    "company_to_items": company_to_items,
                    "year_to_companies": year_to_companies,
                    "item_to_companies": item_to_companies,
                    "company_year_to_items": company_year_to_items,
                    "symbol_year_to_items": symbol_year_to_items,
                    "year_to_items": year_to_items
                },
                "total_records": None,  # Optionally count(*)
                "last_updated": None
            }
            logger.info("Loaded metadata from BigQuery.")
        except Exception as e:
            logger.error(f"Error loading metadata from BigQuery: {e}")
            self.metadata = None

    def get_conversation_context(self, conversation_history: Optional[List[Dict[str, str]]]) -> str:
        """Get recent conversation context for better understanding of follow-up questions"""
        if not conversation_history:
            return ""
        
        context_items = []
        # Include last 3 exchanges from conversation
        recent_messages = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        for msg in recent_messages:
            context_items.append(f"{msg.get('role', 'unknown')}: {msg.get('content', '')}")
        
        return "\n".join(context_items)
    
    def parse_user_query(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None, last_query_data: Optional[Dict] = None) -> FinancialDataFilters:
        """Use Gemini to parse user query and extract filter parameters with conversation context"""
        
        if not self.model:
            # Fallback to basic parsing if AI model is not available
            return self._fallback_parse_query(query, last_query_data)
        
        # Get conversation context
        conversation_context = self.get_conversation_context(conversation_history)
        
        # Create a detailed context with associations
        associations_context = ""
        if self.metadata and 'associations' in self.metadata:
            # Show symbol-company mappings
            symbol_mappings = []
            symbol_to_company = self.metadata['associations'].get('symbol_to_company', {})
            for symbol, companies in symbol_to_company.items():
                symbol_mappings.append(f"{symbol}: {', '.join(companies)}")
            associations_context = f"""
                Symbol-Company Mappings:
                {chr(10).join(symbol_mappings)}
                """
        
        prompt = f"""
            You are a financial data query parser. Given a user query and metadata about available financial data,
            extract the relevant filter parameters. Consider the conversation history for context.
            
            CONVERSATION HISTORY:
            {conversation_context}
            
            Available metadata:
            - Companies: {', '.join(self.metadata['companies'][:10])}... (and {len(self.metadata['companies'])-10} more)
            - Symbols: {', '.join(self.metadata['symbols'])}
            - Years: {', '.join(self.metadata['years'])}
            - Standard Items: {', '.join(self.metadata['standard_items'])}
            
            {associations_context}
            
            Current User Query: "{query}"
            
            CRITICAL PARSING RULES:
            1. If user mentions a trading symbol (like MDS, SOS, JBG, etc.), put it in the "symbols" array
            2. If user mentions a full company name, put it in the "companies" array
            3. Symbols are typically 2-5 uppercase letters
            4. Empty list [] means "ALL" - return data for all items in that category
            5. Match symbols case-insensitively (sos = SOS, mds = MDS)
            6. CONTEXT AWARENESS: If the user asks follow-up questions like "what about 2022?" or "show me their revenue", 
            refer to the conversation history to understand which companies/symbols/items they're referring to
            7. For pronouns like "it", "them", "their", "this company" - refer to the most recent companies/symbols discussed
            
            Return a JSON object with this EXACT structure:
            {{
                "companies": [],
                "symbols": [],
                "years": [],
                "standard_items": [],
                "interpretation": "",
                "data_availability_note": "",
                "is_follow_up": true/false,
                "context_used": ""
            }}
            
            Return ONLY the JSON object, no markdown formatting, no code blocks, no additional text.
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up the response - remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
            elif response_text.startswith('```'):
                response_text = response_text[3:]  # Remove ```
            
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove ending ```
            
            # Remove any remaining whitespace
            response_text = response_text.strip()
            
            # Parse the cleaned JSON
            result = json.loads(response_text)
            
            # Convert symbols to uppercase for consistency
            if result.get('symbols'):
                result['symbols'] = [s.upper() for s in result['symbols']]

            # Ensure years are all strings
            if result.get('years'):
                result['years'] = [str(y) for y in result['years']]

            # Normalize standard_items synonyms to canonical names
            if result.get('standard_items'):
                metric_synonyms = {
                    'net profit': 'net_profit',
                    'gross profit': 'gross_profit',
                    'revenue': 'revenue',
                    'eps': 'eps',
                    'profit': 'net_profit',
                    'net profit margin': 'net_profit_margin',
                    'gross profit margin': 'gross_profit_margin',
                    'operating profit': 'operating_profit',
                    'operating income': 'operating_profit',
                    'return on equity': 'roe',
                    'return on asset': 'roa',
                    'return on assets': 'roa',
                    'current ratio': 'current_ratio',
                    'debt to equity ratio': 'debt_to_equity_ratio',
                    'efficiency ratio': 'efficiency_ratio',
                }
                result['standard_items'] = [
                    metric_synonyms.get(item.lower(), item.replace(' ', '_'))
                    for item in result['standard_items']
                ]

            # If this is a follow-up and some filters are empty, try to fill from last query
            if result.get('is_follow_up') and last_query_data:
                last_filters = last_query_data.get('filters', {})
                
                # Carry forward companies/symbols if not specified
                if not result.get('companies') and not result.get('symbols'):
                    result['companies'] = last_filters.get('companies', [])
                    result['symbols'] = last_filters.get('symbols', [])
                
                # Carry forward years if not specified
                if not result.get('years') and last_filters.get('years'):
                    result['years'] = last_filters.get('years', [])
                
                # Carry forward standard_items if not specified
                if not result.get('standard_items') and last_filters.get('standard_items'):
                    result['standard_items'] = last_filters.get('standard_items', [])
                
                # If asking about different metrics, keep the same companies/years
                if result.get('standard_items') and not result.get('companies') and not result.get('symbols'):
                    result['companies'] = last_filters.get('companies', [])
                    result['symbols'] = last_filters.get('symbols', [])
                    if not result.get('years'):
                        result['years'] = last_filters.get('years', [])
            
            # Post-process to ensure consistency using associations
            if self.metadata and 'associations' in self.metadata:
                result = self._post_process_filters(result)
            
            return FinancialDataFilters(**result)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Raw response: {response_text if 'response_text' in locals() else 'No response'}")
            return self._fallback_parse_query(query, last_query_data)
            
        except Exception as e:
            logger.error(f"Error parsing query with AI: {str(e)}")
            return self._fallback_parse_query(query, last_query_data)
    
    def _post_process_filters(self, result: Dict) -> Dict:
        """Post-process filters to ensure consistency using associations and metadata"""
        if not self.metadata or 'associations' not in self.metadata:
            return result
        associations = self.metadata['associations']
        # Normalize for case-insensitive matching
        all_companies = set(self.metadata.get('companies', []))
        all_symbols = set(self.metadata.get('symbols', []))
        all_standard_items = set(self.metadata.get('standard_items', []))
        company_to_symbol = associations.get('company_to_symbol', {})
        symbol_to_company = associations.get('symbol_to_company', {})

        # --- Filter symbols: only keep valid ones ---
        if 'symbols' in result and result['symbols']:
            valid_symbols = []
            for symbol in result['symbols']:
                # Accept if exact match (case-insensitive)
                for s in all_symbols:
                    if symbol.upper() == s.upper():
                        valid_symbols.append(s)
                        break
            result['symbols'] = valid_symbols
        # --- Filter companies: only keep valid ones (with partial match fallback) ---
        if 'companies' in result and result['companies']:
            valid_companies = []
            for company in result['companies']:
                found = False
                for c in all_companies:
                    if company.lower() == c.lower():
                        valid_companies.append(c)
                        found = True
                        break
                if not found:
                    # Try partial match
                    for c in all_companies:
                        if company.lower() in c.lower() or c.lower() in company.lower():
                            valid_companies.append(c)
                            found = True
                            break
                if not found:
                    logger.warning(f"Company '{company}' not found")
            result['companies'] = valid_companies

        # --- If symbols are specified, add ALL associated companies (and filter again) ---
        if result.get('symbols'):
            companies = set()
            for symbol in result['symbols']:
                if symbol in symbol_to_company:
                    companies.update(symbol_to_company[symbol])
            if companies:
                # Only keep valid companies
                result['companies'] = [c for c in companies if c in all_companies]
                logger.info(f"Found companies for symbols {result['symbols']}: {', '.join(result['companies'])}")
            elif result['symbols']:
                result['companies'] = []
                logger.error(f"No companies found for symbols: {result['symbols']}")
        # --- If companies are specified, add their symbols (and filter again) ---
        if result.get('companies'):
            symbols = set()
            for company in result['companies']:
                if company in company_to_symbol:
                    symbols.update(company_to_symbol[company])
            if symbols:
                # Only keep valid symbols
                result['symbols'] = [s for s in symbols if s in all_symbols]
                logger.info(f"Found symbols for companies: {result['symbols']}")
            elif result['companies']:
                result['symbols'] = []
                logger.error(f"No symbols found for companies: {result['companies']}")
        
        # --- Filter standard_items: only keep valid ones (case-insensitive) ---
        if 'standard_items' in result and result['standard_items']:
            valid_items = []
            for item in result['standard_items']:
                # Accept if exact match (case-insensitive)
                for s in all_standard_items:
                    if item.lower() == s.lower():
                        valid_items.append(s)  # Keep the original case from metadata
                        break
            result['standard_items'] = valid_items
            if len(valid_items) != len(result['standard_items']):
                logger.warning(f"Some standard_items were not found in metadata")
        
        return result
    
    def _fallback_parse_query(self, query: str, last_query_data: Optional[Dict] = None) -> FinancialDataFilters:
        """Fallback manual parsing when AI is not available"""
        logger.info("Using fallback manual parsing...")
        
        result = {
            "companies": [],
            "symbols": [],
            "years": [],
            "standard_items": [],
            "interpretation": f"Manual parse of: {query}",
            "data_availability_note": "",
            "is_follow_up": False,
            "context_used": ""
        }
        
        # Check for follow-up indicators
        follow_up_indicators = ['what about', 'how about', 'and', 'also', 'their', 'its', 'show me more', 'now show']
        query_lower = query.lower()
        for indicator in follow_up_indicators:
            if indicator in query_lower:
                result['is_follow_up'] = True
                break
        
        # If it's a follow-up, try to use last query data
        if result['is_follow_up'] and last_query_data:
            last_filters = last_query_data.get('filters', {})
            result['companies'] = last_filters.get('companies', [])
            result['symbols'] = last_filters.get('symbols', [])
            result['context_used'] = "Using previous query context"
        
        # Check for symbols in the query
        if self.metadata:
            query_upper = query.upper()
            for symbol in self.metadata['symbols']:
                if symbol in query_upper:
                    result['symbols'].append(symbol)
                    logger.info(f"Found symbol {symbol} in query")
        
        # Check for years
        year_pattern = r'\b(20\d{2})\b'
        years_found = re.findall(year_pattern, query)
        if years_found:
            result['years'] = years_found
            logger.info(f"Found years: {years_found}")
        
        # Check for common metrics
        metrics_map = {
            'revenue': 'revenue',
            'profit': 'net_profit',
            'net profit': 'net_profit',
            'gross profit': 'gross_profit',
            'assets': 'total_assets',
            'eps': 'EPS'
        }
        
        for keyword, metric in metrics_map.items():
            if keyword in query_lower:
                result['standard_items'].append(metric)
                logger.info(f"Found metric: {metric}")
        
        # If we found symbols, get associated companies
        if result['symbols'] and self.metadata and 'associations' in self.metadata:
            companies = set()
            symbol_to_company = self.metadata['associations'].get('symbol_to_company', {})
            for symbol in result['symbols']:
                if symbol in symbol_to_company:
                    companies.update(symbol_to_company[symbol])
            if companies:
                result['companies'] = list(companies)
                logger.info(f"Found companies for symbols: {', '.join(result['companies'])}")
        
        return FinancialDataFilters(**result)
    
    def query_data(self, filters: FinancialDataFilters) -> List[FinancialDataRecord]:
        """Query BigQuery for financial data based on filters."""
        if not self.bq_client:
            raise RuntimeError("BigQuery client not initialized")
        query = f"SELECT Company, Symbol, CAST(Year AS STRING) as Year, standard_item, item, unit_multiplier, item_type, item_name FROM `{self.project_id}.{self.dataset}.{self.table}` WHERE 1=1"
        params = []
        if filters.companies:
            query += " AND Company IN UNNEST(@companies)"
            params.append(bigquery.ArrayQueryParameter("companies", "STRING", filters.companies))
        if filters.symbols:
            query += " AND Symbol IN UNNEST(@symbols)"
            params.append(bigquery.ArrayQueryParameter("symbols", "STRING", filters.symbols))
        if filters.years:
            query += " AND CAST(Year AS STRING) IN UNNEST(@years)"
            params.append(bigquery.ArrayQueryParameter("years", "STRING", filters.years))
        if filters.standard_items:
            # Use case-insensitive matching for standard_items
            query += " AND LOWER(standard_item) IN UNNEST(@items)"
            # Convert all standard_items to lowercase for case-insensitive matching
            lowercase_items = [item.lower() for item in filters.standard_items]
            params.append(bigquery.ArrayQueryParameter("items", "STRING", lowercase_items))
        job_config = bigquery.QueryJobConfig()
        if params:
            job_config.query_parameters = params
        logger.info(f"Executing BigQuery: {query} | Params: {params}")
        try:
            results = self.bq_client.query(query, job_config=job_config).result()
            records = []
            for row in results:
                unit_multiplier = safe_float(get_row_attr(row, 'unit_multiplier'))
                if unit_multiplier is None:
                    unit_multiplier = 1
                item = safe_float(get_row_attr(row, 'item'))
                calculated_value = safe_float(get_row_attr(row, 'calculated_value')) if (hasattr(row, 'calculated_value') or 'calculated_value' in dir(row)) and get_row_attr(row, 'calculated_value') is not None else None
                # Format value
                if calculated_value is not None:
                    actual_value = calculated_value
                elif item is not None:
                    actual_value = item * unit_multiplier
                else:
                    actual_value = None
                if actual_value is None:
                    formatted_value = "N/A"
                elif unit_multiplier == 1000000000.0 or abs(actual_value) >= 1e9:
                    formatted_value = f"{actual_value/1e9:,.2f}B"
                elif unit_multiplier == 1000000.0 or abs(actual_value) >= 1e6:
                    formatted_value = f"{actual_value/1e6:,.2f}M"
                elif abs(actual_value) >= 1000:
                    formatted_value = f"{actual_value:,.0f}"
                else:
                    formatted_value = f"{actual_value:,.2f}"
                item_type = get_row_attr(row, 'item_type') if hasattr(row, 'item_type') or 'item_type' in dir(row) else ''
                if item_type == 'ratio' and unit_multiplier == 1.0 and item is not None:
                    formatted_value = f"{item:.2f}%"
                record = FinancialDataRecord(
                    company=str(get_row_attr(row, 'company') if hasattr(row, 'company') or 'company' in dir(row) else get_row_attr(row, 'Company')),
                    symbol=str(get_row_attr(row, 'symbol') if hasattr(row, 'symbol') or 'symbol' in dir(row) else get_row_attr(row, 'Symbol')),
                    year=str(get_row_attr(row, 'year') if hasattr(row, 'year') or 'year' in dir(row) else get_row_attr(row, 'Year')),
                    standard_item=str(get_row_attr(row, 'standard_item')),
                    item=item,
                    unit_multiplier=unit_multiplier,
                    formatted_value=formatted_value
                )
                records.append(record)
            logger.info(f"BigQuery returned {len(records)} records.")
            return records
        except Exception as e:
            logger.error(f"Error querying BigQuery: {e}")
            return []
    
    def validate_data_availability(self, filters: FinancialDataFilters) -> Dict[str, Any]:
        logger.info(f"IN validate_data_availability: filters type: {type(filters)}, filters: {filters}")
        availability_info = {
            "has_data": True,
            "warnings": [],
            "suggestions": []
        }
        
        if not self.metadata or not self.metadata.get('associations'):
            logger.info("No metadata or associations available.")
            return availability_info
        
        # Check if specific company-year-item combinations exist
        if filters.companies and filters.years and filters.standard_items:
            logger.info("Checking company-year-item combinations.")
            missing_data = []
            available_alternatives = {}
            
            if 'company_year_to_items' in self.metadata['associations']:
                for company in filters.companies:
                    logger.info(f"Checking company: {company}")
                    company_year_items = self.metadata['associations'].get('company_year_to_items', {}).get(company, {})
                    
                    for year in filters.years:
                        logger.info(f"Checking year: {year}")
                        if year not in company_year_items:
                            available_years = self.metadata['associations'].get('company_to_years', {}).get(company, [])
                            if available_years:
                                available_alternatives[company] = available_years[-3:]  # Last 3 years
                            missing_data.append(f"{company} has no data for {year}")
                        else:
                            available_items = company_year_items[year]
                            # Robust fix for BigQuery Row 'items' field
                            if hasattr(available_items, 'get'):
                                available_items = available_items.get('items', [])
                                logger.info("Extracted available_items using .get('items', [])")
                            elif hasattr(available_items, 'items') and not callable(available_items.items):
                                available_items = available_items.items
                                logger.info("Extracted available_items using attribute access (not callable)")
                            elif hasattr(available_items, 'items') and callable(available_items.items):
                                logger.warning("available_items is still a method after attempted extraction; falling back to empty list.")
                                available_items = []
                            else:
                                logger.warning("Could not extract 'items' from available_items; falling back to empty list.")
                                available_items = []
                            logger.info(f"available_items type: {type(available_items)}, available_items: {available_items}")
                            for item in filters.standard_items:
                                logger.info(f"Checking item: {item}")
                                if item not in available_items:
                                    missing_data.append(f"{company} ({year}) missing: {item}")
            
            if missing_data:
                availability_info['warnings'] = missing_data[:5]  # Limit warnings
                if available_alternatives:
                    for company, years in available_alternatives.items():
                        availability_info['suggestions'].append(
                            f"Try {company} for years: {', '.join(years)}"
                        )
        
        # Check if items are available for ANY of the requested companies
        elif filters.companies and filters.standard_items:
            logger.info("Checking item availability for companies.")
            for item in filters.standard_items:
                logger.info(f"Checking item: {item}")
                item_companies = set(self.metadata['associations'].get('item_to_companies', {}).get(item, []))
                requested_companies = set(filters.companies)
                
                if not item_companies.intersection(requested_companies):
                    availability_info['warnings'].append(f"'{item}' not available for any selected companies")
                    # Suggest companies that have this item
                    available_for = list(item_companies)[:3]
                    if available_for:
                        availability_info['suggestions'].append(
                            f"'{item}' is available for: {', '.join(available_for)}"
                        )
        
        # Note: We don't call query_data here to avoid duplicate processing
        # The main endpoint will call query_data and check if results are empty
        
        logger.info(f"Returning availability_info: {availability_info}")
        return availability_info
    
    def format_response(self, records: List[FinancialDataRecord], query: str, interpretation: str, is_follow_up: bool = False, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Format the query results into a readable response with conversation awareness"""
        if not records:
            return "No data found matching your query criteria."
        if not self.model:
            # Fallback to basic formatting
            return f"Found {len(records)} records matching your query. Here's a summary of the data."
        # Use Gemini to create a natural language response
        data_summary = [r.dict() for r in records[:20]]  # Limit to first 20 records for prompt
        # Get conversation context for more natural responses
        conversation_context = ""
        if is_follow_up and conversation_history:
            recent_messages = conversation_history[-4:]
            conversation_context = "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in recent_messages])
        prompt = f"""
            Create a concise, informative response to the user's financial data query.
            {f'This is a follow-up question in an ongoing conversation.' if is_follow_up else ''}
            {"Recent conversation:" if conversation_context else ""}
            {conversation_context}
            User Query: "{query}"
            Query Interpretation: "{interpretation}"
            Data Found ({len(records)} total records, showing first {min(20, len(records))}):
            {json.dumps(data_summary, indent=2)}
            Create a natural language response that:
            1. {"Acknowledges this is a follow-up and references previous context" if is_follow_up else "Confirms what data was found"}
            2. Highlights key insights or patterns
            3. Formats numbers appropriately (e.g., millions, billions)
            4. Suggests relevant follow-up questions that involve querying more financial data
            5. {"Maintains conversational continuity" if is_follow_up else "Sets up potential follow-up questions"}
            
            IMPORTANT: Only suggest follow-up questions about querying financial data from the database. 
            Do NOT suggest creating charts, performing calculations, generating reports, or any other actions.
            Valid follow-up suggestions include:
            - Asking about different years (e.g., "What about 2022?" or "How did this compare in 2021?")
            - Asking about different companies or competitors
            - Asking about different financial metrics (revenue, profit margins, ratios, etc.)
            - Asking for comparative analysis between companies or time periods
            - Asking about specific financial items not yet shown
            
            Keep the response concise but informative. Be conversational and natural.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            # Fallback to basic formatting
            return f"Found {len(records)} records matching your query. Here's a summary of the data." 