import pandas as pd
import json
import os
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import google.generativeai as genai
from app.models import FinancialDataFilters, FinancialDataRecord

logger = logging.getLogger(__name__)

class FinancialDataManager:
    """
    Manager class for handling financial data queries and processing
    """
    
    def __init__(self, csv_path: str = "financial_data.csv", metadata_path: str = "metadata.json"):
        self.csv_path = csv_path
        self.metadata_path = metadata_path
        self.df: Optional[pd.DataFrame] = None
        self.metadata: Optional[Dict] = None
        self.model = None
        self._initialize_ai_model()
    
    def _initialize_ai_model(self):
        """Initialize the Gemini AI model"""
        try:
            # Use the same API key configuration as the main app
            api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
            if api_key:
                logger.info(f"Found API key (length: {len(api_key)}), initializing Gemini AI model...")
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                logger.info("✅ Gemini AI model initialized successfully for financial data queries")
            else:
                logger.warning("❌ GOOGLE_API_KEY/GEMINI_API_KEY not found, falling back to basic parsing")
                # Log available environment variables for debugging
                env_vars = [key for key in os.environ.keys() if 'API' in key or 'KEY' in key]
                logger.info(f"Available API-related env vars: {env_vars}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini AI model: {e}")
            self.model = None
    
    def load_data(self) -> bool:
        """Load the CSV file and metadata"""
        try:
            # Check if CSV file exists
            if not os.path.exists(self.csv_path):
                logger.warning(f"Financial data CSV not found at {self.csv_path}")
                return False
            
            # Load CSV file
            self.df = pd.read_csv(self.csv_path)
            
            # Clean up year column - convert to string for consistent handling
            self.df['Year'] = self.df['Year'].astype(str)
            
            # Check if we have the expected columns
            required_columns = ['Company', 'Symbol', 'Year', 'standard_item', 'item_value', 'unit_multiplier']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                logger.info(f"Available columns: {list(self.df.columns)}")
                return False
            
            logger.info(f"Loaded financial data: {len(self.df)} records")
            logger.info(f"Available columns: {list(self.df.columns)}")
            logger.info(f"Unique companies: {len(self.df['Company'].unique())}")
            logger.info(f"Unique symbols: {len(self.df['Symbol'].unique())}")
            logger.info(f"Year range: {self.df['Year'].min()} - {self.df['Year'].max()}")
            logger.info(f"Unique metrics: {len(self.df['standard_item'].unique())}")
            
            # Load or create metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                # Check if metadata has all required associations
                required_associations = [
                    'company_year_to_items', 
                    'symbol_year_to_items', 
                    'year_to_items'
                ]
                
                if 'associations' not in self.metadata or any(
                    key not in self.metadata['associations'] 
                    for key in required_associations
                ):
                    logger.info("Metadata is incomplete. Regenerating with full associations...")
                    self.create_metadata()
                else:
                    logger.info("Loaded existing metadata")
            else:
                logger.info("Creating new metadata...")
                self.create_metadata()
                
            return True
        except Exception as e:
            logger.error(f"Error loading financial data: {str(e)}")
            return False
    
    def create_metadata(self):
        """Create metadata from the dataframe"""
        if self.df is None:
            raise ValueError("DataFrame not loaded")
        
        try:
            # Create company-symbol mapping
            company_symbol_map = self.df.groupby('Company')['Symbol'].unique().apply(list).to_dict()
            
            # Create symbol-company mapping (reverse lookup)
            symbol_company_map = {}
            for company, symbols in company_symbol_map.items():
                for symbol in symbols:
                    if symbol not in symbol_company_map:
                        symbol_company_map[symbol] = []
                    symbol_company_map[symbol].append(company)
            
            # Create other mappings
            company_years_map = self.df.groupby('Company')['Year'].unique().apply(sorted).apply(list).to_dict()
            company_items_map = self.df.groupby('Company')['standard_item'].unique().apply(sorted).apply(list).to_dict()
            year_companies_map = self.df.groupby('Year')['Company'].unique().apply(sorted).apply(list).to_dict()
            item_companies_map = self.df.groupby('standard_item')['Company'].unique().apply(sorted).apply(list).to_dict()
            
            # Create company-year-items mapping
            company_year_items_map = {}
            for company in self.df['Company'].unique():
                company_year_items_map[company] = {}
                company_df = self.df[self.df['Company'] == company]
                for year in company_df['Year'].unique():
                    year_items = company_df[company_df['Year'] == year]['standard_item'].unique().tolist()
                    company_year_items_map[company][year] = sorted(year_items)
            
            # Create year-item mapping
            year_items_map = self.df.groupby('Year')['standard_item'].unique().apply(sorted).apply(list).to_dict()
            
            # Create symbol-year-items mapping
            symbol_year_items_map = {}
            for symbol in self.df['Symbol'].unique():
                symbol_year_items_map[symbol] = {}
                symbol_df = self.df[self.df['Symbol'] == symbol]
                for year in symbol_df['Year'].unique():
                    year_items = symbol_df[symbol_df['Year'] == year]['standard_item'].unique().tolist()
                    symbol_year_items_map[symbol][year] = sorted(year_items)
            
            self.metadata = {
                "companies": sorted(self.df['Company'].unique().tolist()),
                "symbols": sorted(self.df['Symbol'].unique().tolist()),
                "years": sorted(self.df['Year'].unique().tolist()),
                "standard_items": sorted(self.df['standard_item'].unique().tolist()),
                "associations": {
                    "company_to_symbol": company_symbol_map,
                    "symbol_to_company": symbol_company_map,
                    "company_to_years": company_years_map,
                    "company_to_items": company_items_map,
                    "year_to_companies": year_companies_map,
                    "item_to_companies": item_companies_map,
                    "company_year_to_items": company_year_items_map,
                    "symbol_year_to_items": symbol_year_items_map,
                    "year_to_items": year_items_map
                },
                "total_records": len(self.df),
                "last_updated": pd.Timestamp.now().isoformat()
            }
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            logger.info("Created and saved financial data metadata")
            
        except Exception as e:
            logger.error(f"Error creating metadata: {str(e)}")
            raise
    
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
            for symbol, companies in self.metadata['associations']['symbol_to_company'].items():
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
        """Post-process filters to ensure consistency using associations"""
        if not self.metadata or 'associations' not in self.metadata:
            return result
        
        # If symbols are specified, add ALL associated companies
        if result.get('symbols') and len(result['symbols']) > 0:
            companies = set()
            valid_symbols = []
            
            for symbol in result['symbols']:
                if symbol in self.metadata['associations']['symbol_to_company']:
                    companies.update(self.metadata['associations']['symbol_to_company'][symbol])
                    valid_symbols.append(symbol)
                else:
                    logger.warning(f"Symbol '{symbol}' not found in available symbols")
            
            # Update symbols to only include valid ones
            result['symbols'] = valid_symbols
            
            if companies:
                result['companies'] = list(companies)
                logger.info(f"Found companies for symbols {result['symbols']}: {', '.join(result['companies'])}")
            elif valid_symbols:
                logger.error(f"No companies found for symbols: {result['symbols']}")
        
        # If companies are specified, add their symbols
        elif result.get('companies') and len(result['companies']) > 0:
            symbols = set()
            valid_companies = []
            
            for company in result['companies']:
                if company in self.metadata['associations']['company_to_symbol']:
                    symbols.update(self.metadata['associations']['company_to_symbol'][company])
                    valid_companies.append(company)
                else:
                    # Try partial matching
                    matched = False
                    for actual_company in self.metadata['companies']:
                        if company.lower() in actual_company.lower() or actual_company.lower() in company.lower():
                            if actual_company in self.metadata['associations']['company_to_symbol']:
                                symbols.update(self.metadata['associations']['company_to_symbol'][actual_company])
                                valid_companies.append(actual_company)
                                matched = True
                                break
                    
                    if not matched:
                        logger.warning(f"Company '{company}' not found")
            
            # Update companies list
            result['companies'] = valid_companies
            
            if symbols:
                result['symbols'] = list(symbols)
                logger.info(f"Found symbols for companies: {result['symbols']}")
        
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
            for symbol in result['symbols']:
                if symbol in self.metadata['associations']['symbol_to_company']:
                    companies.update(self.metadata['associations']['symbol_to_company'][symbol])
            if companies:
                result['companies'] = list(companies)
                logger.info(f"Found companies for symbols: {', '.join(result['companies'])}")
        
        return FinancialDataFilters(**result)
    
    def query_data(self, filters: FinancialDataFilters) -> pd.DataFrame:
        """Query the dataframe based on filters"""
        if self.df is None:
            raise ValueError("DataFrame not loaded")
        
        filtered_df = self.df.copy()
        
        logger.info(f"Starting with {len(filtered_df)} total records")
        logger.info(f"Applying filters: companies={filters.companies}, symbols={filters.symbols}, years={filters.years}, items={filters.standard_items}")
        
        # Apply company filter
        if filters.companies and len(filters.companies) > 0:
            unique_companies = filtered_df['Company'].unique()
            matching_companies = [c for c in filters.companies if c in unique_companies]
            logger.info(f"Found matching companies in data: {matching_companies}")
            
            if matching_companies:
                filtered_df = filtered_df[filtered_df['Company'].isin(matching_companies)]
                logger.info(f"After company filter: {len(filtered_df)} records")
            else:
                logger.warning(f"None of the requested companies found in data: {filters.companies}")
                logger.info(f"Available companies sample: {list(unique_companies)[:10]}")
        
        # Apply symbol filter
        if filters.symbols and len(filters.symbols) > 0:
            unique_symbols = filtered_df['Symbol'].unique()
            logger.info(f"Looking for symbols: {filters.symbols}")
            logger.info(f"Available symbols in current data: {list(unique_symbols)}")
            
            filtered_df = filtered_df[filtered_df['Symbol'].isin(filters.symbols)]
            logger.info(f"After symbol filter: {len(filtered_df)} records")
        
        # Apply year filter
        if filters.years and len(filters.years) > 0:
            # Convert years to string for comparison
            year_strings = [str(y) for y in filters.years]
            filtered_df = filtered_df[filtered_df['Year'].isin(year_strings)]
            logger.info(f"After year filter: {len(filtered_df)} records")
        
        # Apply standard items filter
        if filters.standard_items and len(filters.standard_items) > 0:
            if len(filtered_df) > 0:
                available_items = filtered_df['standard_item'].unique()
                logger.info(f"Available items in filtered data: {list(available_items)[:10]}")
            
            filtered_df = filtered_df[filtered_df['standard_item'].isin(filters.standard_items)]
            logger.info(f"After item filter: {len(filtered_df)} records")
        
        if len(filtered_df) == 0:
            logger.warning("No data found with current filters")
        else:
            logger.info(f"Found {len(filtered_df)} matching records")
        
        return filtered_df
    
    def validate_data_availability(self, filters: FinancialDataFilters) -> Dict[str, Any]:
        """Validate what data is actually available for the given filters"""
        availability_info = {
            "has_data": True,
            "warnings": [],
            "suggestions": []
        }
        
        if not self.metadata or not self.metadata.get('associations'):
            return availability_info
        
        # Check if specific company-year-item combinations exist
        if filters.companies and filters.years and filters.standard_items:
            missing_data = []
            available_alternatives = {}
            
            if 'company_year_to_items' in self.metadata['associations']:
                for company in filters.companies:
                    company_year_items = self.metadata['associations'].get('company_year_to_items', {}).get(company, {})
                    
                    for year in filters.years:
                        if year not in company_year_items:
                            # This company doesn't have data for this year
                            available_years = self.metadata['associations'].get('company_to_years', {}).get(company, [])
                            if available_years:
                                available_alternatives[company] = available_years[-3:]  # Last 3 years
                            missing_data.append(f"{company} has no data for {year}")
                        else:
                            # Check which items are available
                            available_items = company_year_items[year]
                            for item in filters.standard_items:
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
            for item in filters.standard_items:
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
        
        return availability_info
    
    def format_response(self, df: pd.DataFrame, query: str, interpretation: str, is_follow_up: bool = False, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Format the query results into a readable response with conversation awareness"""
        if df.empty:
            return "No data found matching your query criteria."
        
        if not self.model:
            # Fallback to basic formatting
            return f"Found {len(df)} records matching your query. Here's a summary of the data."
        
        # Use Gemini to create a natural language response
        data_summary = df.to_dict('records')[:20]  # Limit to first 20 records for prompt
        
        # Get conversation context for more natural responses
        conversation_context = ""
        if is_follow_up and conversation_history:
            recent_messages = conversation_history[-4:]
            conversation_context = "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in recent_messages])
        
        prompt = f"""
        Create a concise, informative response to the user's financial data query.
        {f"This is a follow-up question in an ongoing conversation." if is_follow_up else ""}
        
        {"Recent conversation:" if conversation_context else ""}
        {conversation_context}
        
        User Query: "{query}"
        Query Interpretation: "{interpretation}"
        
        Data Found ({len(df)} total records, showing first {min(20, len(df))}):
        {json.dumps(data_summary, indent=2)}
        
        Create a natural language response that:
        1. {"Acknowledges this is a follow-up and references previous context" if is_follow_up else "Confirms what data was found"}
        2. Highlights key insights or patterns
        3. Formats numbers appropriately (e.g., millions, billions)
        4. Suggests relevant follow-up questions based on what was found
        5. {"Maintains conversational continuity" if is_follow_up else "Sets up potential follow-up questions"}
        
        Keep the response concise but informative. Be conversational and natural.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            # Fallback to basic formatting
            return f"Found {len(df)} records matching your query. Here's a summary of the data."
    
    def convert_df_to_records(self, df: pd.DataFrame) -> List[FinancialDataRecord]:
        """Convert DataFrame rows to FinancialDataRecord objects"""
        records = []
        for _, row in df.head(50).iterrows():  # Limit to first 50 records
            try:
                # Format value based on unit multiplier
                unit_multiplier = float(row.get('unit_multiplier', 1))
                item_value = float(row.get('item_value', 0))
                
                # Calculate the actual value
                if 'calculated_value' in row and pd.notna(row['calculated_value']):
                    actual_value = float(row['calculated_value'])
                else:
                    actual_value = item_value * unit_multiplier
                
                # Format the value for display
                if unit_multiplier == 1000000000.0 or abs(actual_value) >= 1e9:
                    formatted_value = f"{actual_value/1e9:,.2f}B"
                elif unit_multiplier == 1000000.0 or abs(actual_value) >= 1e6:
                    formatted_value = f"{actual_value/1e6:,.2f}M"
                elif abs(actual_value) >= 1000:
                    formatted_value = f"{actual_value:,.0f}"
                else:
                    formatted_value = f"{actual_value:,.2f}"
                
                # Handle ratio/percentage items differently
                item_type = row.get('item_type', '')
                if item_type == 'ratio' and unit_multiplier == 1.0:
                    formatted_value = f"{item_value:.2f}%"
                
                record = FinancialDataRecord(
                    company=str(row.get('Company', '')),
                    symbol=str(row.get('Symbol', '')),
                    year=str(row.get('Year', '')),
                    standard_item=str(row.get('standard_item', '')),
                    item_value=float(item_value),
                    unit_multiplier=int(unit_multiplier),
                    formatted_value=formatted_value
                )
                records.append(record)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing record: {e}")
                continue
        
        return records 