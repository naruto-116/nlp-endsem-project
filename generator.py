"""
LLM generation module using Gemini or Groq API.
Handles answer generation with citation formatting.
"""
import os
from typing import Dict, List, Optional

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class LegalAnswerGenerator:
    """
    Generates legal answers using LLM (Gemini or Groq) with knowledge graph support.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", provider: str = "gemini", knowledge_graph=None):
        """
        Initialize the generator.
        
        Args:
            api_key: API key (Gemini or Groq)
            model_name: Model to use (Gemini 2.5 Flash)
            provider: "gemini" or "groq"
            knowledge_graph: Optional LegalKnowledgeGraph for citation enrichment
        """
        self.provider = provider
        self.model_name = model_name
        self.knowledge_graph = knowledge_graph
        
        if provider == "gemini":
            if not genai:
                raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model_name)
            print(f"[OK] Initialized Gemini generator with model: {model_name}")
        else:
            if not Groq:
                raise ImportError("groq package not installed. Run: pip install groq")
            self.client = Groq(api_key=api_key)
            print(f"[OK] Initialized Groq generator with model: {model_name}")
    
    def create_prompt(self, query: str, context: str, related_cases: List[str] = None) -> str:
        """
        Create a structured prompt for legal QA with knowledge graph suggestions.
        
        Args:
            query: User's question
            context: Retrieved legal context
            related_cases: Optional list of related case names from knowledge graph
            
        Returns:
            Formatted prompt
        """
        # Add related cases section if knowledge graph provided suggestions
        related_section = ""
        if related_cases and len(related_cases) > 0:
            related_section = f"""

RELATED LANDMARK CASES (from citation network):
{chr(10).join(f"- {case}" for case in related_cases[:5])}
Consider citing these if relevant to the query.
"""
        
        prompt = f"""You are a legal research assistant for Indian Supreme Court law. Provide an informative answer based on the context below.

TASK:
- Answer using ONLY the provided legal context
- Cite sources using the EXACT format provided in context: [Document Name-pageX] for PDFs or [Case Name]
- ALWAYS include page numbers when they are provided in the source
- Be factual and objective
- If information is insufficient, state that clearly
- Reference the related landmark cases when they strengthen your answer

LEGAL CONTEXT:
{context}{related_section}

QUESTION:
{query}

ANSWER WITH CITATIONS:"""
        
        return prompt
    
    def generate_answer(self, query: str, context: str, 
                       max_tokens: int = 8192,
                       temperature: float = 0.3,
                       retrieved_case_ids: List[str] = None) -> Dict:
        """
        Generate an answer to the legal query with knowledge graph enrichment.
        
        Args:
            query: User's question
            context: Retrieved legal context
            max_tokens: Maximum length of answer
            temperature: Sampling temperature (lower = more deterministic)
            retrieved_case_ids: Optional list of retrieved case IDs for graph traversal
            
        Returns:
            Dictionary with 'answer' and metadata
        """
        # Get related cases from knowledge graph if available
        related_cases = []
        if self.knowledge_graph and retrieved_case_ids:
            related_cases = self._get_related_cases_from_graph(retrieved_case_ids)
        
        prompt = self.create_prompt(query, context, related_cases)
        
        try:
            if self.provider == "gemini":
                # Gemini API call with safety settings
                from google.generativeai.types import HarmCategory, HarmBlockThreshold
                
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
                
                response = self.client.generate_content(
                    prompt,
                    generation_config={
                        'max_output_tokens': 8192,
                        'temperature': temperature,
                    },
                    safety_settings=safety_settings
                )
                
                # Check if response has valid content
                try:
                    answer = response.text
                except (ValueError, AttributeError) as e:
                    # Fallback answer when Gemini blocks/fails
                    return {
                        'answer': self._generate_fallback_answer(context, query),
                        'model': self.model_name,
                        'tokens_used': None,
                        'finish_reason': 'fallback',
                        'error': f'Response blocked. Using fallback.'
                    }
                
                answer = response.text
                
                return {
                    'answer': answer,
                    'model': self.model_name,
                    'tokens_used': None,
                    'finish_reason': 'stop'
                }
            else:
                # Groq API call
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a legal AI assistant specializing in Indian Supreme Court law. Always cite cases in [brackets]."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                answer = response.choices[0].message.content
                
                return {
                    'answer': answer,
                    'model': self.model_name,
                    'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else None,
                    'finish_reason': response.choices[0].finish_reason
                }
            
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'model': self.model_name,
                'error': str(e)
            }
    
    def _get_related_cases_from_graph(self, case_ids: List[str], max_related: int = 5) -> List[str]:
        """
        Get related cases from knowledge graph based on citation relationships.
        
        Args:
            case_ids: List of case IDs from retrieved results
            max_related: Maximum number of related cases to return
            
        Returns:
            List of related case names
        """
        if not self.knowledge_graph:
            return []
        
        related_cases = []
        seen_cases = set(case_ids)
        
        for case_id in case_ids[:3]:  # Check top 3 retrieved cases
            if not self.knowledge_graph.case_exists(case_id):
                continue
            
            # Get cases cited by this case
            cited_cases = self.knowledge_graph.get_cites(case_id, limit=max_related)
            for cited_id in cited_cases:
                if cited_id not in seen_cases:
                    case_name = self.knowledge_graph.graph.nodes[cited_id].get('name', cited_id)
                    related_cases.append(case_name)
                    seen_cases.add(cited_id)
                    if len(related_cases) >= max_related:
                        break
            
            if len(related_cases) >= max_related:
                break
        
        # If not enough from citations, add some landmark cases
        if len(related_cases) < max_related:
            landmark_cases = self.knowledge_graph.get_landmark_cases(top_n=max_related - len(related_cases))
            for case_id, _ in landmark_cases:
                if case_id not in seen_cases:
                    case_name = self.knowledge_graph.graph.nodes[case_id].get('name', case_id)
                    related_cases.append(case_name)
        
        return related_cases[:max_related]
    
    def _generate_fallback_answer(self, context: str, query: str) -> str:
        """
        Generate a simple extractive answer from context when API fails.
        """
        # Extract first few sentences from context
        sentences = context.split('\n')[:3]
        context_preview = '\n'.join(sentences)
        
        return f"""Based on the retrieved documents, here is relevant information regarding: "{query}"

{context_preview}

Note: Full AI-generated answer unavailable. The above is extracted directly from retrieved Supreme Court judgments and uploaded documents.

To get a comprehensive answer, please:
1. Verify your API key is valid
2. Check your internet connection
3. Try rephrasing your query
"""
    
    def generate_with_fallback(self, query: str, context: str) -> str:
        """
        Generate answer with fallback to simple response if API fails.
        
        Args:
            query: User's question
            context: Retrieved legal context
            
        Returns:
            Generated answer text
        """
        result = self.generate_answer(query, context)
        
        if 'error' in result:
            # Fallback response
            return f"Unable to generate answer due to API error. Please check your API key and try again.\n\nRetrieved context is available for manual review."
        
        return result['answer']


class DummyGenerator:
    """
    Dummy generator for testing without API keys.
    """
    
    def __init__(self):
        print("⚠ Using dummy generator (no API key provided)")
    
    def generate_answer(self, query: str, context: str, **kwargs) -> Dict:
        """Generate a dummy answer."""
        answer = f"""Based on the retrieved Supreme Court judgments, here is a summary regarding your query: "{query}"

[Dummy Case 1 v. State] and [Dummy Case 2 v. Union of India] are relevant precedents that address this matter.

(This is a dummy response. Please configure your Groq API key in the .env file to get real answers.)

Retrieved Context Preview:
{context[:200]}...
"""
        return {
            'answer': answer,
            'model': 'dummy',
            'tokens_used': 0
        }
    
    def generate_with_fallback(self, query: str, context: str) -> str:
        """Generate fallback answer."""
        result = self.generate_answer(query, context)
        return result['answer']
