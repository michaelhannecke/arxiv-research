import logging
from typing import List, Dict, Any, Optional
import asyncio
import re
from datetime import datetime

import anthropic
from app.core.config import get_settings
from app.api.schemas import Section, ArxivMetadata

logger = logging.getLogger(__name__)
settings = get_settings()

class AnthropicAPIError(Exception):
    """Custom exception for Anthropic API errors"""
    pass

class AnthropicService:
    """Service for interacting with Anthropic's Claude API"""
    
    def __init__(self):
        self.api_key = settings.anthropic.api_key
        self.model = settings.anthropic.model
        self.max_tokens = settings.anthropic.max_tokens
        self.temperature = settings.anthropic.temperature
        self.client = None
        
        if self.api_key:
            try:
                self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
                logger.info("Anthropic service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                self.client = None
        else:
            logger.warning("ANTHROPIC_API_KEY not found - AI features will be disabled")
    
    def is_available(self) -> bool:
        """Check if the Anthropic service is available"""
        return self.client is not None
    
    async def summarize_abstract(self, abstract: str, metadata: ArxivMetadata) -> str:
        """Generate an enhanced abstract summary"""
        if not self.is_available():
            raise AnthropicAPIError("Anthropic API not available - API key not configured")
        
        prompt = f"""You are an expert academic researcher. Please create an enhanced summary of this research paper's abstract.

Paper Title: {metadata.title}
Authors: {', '.join(metadata.authors[:5])}
Categories: {', '.join(metadata.categories)}
Published: {metadata.published.strftime('%Y-%m-%d')}

Original Abstract:
{abstract}

Please provide a clear, accessible summary that:
1. Explains the main problem being addressed
2. Describes the key contribution or solution
3. Highlights the main results or findings
4. Uses accessible language while maintaining technical accuracy
5. Keeps it concise (2-3 sentences)

Enhanced Summary:"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            summary = response.content[0].text.strip()
            logger.info(f"Generated abstract summary: {len(summary)} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating abstract summary: {e}")
            raise AnthropicAPIError(f"Failed to generate abstract summary: {e}")
    
    async def summarize_section(self, section: Section, context: Dict[str, Any] = None) -> str:
        """Generate a summary for a specific section"""
        if not self.is_available():
            raise AnthropicAPIError("Anthropic API not available - API key not configured")
        
        context = context or {}
        paper_title = context.get('paper_title', 'Unknown Paper')
        paper_categories = context.get('paper_categories', [])
        
        # Skip sections that are too short or don't need summarization
        if len(section.content) < 200:
            return section.content
        
        # Different prompts based on section type
        section_type = self._identify_section_type(section.title)
        
        if section_type == "introduction":
            prompt = self._get_introduction_prompt(section, paper_title, paper_categories)
        elif section_type == "methodology":
            prompt = self._get_methodology_prompt(section, paper_title, paper_categories)
        elif section_type == "results":
            prompt = self._get_results_prompt(section, paper_title, paper_categories)
        elif section_type == "conclusion":
            prompt = self._get_conclusion_prompt(section, paper_title, paper_categories)
        else:
            prompt = self._get_general_prompt(section, paper_title, paper_categories)
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=min(self.max_tokens, 1000),  # Limit section summaries
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            summary = response.content[0].text.strip()
            logger.info(f"Generated summary for '{section.title}': {len(summary)} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing section '{section.title}': {e}")
            # Return truncated original content as fallback
            return section.content[:500] + "..." if len(section.content) > 500 else section.content
    
    def _identify_section_type(self, title: str) -> str:
        """Identify the type of section based on title"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['introduction', 'intro', 'background']):
            return "introduction"
        elif any(word in title_lower for word in ['method', 'approach', 'algorithm', 'model', 'architecture']):
            return "methodology"
        elif any(word in title_lower for word in ['result', 'experiment', 'evaluation', 'performance', 'analysis']):
            return "results"
        elif any(word in title_lower for word in ['conclusion', 'discussion', 'future', 'limitation']):
            return "conclusion"
        else:
            return "general"
    
    def _get_introduction_prompt(self, section: Section, paper_title: str, categories: List[str]) -> str:
        return f"""You are summarizing the Introduction section of an academic paper.

Paper: {paper_title}
Field: {', '.join(categories[:3])}
Section: {section.title}

Content:
{section.content[:3000]}

Please provide a concise summary (2-3 sentences) that captures:
1. The main problem or research question
2. Why this problem is important
3. The gap in existing work that this paper addresses

Summary:"""

    def _get_methodology_prompt(self, section: Section, paper_title: str, categories: List[str]) -> str:
        return f"""You are summarizing the Methodology section of an academic paper.

Paper: {paper_title}
Field: {', '.join(categories[:3])}
Section: {section.title}

Content:
{section.content[:3000]}

Please provide a concise summary (2-3 sentences) that captures:
1. The key approach or method used
2. Important technical details or innovations
3. How this method differs from or improves upon existing approaches

Summary:"""

    def _get_results_prompt(self, section: Section, paper_title: str, categories: List[str]) -> str:
        return f"""You are summarizing the Results/Experiments section of an academic paper.

Paper: {paper_title}
Field: {', '.join(categories[:3])}
Section: {section.title}

Content:
{section.content[:3000]}

Please provide a concise summary (2-3 sentences) that captures:
1. The main experimental findings
2. Key performance metrics or improvements
3. How the results compare to baselines or previous work

Summary:"""

    def _get_conclusion_prompt(self, section: Section, paper_title: str, categories: List[str]) -> str:
        return f"""You are summarizing the Conclusion section of an academic paper.

Paper: {paper_title}
Field: {', '.join(categories[:3])}
Section: {section.title}

Content:
{section.content[:3000]}

Please provide a concise summary (2-3 sentences) that captures:
1. The main contributions of the work
2. Key implications or impact
3. Future research directions mentioned

Summary:"""

    def _get_general_prompt(self, section: Section, paper_title: str, categories: List[str]) -> str:
        return f"""You are summarizing a section of an academic paper.

Paper: {paper_title}
Field: {', '.join(categories[:3])}
Section: {section.title}

Content:
{section.content[:3000]}

Please provide a concise, accessible summary (2-3 sentences) that captures the main points and key information from this section.

Summary:"""

    async def extract_keywords(self, full_text: str, metadata: ArxivMetadata) -> List[str]:
        """Extract relevant keywords using AI"""
        if not self.is_available():
            logger.warning("Anthropic API not available, using basic keyword extraction")
            return self._extract_basic_keywords(full_text, metadata)
        
        # Prepare text sample (use abstract + first part of content)
        text_sample = f"{metadata.abstract}\n\n{full_text[:2000]}"
        
        prompt = f"""You are an expert academic researcher. Extract 10-15 key technical terms and concepts from this research paper.

Paper Title: {metadata.title}
Field: {', '.join(metadata.categories)}

Text Sample:
{text_sample}

Please identify the most important keywords that represent:
1. Technical methods and algorithms
2. Key concepts and terminology
3. Application domains
4. Novel contributions
5. Research areas

Return only the keywords, separated by commas, without explanations. Focus on specific, meaningful terms rather than generic words.

Keywords:"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=200,  # Keep keywords concise
                temperature=0.3,  # Lower temperature for more consistent extraction
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            keywords_text = response.content[0].text.strip()
            
            # Parse keywords from response
            keywords = [kw.strip() for kw in keywords_text.split(',')]
            keywords = [kw for kw in keywords if len(kw) > 2 and len(kw) < 30]  # Filter length
            keywords = keywords[:15]  # Limit to 15 keywords
            
            logger.info(f"Extracted {len(keywords)} keywords using AI")
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            # Fallback to basic extraction
            return self._extract_basic_keywords(full_text, metadata)
    
    def _extract_basic_keywords(self, full_text: str, metadata: ArxivMetadata) -> List[str]:
        """Fallback basic keyword extraction"""
        keywords = []
        
        # Add categories as keywords
        keywords.extend(metadata.categories[:5])
        
        # Basic keyword extraction from title and abstract
        text = f"{metadata.title} {metadata.abstract}".lower()
        
        # Common academic/technical terms
        common_terms = [
            r'\b(machine learning|deep learning|neural network|artificial intelligence)\b',
            r'\b(algorithm|optimization|classification|regression|clustering)\b',
            r'\b(model|framework|approach|method|technique|architecture)\b',
            r'\b(analysis|evaluation|experiment|study|research|performance)\b',
            r'\b(dataset|data|training|testing|validation|benchmark)\b',
            r'\b(accuracy|precision|recall|f1|score|metric)\b',
            r'\b(natural language processing|nlp|computer vision|cv|nlg)\b',
            r'\b(transformer|attention|embedding|representation|feature)\b',
            r'\b(supervised|unsupervised|reinforcement|semi-supervised)\b',
            r'\b(convolutional|recurrent|lstm|gru|rnn|cnn)\b'
        ]
        
        for pattern in common_terms:
            matches = re.findall(pattern, text)
            for match in matches:
                if match not in keywords:
                    keywords.append(match)
        
        return keywords[:10]  # Limit to 10 keywords
    
    async def generate_full_summary(self, sections: List[Section], metadata: ArxivMetadata) -> str:
        """Generate a comprehensive summary of the entire paper"""
        if not self.is_available():
            raise AnthropicAPIError("Anthropic API not available - API key not configured")
        
        # Create a structured overview of the paper
        sections_overview = []
        for section in sections[:10]:  # Limit to first 10 sections
            if len(section.content) > 100:  # Only include substantial sections
                sections_overview.append(f"**{section.title}**: {section.summary or section.content[:200]}...")
        
        overview_text = "\n".join(sections_overview)
        
        prompt = f"""You are an expert academic researcher. Create a comprehensive yet accessible summary of this research paper.

Paper Details:
- Title: {metadata.title}
- Authors: {', '.join(metadata.authors[:5])}
- Field: {', '.join(metadata.categories)}
- Published: {metadata.published.strftime('%Y-%m-%d')}

Abstract:
{metadata.abstract}

Section Overview:
{overview_text}

Please provide a comprehensive summary (4-5 sentences) that covers:
1. The main research problem and motivation
2. The key approach or methodology
3. Primary findings and contributions
4. Significance and potential impact

Make it accessible to researchers in related fields while maintaining technical accuracy.

Comprehensive Summary:"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            summary = response.content[0].text.strip()
            logger.info(f"Generated comprehensive summary: {len(summary)} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating comprehensive summary: {e}")
            # Fallback to enhanced abstract
            return f"**Summary**: {metadata.abstract[:500]}..."
    
    async def batch_summarize_sections(self, sections: List[Section], context: Dict[str, Any], max_concurrent: int = 3) -> List[Section]:
        """Summarize multiple sections concurrently"""
        logger.info(f"Starting batch summarization of {len(sections)} sections")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def summarize_with_semaphore(section: Section) -> Section:
            async with semaphore:
                try:
                    summary = await self.summarize_section(section, context)
                    section.summary = summary
                    return section
                except Exception as e:
                    logger.warning(f"Failed to summarize section '{section.title}': {e}")
                    section.summary = None
                    return section
                finally:
                    # Small delay to respect rate limits
                    await asyncio.sleep(0.5)
        
        # Process sections concurrently
        summarized_sections = await asyncio.gather(
            *[summarize_with_semaphore(section) for section in sections],
            return_exceptions=True
        )
        
        # Handle any exceptions
        valid_sections = []
        for i, result in enumerate(summarized_sections):
            if isinstance(result, Exception):
                logger.error(f"Error processing section {i}: {result}")
                sections[i].summary = None
                valid_sections.append(sections[i])
            else:
                valid_sections.append(result)
        
        logger.info(f"Completed batch summarization: {len([s for s in valid_sections if s.summary])} successful")
        return valid_sections

# Singleton instance
anthropic_service = AnthropicService()