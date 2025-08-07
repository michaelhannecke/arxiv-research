import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tempfile
import os

import pdfplumber
from app.core.config import get_settings
from app.api.schemas import Section

logger = logging.getLogger(__name__)
settings = get_settings()

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors"""
    pass

class PDFProcessor:
    """Handles PDF text extraction and processing"""
    
    def __init__(self):
        self.section_patterns = [
            # Common section patterns in academic papers
            r'^(\d+\.?\s*)(abstract|introduction|related\s+work|methodology?|method|approach|results?|evaluation|experiments?|discussion|conclusion|references|acknowledgments?|appendix)',
            r'^([ivx]+\.?\s*)(abstract|introduction|related\s+work|methodology?|method|approach|results?|evaluation|experiments?|discussion|conclusion|references|acknowledgments?|appendix)',
            r'^(abstract|introduction|related\s+work|methodology?|method|approach|results?|evaluation|experiments?|discussion|conclusion|references|acknowledgments?|appendix)$',
        ]
    
    async def extract_content(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extract text and structure from PDF bytes"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_bytes)
                temp_file_path = temp_file.name
            
            try:
                # Extract content using pdfplumber
                content = await self._extract_with_pdfplumber(temp_file_path)
                return content
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            raise PDFProcessingError(f"Failed to extract PDF content: {e}")
    
    async def _extract_with_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """Extract content using pdfplumber"""
        sections = []
        full_text = ""
        page_texts = []
        figures = []
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Processing PDF with {len(pdf.pages)} pages")
                
                # Extract text from each page
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            page_texts.append({
                                'page': page_num,
                                'text': page_text
                            })
                            full_text += page_text + "\n\n"
                        
                        # Extract tables
                        tables_on_page = page.extract_tables()
                        for table_idx, table in enumerate(tables_on_page):
                            if table:
                                tables.append({
                                    'page': page_num,
                                    'table_index': table_idx,
                                    'data': table
                                })
                        
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num}: {e}")
                        continue
                
                # Process the extracted text
                sections = self._parse_sections(full_text)
                figures = self._extract_figure_references(full_text)
                references = self._extract_references(full_text)
                
                return {
                    'full_text': full_text,
                    'sections': [section.dict() for section in sections],
                    'page_texts': page_texts,
                    'figures': figures,
                    'tables': tables,
                    'references': references,
                    'page_count': len(pdf.pages)
                }
                
        except Exception as e:
            logger.error(f"Error with pdfplumber: {e}")
            raise PDFProcessingError(f"pdfplumber extraction failed: {e}")
    
    def _parse_sections(self, text: str) -> List[Section]:
        """Parse sections from extracted text"""
        sections = []
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a section header
            section_match = self._is_section_header(line)
            if section_match:
                # Save previous section
                if current_section:
                    sections.append(Section(
                        title=current_section,
                        level=1,  # Default to level 1 for now
                        content='\n'.join(current_content).strip()
                    ))
                
                # Start new section
                current_section = section_match
                current_content = []
            else:
                # Add line to current section content
                if current_section:
                    current_content.append(line)
                else:
                    # Content before first section (title, authors, etc.)
                    if not sections:
                        sections.append(Section(
                            title="Header",
                            level=1,
                            content=line
                        ))
                    else:
                        sections[0].content += f"\n{line}"
        
        # Don't forget the last section
        if current_section and current_content:
            sections.append(Section(
                title=current_section,
                level=1,
                content='\n'.join(current_content).strip()
            ))
        
        return sections
    
    def _is_section_header(self, line: str) -> Optional[str]:
        """Check if a line is a section header"""
        line_lower = line.lower().strip()
        
        # Check against patterns
        for pattern in self.section_patterns:
            match = re.match(pattern, line_lower)
            if match:
                # Extract the section name
                if len(match.groups()) > 1:
                    return match.group(2).title()  # The section name
                else:
                    return match.group(1).title()  # The whole match
        
        # Additional heuristics
        # Check for numbered sections like "1. Introduction"
        if re.match(r'^\d+\.?\s+[A-Z][a-zA-Z\s]+$', line):
            return line
        
        # Check for capitalized section names
        if (len(line.split()) <= 4 and 
            line.isupper() and 
            len(line) > 3 and 
            not line.isdigit()):
            return line.title()
        
        return None
    
    def _extract_figure_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract figure references and captions"""
        figures = []
        
        # Pattern for figure captions
        figure_pattern = r'(Figure|Fig\.?)\s*(\d+)[:\.]?\s*([^\n]+)'
        matches = re.finditer(figure_pattern, text, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            figure_num = match.group(2)
            caption = match.group(3).strip()
            
            # Clean up caption (remove excessive whitespace)
            caption = re.sub(r'\s+', ' ', caption)
            
            figures.append({
                'number': int(figure_num),
                'caption': caption,
                'position': match.start()
            })
        
        return figures
    
    def _extract_references(self, text: str) -> List[str]:
        """Extract references from the text"""
        references = []
        
        # Find references section
        ref_patterns = [
            r'references?\s*\n',
            r'bibliography\s*\n',
            r'works?\s+cited\s*\n'
        ]
        
        ref_start = None
        for pattern in ref_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                ref_start = match.end()
                break
        
        if ref_start:
            # Extract text after references section
            ref_text = text[ref_start:]
            
            # Split into individual references
            # Look for patterns like [1], (1), or numbered lines
            ref_lines = ref_text.split('\n')
            current_ref = []
            
            for line in ref_lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this starts a new reference
                if (re.match(r'^\[\d+\]', line) or 
                    re.match(r'^\(\d+\)', line) or 
                    re.match(r'^\d+\.', line)):
                    
                    # Save previous reference
                    if current_ref:
                        ref_text = ' '.join(current_ref).strip()
                        if ref_text:
                            references.append(ref_text)
                        current_ref = []
                    
                    current_ref.append(line)
                else:
                    # Continue current reference
                    current_ref.append(line)
            
            # Don't forget the last reference
            if current_ref:
                ref_text = ' '.join(current_ref).strip()
                if ref_text:
                    references.append(ref_text)
        
        return references[:50]  # Limit to first 50 references
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)      # Multiple spaces to single
        
        # Remove page numbers and headers/footers that appear multiple times
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip likely page numbers
            if re.match(r'^\d+$', line) and len(line) <= 3:
                continue
            
            # Skip very short lines that are likely artifacts
            if len(line) < 3:
                continue
            
            # Skip lines with only special characters
            if re.match(r'^[^\w\s]*$', line):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def extract_title_and_authors(self, first_page_text: str) -> Tuple[Optional[str], List[str]]:
        """Extract title and authors from first page text"""
        lines = first_page_text.split('\n')[:20]  # Look at first 20 lines
        
        title = None
        authors = []
        
        # Simple heuristic: title is usually the first long line
        for i, line in enumerate(lines):
            line = line.strip()
            if len(line) > 20 and not re.match(r'^(abstract|keywords?)', line, re.IGNORECASE):
                title = line
                break
        
        # Look for author patterns
        author_patterns = [
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s*,\s*([A-Z][a-z]+\s+[A-Z][a-z]+))*',
            r'^([A-Z]\.\s*[A-Z][a-z]+)(?:\s*,\s*([A-Z]\.\s*[A-Z][a-z]+))*'
        ]
        
        for line in lines[1:10]:  # Skip first line (likely title)
            line = line.strip()
            for pattern in author_patterns:
                if re.match(pattern, line):
                    # Split by comma and clean
                    potential_authors = [a.strip() for a in line.split(',')]
                    for author in potential_authors:
                        if author and len(author.split()) >= 2:
                            authors.append(author)
                    break
        
        return title, authors

# Singleton instance
pdf_processor = PDFProcessor()