"""Detect document sections using simple heuristics"""

import re


def detect_sections(pages_data: list) -> list:
    """
    Detect academic section headers and assign section labels to pages.
    
    Args:
        pages_data: List of dicts with format [{"page": int, "text": str}]
        
    Returns:
        List of dicts with format [{"page": int, "section": str, "text": str}]
        
    Section detection logic:
    - Searches for common academic section headers
    - When a header is found, updates the current section
    - Pages without headers inherit the previous section
    - Default section is "Unknown"
    """
    # Define section keywords (order matters for matching priority)
    section_patterns = [
        (r'\babstract\b', 'Abstract'),
        (r'\bintroduction\b', 'Introduction'),
        (r'\bmethodology\b', 'Methodology'),
        (r'\bmethod\b', 'Method'),
        (r'\bresults?\b', 'Results'),
        (r'\bdiscussion\b', 'Discussion'),
        (r'\bconclusion\b', 'Conclusion'),
        (r'\breferences?\b', 'References'),
        (r'\brelated\s+work\b', 'Related Work'),
        (r'\bexperiments?\b', 'Experiments'),
    ]
    
    pages_with_sections = []
    current_section = "Unknown"
    
    for page_data in pages_data:
        page_num = page_data["page"]
        text = page_data["text"]
        
        # Check if this page contains a section header
        detected_section = _detect_section_header(text, section_patterns)
        
        # Update current section if a new header is found
        if detected_section:
            current_section = detected_section
        
        # Assign section to this page
        pages_with_sections.append({
            "page": page_num,
            "section": current_section,
            "text": text
        })
    
    return pages_with_sections


def _detect_section_header(text: str, section_patterns: list) -> str:
    """
    Detect if text contains a section header.
    
    Args:
        text: Page text to search
        section_patterns: List of (regex_pattern, section_name) tuples
        
    Returns:
        Section name if found, None otherwise
        
    Detection strategy:
    - Look for section keywords at the start of lines
    - Case-insensitive matching
    - Prioritize earlier patterns in the list
    """
    # Split text into lines for line-based matching
    lines = text.split('\n')
    
    # Check first 10 lines (section headers are usually at the top)
    for line in lines[:10]:
        line_stripped = line.strip().lower()
        
        # Skip very short lines (likely not headers)
        if len(line_stripped) < 3:
            continue
        
        # Check each section pattern
        for pattern, section_name in section_patterns:
            # Match at start of line or as standalone word
            if re.search(pattern, line_stripped, re.IGNORECASE):
                # Additional check: line should be relatively short (headers are concise)
                if len(line_stripped) < 50:
                    return section_name
    
    return None


if __name__ == "__main__":
    # Test the section detector
    
    # Sample pages data (simulating PDF extraction output)
    test_pages = [
        {"page": 1, "text": "Research Paper Title\nAuthors\n\nAbstract\nThis paper presents..."},
        {"page": 2, "text": "...continued abstract text..."},
        {"page": 3, "text": "1. Introduction\nIn recent years..."},
        {"page": 4, "text": "...more introduction content..."},
        {"page": 5, "text": "2. Methodology\nWe propose a novel approach..."},
        {"page": 6, "text": "3. Results\nTable 1 shows the performance..."},
        {"page": 7, "text": "...additional results..."},
        {"page": 8, "text": "4. Conclusion\nIn this work we demonstrated..."},
    ]
    
    print("Testing Section Detection")
    print("=" * 60)
    
    # Run section detection
    pages_with_sections = detect_sections(test_pages)
    
    # Display results
    for page_data in pages_with_sections:
        print(f"Page {page_data['page']}: {page_data['section']}")
        print(f"  Text preview: {page_data['text'][:60]}...")
        print("-" * 60)
    
    # Summary
    sections_found = set(p['section'] for p in pages_with_sections)
    print(f"\nSections detected: {', '.join(sections_found)}")
