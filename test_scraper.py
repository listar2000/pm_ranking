#!/usr/bin/env python3
"""
Test script for the GJO scraper using the existing HTML file
"""

from bs4 import BeautifulSoup
import re
from typing import Dict, Optional

def test_html_parsing():
    """Test parsing the existing HTML file"""
    
    # Read the existing HTML file
    with open('crawler/html/challenge_97_3.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all problem rows
    problem_rows = soup.find_all('div', class_='question-row-component')
    print(f"Found {len(problem_rows)} problem rows")
    
    problems = []
    for row in problem_rows:
        problem_info = extract_problem_info(row)
        if problem_info:
            problems.append(problem_info)
    
    print(f"Successfully extracted {len(problems)} problems")
    
    # Print first few problems
    print("\nFirst 3 problems:")
    for i, problem in enumerate(problems[:3]):
        print(f"{i+1}. {problem['title']}")
        print(f"   ID: {problem['problem_id']}")
        print(f"   URL: {problem['url']}")
        print(f"   Status: {problem['metadata'].get('status', 'Unknown')}")
        print(f"   Forecasters: {problem['metadata'].get('num_forecasters', 'Unknown')}")
        print(f"   Forecasts: {problem['metadata'].get('num_forecasts', 'Unknown')}")
        print()
    
    return problems

def extract_problem_info(row_element) -> Optional[Dict]:
    """Extract problem information from a single problem row element."""
    try:
        # Extract problem ID from the row ID
        row_id = row_element.get('id', '')
        problem_id_match = re.search(r'row-table-question-(\d+)', row_id)
        if not problem_id_match:
            return None
        
        problem_id = problem_id_match.group(1)
        
        # Find the problem link
        link_element = row_element.find('h5').find('a')
        if not link_element:
            return None
        
        problem_url = link_element.get('href')
        problem_title = link_element.get_text(strip=True)
        
        # Extract metadata
        metadata = extract_problem_metadata(row_element)
        
        return {
            'problem_id': problem_id,
            'title': problem_title,
            'url': problem_url,
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"Error extracting problem info: {e}")
        return None

def extract_problem_metadata(row_element) -> Dict:
    """Extract additional metadata from a problem row."""
    metadata = {}
    
    try:
        # Extract status
        status_element = row_element.find('span', class_='info-heading')
        if status_element:
            metadata['status'] = status_element.get_text(strip=True)
        
        # Extract end date
        end_date_element = row_element.find('span', attrs={'data-localizable-timestamp': True})
        if end_date_element:
            metadata['end_date'] = end_date_element.get('data-localizable-timestamp')
        
        # Extract number of forecasters
        forecasters_element = row_element.find('a', attrs={'data-sort': 'predictors_count'})
        if forecasters_element:
            forecasters_text = forecasters_element.get_text(strip=True)
            metadata['num_forecasters'] = int(forecasters_text.split()[0])
        
        # Extract number of forecasts
        forecasts_element = row_element.find('a', attrs={'data-sort': 'prediction_sets_count'})
        if forecasts_element:
            forecasts_text = forecasts_element.get_text(strip=True)
            metadata['num_forecasts'] = int(forecasts_text.split()[0])
            
    except Exception as e:
        print(f"Error extracting metadata: {e}")
    
    return metadata

if __name__ == "__main__":
    test_html_parsing() 