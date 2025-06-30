#!/usr/bin/env python3
"""
Test script for the enhanced GJO scraper
"""

from scrape_and_build_gjo_data import GJOScraper
import json

def test_enhanced_scraper():
    """Test the enhanced scraper with a small subset"""
    
    scraper = GJOScraper(max_workers=1)  # Single worker for testing to avoid rate limiting
    
    # Test with just the first few problems from challenge 97
    print("Testing enhanced scraper...")
    
    # Get basic problems first
    problems = scraper.get_challenge_problems(97, status="resolved")
    
    if not problems:
        print("No problems found!")
        return
    
    # Test with just the first 2 problems to be conservative
    test_problems = problems[:2]
    print(f"Testing with {len(test_problems)} problems...")
    
    # Enrich with details
    enriched_problems = scraper.enrich_problems_with_details(test_problems)
    
    # Save test results
    with open('test_enriched_problems.json', 'w', encoding='utf-8') as f:
        json.dump(enriched_problems, f, indent=2, ensure_ascii=False)
    
    print(f"Test completed. Saved {len(enriched_problems)} enriched problems.")
    
    # Print results
    for i, problem in enumerate(enriched_problems):
        print(f"\n{i+1}. {problem['title']}")
        print(f"   ID: {problem['problem_id']}")
        print(f"   Options: {len(problem.get('options', []))}")
        print(f"   Correct Answer: {problem.get('correct_answer', 'Unknown')}")
        if problem.get('options'):
            print(f"   First few options: {problem['options'][:3]}")

if __name__ == "__main__":
    test_enhanced_scraper() 