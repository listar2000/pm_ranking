"""
Good Judgment Open (GJO) Problem Data Scraper

This script extracts problem URLs from GJO challenge pages to be used for the
prediction data scraper in `scrape_gjo_predictions_data.py`.
"""

import requests
from bs4 import BeautifulSoup
import time
import json
from typing import List, Dict, Optional
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random


class GJOProblemScraper:
    """Scraper for Good Judgment Open website"""

    def __init__(self, base_url: str = "https://www.gjopen.com", max_workers: int = 2):
        self.base_url = base_url
        self.session = requests.Session()
        self.max_workers = max_workers

        # Set up headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

        # Thread-local storage for sessions
        self._local = threading.local()

    def _get_session(self):
        """Get thread-local session"""
        if not hasattr(self._local, 'session'):
            self._local.session = requests.Session()
            self._local.session.headers.update(self.session.headers)
        return self._local.session

    def _random_sleep(self, min_seconds: float = 1.0, max_seconds: float = 3.0):
        """Sleep for a random amount of time to avoid rate limiting"""
        sleep_time = random.uniform(min_seconds, max_seconds)
        time.sleep(sleep_time)

    def get_challenge_problems(self, challenge_id: int, status: str = "resolved") -> List[Dict]:
        """
        Extract all problem URLs from a challenge page.

        Args:
            challenge_id: The challenge ID
            status: Filter by status (resolved, active, pending, etc.)

        Returns:
            List of dictionaries containing problem information
        """
        problems = []
        page = 1

        while True:
            # Construct the challenge URL
            url = f"{self.base_url}/challenges/{challenge_id}?status={status}&page={page}"
            print(f"Scraping page {page}: {url}")

            try:
                response = self.session.get(url)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')

                # Find all problem rows
                problem_rows = soup.find_all(
                    'div', class_='question-row-component')

                if not problem_rows:
                    print(f"No more problems found on page {page}")
                    break

                for row in problem_rows:
                    problem_info = self._extract_problem_info(row)
                    if problem_info:
                        problems.append(problem_info)

                # Check if there's a next page
                next_link = soup.find('a', rel='next')
                if not next_link:
                    print(f"No next page found, stopping at page {page}")
                    break

                page += 1

                # Be respectful - add a longer delay between requests
                self._random_sleep(0.5, 1.0)

            except requests.RequestException as e:
                print(f"Error scraping page {page}: {e}")
                if "Too Many Requests" in str(e):
                    print("Rate limited! Waiting longer...")
                    time.sleep(20)  # Waitx 20 seconds if rate limited
                break

        print(f"Total problems found: {len(problems)}")
        return problems

    def enrich_problems_with_details(self, problems: List[Dict]) -> List[Dict]:
        """
        Enrich problem information by fetching details from individual problem pages.
        Uses parallel processing to speed up the process.

        Args:
            problems: List of basic problem information

        Returns:
            List of enriched problem information
        """
        print(f"Enriching {len(problems)} problems with details...")

        enriched_problems = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_problem = {
                executor.submit(self._fetch_problem_details, problem): problem
                for problem in problems
            }

            # Process completed tasks
            for future in as_completed(future_to_problem):
                problem = future_to_problem[future]
                try:
                    enriched_problem = future.result()
                    if enriched_problem:
                        enriched_problems.append(enriched_problem)
                        print(
                            f"✓ Enriched problem {enriched_problem['problem_id']}: {enriched_problem['title'][:50]}...")
                    else:
                        print(
                            f"✗ Failed to enrich problem {problem['problem_id']}")
                except Exception as e:
                    print(
                        f"✗ Error enriching problem {problem['problem_id']}: {e}")
                    if "Too Many Requests" in str(e):
                        print(
                            "Rate limited during enrichment! Consider reducing workers or increasing delays.")

        print(f"Successfully enriched {len(enriched_problems)} problems")
        return enriched_problems

    def _fetch_problem_details(self, problem: Dict) -> Optional[Dict]:
        """
        Fetch detailed information for a single problem.

        Args:
            problem: Basic problem information

        Returns:
            Enriched problem information or None if failed
        """
        try:
            session = self._get_session()
            response = session.get(problem['url'])
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract options and correct answer
            options, correct_answer = self._extract_problem_options(soup)

            # Add the new information to the problem
            enriched_problem = problem.copy()
            enriched_problem['options'] = options
            enriched_problem['correct_answer'] = correct_answer

            # Longer delay to be more respectful
            self._random_sleep(1.5, 3.0)

            return enriched_problem

        except requests.RequestException as e:
            if "Too Many Requests" in str(e):
                print(
                    f"Rate limited for problem {problem['problem_id']}, waiting...")
                time.sleep(60)  # Wait 1 minute if rate limited
            print(
                f"Error fetching details for problem {problem['problem_id']}: {e}")
            return None
        except Exception as e:
            print(
                f"Error fetching details for problem {problem['problem_id']}: {e}")
            return None

    def _extract_problem_options(self, soup: BeautifulSoup) -> tuple[List[str], Optional[str]]:
        """
        Extract problem options and correct answer from a problem page.

        Args:
            soup: BeautifulSoup object of the problem page

        Returns:
            Tuple of (options_list, correct_answer)
        """
        options = []
        correct_answer = None

        try:
            # Find the consensus table
            table = soup.find(
                'table', class_='consensus-table')  # type: ignore
            if not table:
                return options, correct_answer

            # Find all rows in the tbody
            tbody = table.find('tbody')  # type: ignore
            if not tbody:
                return options, correct_answer

            rows = tbody.find_all('tr')  # type: ignore

            for row in rows:
                cells = row.find_all('td')  # type: ignore
                if len(cells) >= 3:
                    option_text = cells[0].get_text(strip=True)
                    options.append(option_text)

                    # Check if this option is correct (has checkmark icon)
                    correct_cell = cells[1]
                    checkmark = correct_cell.find(
                        'i', class_='fa-check-circle')  # type: ignore
                    if checkmark:
                        correct_answer = option_text

            return options, correct_answer

        except Exception as e:
            print(f"Error extracting options: {e}")
            return options, correct_answer

    def _extract_problem_info(self, row_element) -> Optional[Dict]:
        """
        Extract problem information from a single problem row element.

        Args:
            row_element: BeautifulSoup element representing a problem row

        Returns:
            Dictionary with problem information or None if extraction fails
        """
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
            metadata = self._extract_problem_metadata(row_element)

            return {
                'problem_id': problem_id,
                'title': problem_title,
                'url': problem_url,
                'metadata': metadata
            }

        except Exception as e:
            print(f"Error extracting problem info: {e}")
            return None

    def _extract_problem_metadata(self, row_element) -> Dict:
        """
        Extract additional metadata from a problem row.

        Args:
            row_element: BeautifulSoup element representing a problem row

        Returns:
            Dictionary with metadata
        """
        metadata = {}

        try:
            # Extract status
            status_element = row_element.find('span', class_='info-heading')
            if status_element:
                metadata['status'] = status_element.get_text(strip=True)

            # Extract end date
            end_time_element = row_element.find(
                'span', attrs={'data-localizable-timestamp': True})
            if end_time_element:
                metadata['end_time'] = end_time_element.get(
                    'data-localizable-timestamp')

            # Extract number of forecasters
            forecasters_element = row_element.find(
                'a', attrs={'data-sort': 'predictors_count'})
            if forecasters_element:
                forecasters_text = forecasters_element.get_text(strip=True)
                metadata['num_forecasters'] = int(forecasters_text.split()[0])

            # Extract number of forecasts
            forecasts_element = row_element.find(
                'a', attrs={'data-sort': 'prediction_sets_count'})
            if forecasts_element:
                forecasts_text = forecasts_element.get_text(strip=True)
                metadata['num_forecasts'] = int(forecasts_text.split()[0])

        except Exception as e:
            print(f"Error extracting metadata: {e}")

        return metadata


def main():
    """Main function to demonstrate the scraper"""
    scraper = GJOProblemScraper(
        max_workers=1)  # Single worker to avoid rate limiting

    # Example: Extract problems from challenge 97 (Sportsball Challenge)
    challenge_id = 97
    print(f"Scraping challenge {challenge_id}...")

    # Step 1: Get basic problem information
    problems = scraper.get_challenge_problems(challenge_id, status="resolved")

    # Step 2: Enrich with detailed information
    enriched_problems = scraper.enrich_problems_with_details(problems)

    # Save results to JSON file
    output_file = f"data/{challenge_id}_challenge_metadata.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_problems, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(enriched_problems)} enriched problems to {output_file}")

    # Print first few problems as example
    print("\nFirst 3 enriched problems:")
    for i, problem in enumerate(enriched_problems[:3]):
        print(f"{i+1}. {problem['title']}")
        print(f"   ID: {problem['problem_id']}")
        print(f"   URL: {problem['url']}")
        print(f"   Status: {problem['metadata'].get('status', 'Unknown')}")
        print(f"   Options: {len(problem.get('options', []))}")
        print(f"   Correct Answer: {problem.get('correct_answer', 'Unknown')}")
        print(
            f"   Forecasters: {problem['metadata'].get('num_forecasters', 'Unknown')}")
        print(
            f"   Forecasts: {problem['metadata'].get('num_forecasts', 'Unknown')}")
        print()


if __name__ == "__main__":
    main()
