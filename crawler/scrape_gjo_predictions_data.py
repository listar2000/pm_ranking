"""
Good Judgment Open (GJO) Predictions Data Scraper

This script extracts user predictions from GJO problem pages using Playwright.
"""

import asyncio
from playwright.async_api import async_playwright
import json
from typing import List, Dict, Optional
import random
from bs4 import BeautifulSoup
import os


class GJOPredictionsScraper:
    """Scraper for Good Judgment Open prediction data using Playwright"""

    def __init__(self, base_url: str = "https://www.gjopen.com"):
        self.base_url = base_url
        self.browser = None
        self.context = None
        self.seen_users = set()  # Track users to only take first prediction
        self.retry_count = 0
        self.max_retries = 3

    async def __aenter__(self):
        """Async context manager entry"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        self.context = await self.browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36'
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    def _get_sleep_time(self, base_time: float = 1.0) -> float:
        """Get adaptive sleep time based on retry count"""
        if self.retry_count == 0:
            return random.uniform(base_time, base_time * 2)
        else:
            # Exponential backoff with jitter
            backoff_time = base_time * \
                (2 ** self.retry_count) + random.uniform(0, 2)
            return min(backoff_time, 5.0)  # Cap at 5 seconds

    async def get_predictions_for_problem(self, problem_id: int, problem_options: List[str], max_pages: int = 10) -> List[Dict]:
        """
        Extract all predictions for a given problem.

        Args:
            problem_id: The problem ID
            problem_options: List of options for this problem (in order)
            max_pages: Maximum number of pages to scrape

        Returns:
            List of prediction dictionaries
        """
        all_predictions = []
        page = 1

        print(f"Scraping predictions for problem {problem_id}...")

        # Create a new page for this problem
        page_obj = await self.context.new_page()  # type: ignore

        try:
            while page <= max_pages:
                try:
                    # Construct the AJAX URL for predictions
                    url = f"{self.base_url}/comments"
                    params = {
                        'commentable_id': problem_id,
                        'commentable_type': 'Forecast::Question',
                        'page': page
                    }

                    # Build the full URL with parameters
                    full_url = f"{url}?commentable_id={problem_id}&commentable_type=Forecast%3A%3AQuestion&page={page}"

                    print(f"  Scraping page {page}: {full_url}")

                    # Navigate to the page
                    response = await page_obj.goto(full_url, wait_until='networkidle')

                    if not response or response.status != 200:
                        print(
                            f"  Failed to load page {page}, status: {response.status if response else 'No response'}")
                        break

                    # Wait for the content to load
                    await page_obj.wait_for_selector('.flyover-comment', timeout=3000)

                    # Get the page content
                    content = await page_obj.content()

                    # Parse with BeautifulSoup
                    soup = BeautifulSoup(content, 'html.parser')

                    # Find all prediction comments
                    prediction_comments = soup.find_all(
                        'div', class_='flyover-comment rationale')

                    if not prediction_comments:
                        print(f"  No more predictions found on page {page}")
                        break

                    page_predictions = []
                    for comment in prediction_comments:
                        prediction_data = self._extract_prediction_from_comment(
                            comment, problem_id, problem_options)
                        if prediction_data:
                            page_predictions.append(prediction_data)

                    all_predictions.extend(page_predictions)
                    print(
                        f"  Found {len(page_predictions)} new predictions on page {page}")

                    # Check if this is the last page (no more comments)
                    if len(prediction_comments) < 10:  # Assuming 10 comments per page
                        print(f"  Fewer than 10 comments found, likely last page")
                        break

                    page += 1

                    # Reset retry count on successful request
                    self.retry_count = 0

                    # Be respectful - add a delay between requests
                    sleep_time = self._get_sleep_time(1.0)
                    await asyncio.sleep(sleep_time)

                except Exception as e:
                    print(f"  Error scraping page {page}: {e}")
                    self.retry_count += 1

                    if self.retry_count <= self.max_retries:
                        print(
                            f"  Retrying... (attempt {self.retry_count}/{self.max_retries})")
                        # sleep_time = self._get_sleep_time(1.0)
                        await asyncio.sleep(2.0)
                        continue
                    else:
                        print(
                            f"  Max retries reached, stopping for problem {problem_id}")
                        break

        finally:
            await page_obj.close()

        print(
            f"Total predictions found for problem {problem_id}: {len(all_predictions)}")
        return all_predictions

    def _extract_prediction_from_comment(self, comment_element, problem_id: int, problem_options: List[str]) -> Optional[Dict]:
        """
        Extract prediction information from a single comment element.

        Args:
            comment_element: BeautifulSoup element representing a prediction comment
            problem_id: The problem ID
            problem_options: List of options for this problem (in order)

        Returns:
            Dictionary with prediction information or None if extraction fails
        """
        try:
            # Extract username
            username_element = comment_element.find(
                'span', class_='membership-username')
            if not username_element:
                return None

            username = username_element.get_text(strip=True)

            # Check if we've seen this user before (only take first prediction)
            user_key = f"{problem_id}_{username}"
            if user_key in self.seen_users:
                return None

            # Extract timestamp
            timestamp_element = comment_element.find(
                'span', attrs={'data-localizable-timestamp': True})
            timestamp = timestamp_element.get(
                'data-localizable-timestamp') if timestamp_element else None

            # Extract prediction values as list
            prediction_values = self._extract_prediction_values_as_list(
                comment_element, problem_options)

            if not prediction_values:
                return None

            # Mark this user as seen
            self.seen_users.add(user_key)

            return {
                'problem_id': problem_id,
                'username': username,
                'timestamp': timestamp,
                'prediction': prediction_values
            }

        except Exception as e:
            print(f"Error extracting prediction: {e}")
            return None

    def _extract_prediction_values_as_list(self, comment_element, problem_options: List[str]) -> Optional[List[float]]:
        """
        Extract prediction values as a list of probabilities in the same order as problem_options.

        Args:
            comment_element: BeautifulSoup element representing a prediction comment
            problem_options: List of options for this problem (in order)

        Returns:
            List of probabilities in the same order as problem_options
        """
        try:
            # Create a mapping of option text to probability
            option_to_prob = {}

            # Find the prediction values container
            values_container = comment_element.find(
                'div', class_='prediction-values')
            if not values_container:
                return None

            # Find all prediction rows
            prediction_rows = values_container.find_all(
                'div', class_='row row-condensed')

            for row in prediction_rows:
                # Skip the header row
                if row.find('div', class_='info-heading'):
                    continue

                # Extract probability and answer
                probability_col = row.find('div', class_='probability-col')
                answer_col = row.find('div', class_='answer-col')

                if probability_col and answer_col:
                    # Extract probability value
                    probability_element = probability_col.find(
                        'span', class_='probability-value')
                    if probability_element:
                        probability_text = probability_element.get_text(
                            strip=True)
                        # Remove % sign and convert to float
                        probability = float(
                            probability_text.replace('%', '')) / 100.0

                        # Extract answer text
                        answer_text = answer_col.get_text(strip=True)

                        option_to_prob[answer_text] = probability

            if not option_to_prob:
                return None

            # Convert to list in the same order as problem_options
            prediction_list = []
            for option in problem_options:
                prediction_list.append(option_to_prob.get(option, 0.0))

            return prediction_list

        except Exception as e:
            print(f"Error extracting prediction values: {e}")
            return None

    async def get_predictions_for_all_problems(self, problems_file: str, output_file: str, max_pages_per_problem: int = 10, start_from_problem: int = -1) -> int:
        """
        Extract predictions for all problems from the enriched problems JSON file and stream them to a file.

        Args:
            problems_file: Path to the JSON file with enriched problems
            output_file: Path to the output JSON file for streaming predictions
            max_pages_per_problem: Maximum pages to scrape per problem

        Returns:
            Total number of predictions collected
        """
        print(f"Loading problems from {problems_file}...")

        # Load the enriched problems
        with open(problems_file, 'r', encoding='utf-8') as f:
            problems = json.load(f)

        if start_from_problem > 0:
            # we only consider problems with id <= start_from_problem
            problems = [p for p in problems if int(
                p['problem_id']) <= start_from_problem]
            print(
                f"Continue from problem {start_from_problem}, {len(problems)} problems to process")
        else:
            print(f"Found {len(problems)} problems to process")

        # Initialize the output file with an empty array if it doesn't exist
        if not os.path.exists(output_file):
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('[\n')
                f.flush()

        total_predictions = 0

        for i, problem in enumerate(problems):
            problem_id = int(problem['problem_id'])
            problem_options = problem.get('options', [])

            if not problem_options:
                print(f"Skipping problem {problem_id}: no options found")
                continue

            print(f"\nProcessing problem {i+1}/{len(problems)}: {problem_id}")
            print(f"Options: {problem_options}")

            try:
                predictions = await self.get_predictions_for_problem(problem_id, problem_options, max_pages_per_problem)

                # Stream predictions for this problem to the file
                if predictions:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        for j, prediction in enumerate(predictions):
                            # Add comma if not the first prediction overall
                            if total_predictions > 0 or j > 0:
                                f.write(',\n')

                            json.dump(prediction, f, ensure_ascii=False,
                                      separators=(',', ': '))
                            f.flush()  # Ensure data is written immediately

                    total_predictions += len(predictions)
                    print(
                        f"✓ Completed problem {problem_id}: {len(predictions)} predictions (Total: {total_predictions})")

                self.retry_count = 0
                # Add delay between problems
                sleep_time = self._get_sleep_time(1.0)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                print(f"✗ Error processing problem {problem_id}: {e}")
                continue

        # Close the JSON array
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write('\n]')
            f.flush()

        print(f"\nTotal predictions collected: {total_predictions}")
        return total_predictions


async def scrape_all_predictions(start_from_problem: int = -1):
    """Scrape predictions for all problems"""
    async with GJOPredictionsScraper() as scraper:
        # Use the enriched problems file
        problems_file = "challenge_97_problems_enriched.json"

        try:
            output_file = "all_predictions.json"
            total_predictions = await scraper.get_predictions_for_all_problems(problems_file, output_file, max_pages_per_problem=100, start_from_problem=start_from_problem)

            print(f"Saved {total_predictions} predictions to {output_file}")

            # Print some statistics
            # Note: We can't easily get unique users/problems without loading the file
            # since we're streaming. For now, just report the total count.
            print(f"Total predictions: {total_predictions}")

        except FileNotFoundError:
            print(
                f"Problems file {problems_file} not found. Please run the problem scraper first.")


async def test_single_problem():
    """Test scraping predictions for a single problem"""
    async with GJOPredictionsScraper() as scraper:
        # Test with problem 3700 (the one from the example)
        problem_id = 3700
        problem_options = [
            "Arizona", "Iowa State", "Kansas", "Kansas State",
            "Oklahoma State", "Texas Christian University (TCU)",
            "University of Central Florida (UCF)", "Utah", "Another team"
        ]

        predictions = await scraper.get_predictions_for_problem(problem_id, problem_options, max_pages=2)

        # Save results
        output_file = f"predictions_problem_{problem_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(predictions)} predictions to {output_file}")

        # Print first few predictions as example
        print("\nFirst 3 predictions:")
        for i, prediction in enumerate(predictions[:3]):
            print(f"{i+1}. User: {prediction['username']}")
            print(f"   Timestamp: {prediction['timestamp']}")
            print(f"   Prediction: {prediction['prediction']}")
            print()


async def main():
    """Main function to demonstrate the scraper"""
    # Uncomment one of these:
    # await test_single_problem()  # Test with single problem
    # Scrape all problems, start from the beginning (-1)
    await scrape_all_predictions(start_from_problem=-1)


if __name__ == "__main__":
    asyncio.run(main())
