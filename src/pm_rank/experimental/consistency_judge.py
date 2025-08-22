"""
Experimental Consistency Judge: 
using OpenRouter LLM-as-a-judge to supplement consistency information for prediction problems.
"""
try:
    from dotenv import load_dotenv
    load_dotenv()
    from openai import OpenAI
    from tenacity import retry, stop_after_attempt, wait_random_exponential
except ImportError:
    print("For this experimental feature, additional dependencies are required. Please install them with `pip install openai dotenv`.")
    exit(1)

import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from pydantic import BaseModel

from pm_rank.experimental.prompts import TASK_TO_PROMPT_DICT
from pm_rank.data import ForecastProblem


# define the processing functions based on the task
def process_logical_chain_output(output: BaseModel) -> Dict[str, Any]:
    """
    Process the output of the logical chain task.
    """
    return {
        "has_chain": output.has_chain,
        "chains": output.chains
    }

def process_mutually_exclusive_output(output: BaseModel) -> Dict[str, Any]:
    """
    Process the output of the mutually exclusive task.
    """
    return {
        "has_exclusive_set": output.has_exclusive_set,
        "exclusive_sets": output.exclusive_sets,
        "is_comprehensive": output.is_comprehensive
    }

TASK_TO_PROCESS_FN_DICT = {
    "logical-chain": process_logical_chain_output,
    "mutually-exclusive": process_mutually_exclusive_output
}


class OpenRouterConsistencyJudge:
    def __init__(self, model: str = "gpt-4o-mini", task: str = "logical-chain"):
        self.model = model
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        # TODO: add more tasks
        assert task in TASK_TO_PROMPT_DICT.keys(), f"Invalid task: {task}"
        self.task = task
        self.process_fn = TASK_TO_PROCESS_FN_DICT[task]

        self.developer_prompt = TASK_TO_PROMPT_DICT[task]["developer_prompt"]
        self.user_prompt = TASK_TO_PROMPT_DICT[task]["user_prompt"]
        self.response_format = TASK_TO_PROMPT_DICT[task]["response_format"]

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
    async def async_completion_with_retry(self, **kwargs):
        # Use thread pool executor to run the synchronous client in a non-blocking way
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: self.client.beta.chat.completions.parse(**kwargs)
            ),
            timeout=700.0
        )

    async def async_judge_single_problem(self, problem: ForecastProblem) -> BaseModel:
        try:
            outcome_list_json = json.dumps(problem.options)
            prompt = self.user_prompt.format(event_text=problem.title, outcome_list=outcome_list_json)

            response = await self.async_completion_with_retry(
                model=self.model,
                messages=[{"role": "developer", "content": self.developer_prompt}, {"role": "user", "content": prompt}],
                response_format=self.response_format
            )
            
            llm_output = response.choices[0].message.parsed

            if response.choices[0].message.refusal:
                raise ValueError(f"LLM refusal: {response.choices[0].message.refusal}")
            
            # If parsed output is None, try to parse the content manually
            if llm_output is None:
                content = response.choices[0].message.content
                if content:
                    try:
                        parsed_content = json.loads(content)
                        # Create an instance of the response format class
                        llm_output = self.response_format(**parsed_content)
                    except (json.JSONDecodeError, TypeError) as e:
                        raise ValueError(f"Failed to parse LLM response: {e}. Content: {content}")
                else:
                    raise ValueError("LLM returned empty content")

            return llm_output
        except asyncio.TimeoutError as e:
            raise ValueError(f"OpenRouter API timeout: {e}")
        except Exception as e:
            # provide richer information based on the error
            error_msg = str(e)
            if "429" in error_msg:
                raise ValueError(f"OpenRouter API rate limit exceeded: {error_msg} for model {self.model}")
            elif "500" in error_msg:
                raise ValueError(f"OpenRouter API server error: {error_msg} for model {self.model}")
            elif "timeout" in error_msg.lower():
                raise ValueError(f"OpenRouter API timeout: {error_msg} for model {self.model}")
            else:
                raise ValueError(f"OpenRouter API error: {error_msg} for model {self.model}")

    async def async_judge_multiple_problems(
        self, 
        problems: List[ForecastProblem], 
        output_file: str = None,
        max_concurrent: int = 5
    ) -> Dict[str, Any]:
        """
        Process multiple problems and return/save results.
        
        Args:
            problems: List of ForecastProblem objects to process
            output_file: Optional path to save results as JSON
            max_concurrent: Maximum number of concurrent API calls
            
        Returns:
            Dictionary containing results for all problems
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_problem(problem: ForecastProblem) -> Dict[str, Any]:
            async with semaphore:
                try:
                    print(f"Processing problem: {problem.problem_id}")
                    result = await self.async_judge_single_problem(problem)
                    return {
                        "problem_id": problem.problem_id,
                        "title": problem.title,
                        "options": problem.options,
                        "status": "success",
                        "result": self.process_fn(result),
                        "error": None
                    }
                except Exception as e:
                    print(f"Error processing problem {problem.problem_id}: {e}")
                    return {
                        "problem_id": problem.problem_id,
                        "title": problem.title,
                        "options": problem.options,
                        "status": "error",
                        "result": None,
                        "error": str(e)
                    }
        
        # Process all problems concurrently
        print(f"Starting batch processing of {len(problems)} problems with max {max_concurrent} concurrent requests...")
        tasks = [process_single_problem(problem) for problem in problems]
        results = await asyncio.gather(*tasks)
        
        # Organize results
        batch_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "task": self.task,
                "total_problems": len(problems),
                "successful": sum(1 for r in results if r["status"] == "success"),
                "failed": sum(1 for r in results if r["status"] == "error")
            },
            "results": results
        }
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")
        
        print(f"Batch processing completed: {batch_results['metadata']['successful']}/{batch_results['metadata']['total_problems']} successful")
        
        return batch_results



# local testing
if __name__ == "__main__":
    from pm_rank.data import ProphetArenaChallengeLoader

    judge = OpenRouterConsistencyJudge(model="google/gemini-2.5-flash", task="mutually-exclusive")

    data_file = "/net/scratch2/listar2000/pm_ranking/slurm/latest_predictions_df_08_20.csv"

    challenge_loader = ProphetArenaChallengeLoader(predictions_file=data_file, use_bid_for_odds=False)

    prophet_challenge = challenge_loader.load_challenge(add_market_baseline=True)

    prophet_problems = prophet_challenge.get_problems()
    
    # actually run the async function
    max_concurrent = 20
    llm_output = asyncio.run(judge.async_judge_multiple_problems(prophet_problems, 
        output_file="prophet_problems_mutually_exclusive_output.json", max_concurrent=max_concurrent))