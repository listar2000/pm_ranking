"""
Defining the prediction market data structure and functions to load them 
from different types of data sources.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Iterator, Literal, Tuple, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime, timedelta
from functools import cached_property
import math, random

SMOOTH_ODDS_EPS = 5e-3

class ForecastEvent(BaseModel):
    """Individual forecast from a user for a specific problem."""
    forecast_id: str = Field(description="The unique identifier of the forecast")
    problem_id: str = Field(description="The id of the problem")
    username: str = Field(description="The user name/id of the forecaster")
    timestamp: datetime = Field(description="The timestamp of the forecast")
    probs: List[float] = Field(description="The forecasted probabilities for each option")
    unnormalized_probs: Optional[List[float]] = Field(default=None, description="The unnormalized forecasted probabilities for each option")
    weight: float = Field(description="The weight of the forecast. This is used to weight the forecast in scoring/ranking. Default to 1.", default=1.0)

    @field_validator('weight')
    def validate_weight(cls, v):
        """Validate that weight is non-negative."""
        if v < 0:
            raise ValueError("Weight must be non-negative")
        return v

    @field_validator('probs')
    def validate_probabilities(cls, v):
        """Validate that probabilities sum to 1 and are non-negative."""
        if not v:
            raise ValueError("Probabilities list cannot be empty")
        if not all(0 <= p <= 1 for p in v):
            raise ValueError("All probabilities must be between 0 and 1")
        if not math.isclose(sum(v), 1.0, abs_tol=1e-6):
            raise ValueError(f"Probabilities must sum to 1, got {sum(v)}")
        return v

    @model_validator(mode='after')
    def set_unnormalized_probs_default(self):
        """Set unnormalized_probs to probs if not provided."""
        if self.unnormalized_probs is None:
            self.unnormalized_probs = self.probs
        return self

    @field_validator('unnormalized_probs')
    def validate_unnormalized_probabilities(cls, v):
        """Validate that unnormalized probabilities are non-negative.
        we only require every number to be in [0, 1], and the vector dimension is the same as the number of options.
        """
        if v is None:
            return v
        
        if not all(0 <= p <= 1 for p in v):
            raise ValueError("All unnormalized probabilities must be in [0, 1]")
     
        return v


class ProphetArenaForecastEvent(ForecastEvent):
    """Specialized forecast event for Prophet Arena."""
    submission_id: str = Field(description="The id of the submission batch. Note this might not be unique across different forecasters.")


class ForecastProblem(BaseModel):
    """A prediction problem with multiple options and forecasts."""
    title: str = Field(description="The title of the problem")
    problem_id: str = Field(description="The id of the problem")
    options: List[str] = Field(description="The available options for the problem")
    correct_option_idx: List[int] = Field(description="The indices of the correct answer, might be multiple ones")
    forecasts: List[ForecastEvent] = Field(description="All forecasts for this problem")
    end_time: datetime = Field(description="The end time of the problem")
    num_forecasters: int = Field(description="The number of forecasters")
    url: Optional[str] = Field(None, description="The URL of the problem")
    # TODO: in the future, the problem might have mutiple `odds`/`no_odds` depending on the submission_id of its events.
    odds: Optional[List[float]] = Field(None, description="The odds for each option")
    no_odds: Optional[List[float]] = Field(None, description="The odds for each option to not realize")
    category: Optional[str] = Field(None, description="The category of the problem")

    @field_validator('correct_option_idx')
    def validate_correct_option_idx(cls, v, info):
        """Validate that correct_option_idx is in the options list."""
        if not all(0 <= idx < len(info.data['options']) for idx in v):
            raise ValueError(f"All correct_option_idx must be in [0, {len(info.data['options']) - 1}]")
        
        if not all(isinstance(idx, int) for idx in v):
            raise ValueError("All correct_option_idx must be integers")
        
        if not len(set(v)) == len(v):
            raise ValueError("All correct_option_idx must be unique")
        
        return v

    @field_validator('forecasts')
    def validate_forecasts(cls, v, info):
        """Validate that all forecasts have (1) correct number of probabilities, (2) unique `forecast_id`."""
        if info.data and 'options' in info.data:
            expected_length = len(info.data['options'])
            for forecast in v:
                if len(forecast.probs) != expected_length:
                    raise ValueError(
                        f"Forecast by {forecast.username} has {len(forecast.probs)} probabilities, "
                        f"expected {expected_length}"
                    )

        forecast_id_set = set()
        for forecast in v:
            if forecast.forecast_id in forecast_id_set:
                raise ValueError(f"Forecast by {forecast.username} has duplicate `forecast_id`: {forecast.forecast_id}")
            forecast_id_set.add(forecast.forecast_id)
        
        return v

    @field_validator('odds')
    def validate_odds(cls, v, info):
        """Validate that odds match the number of options if provided."""
        if v is not None and info.data and 'options' in info.data:
            if len(v) != len(info.data['options']):
                raise ValueError(f"Number of odds ({len(v)}) must match number of options ({len(info.data['options'])})")

        # check that odds each has to be in [0, 1]
        if v is not None and not all(0 <= p <= 1 for p in v):
            raise ValueError("All odds (implied probabilities) must be in [0, 1]")
        return v

    @model_validator(mode='after')
    def smooth_odds(self):
        """Smooth the odds to not be too close to 0 or 1.
        """
        if self.odds is not None:
            self.odds = [max(SMOOTH_ODDS_EPS, min(1 - SMOOTH_ODDS_EPS, odd)) for odd in self.odds]

        if self.no_odds is not None:
            self.no_odds = [max(SMOOTH_ODDS_EPS, min(1 - SMOOTH_ODDS_EPS, odd)) for odd in self.no_odds]
        return self

    @property
    def has_odds(self) -> bool:
        """Check if the problem has odds data."""
        return self.odds is not None and len(self.odds) > 0

    @property
    def has_no_odds(self) -> bool:
        """Check if the problem has no_odds data."""
        return self.no_odds is not None and len(self.no_odds) > 0
    
    @cached_property
    def crowd_probs(self) -> List[float]:
        """Calculate crowd probabilities from the forecasts."""
        if not self.forecasts:
            return []
        
        # Calculate average probability for each option across all forecasts
        num_options = len(self.options)
        crowd_probs = [0.0] * num_options
        
        for forecast in self.forecasts:
            for i, prob in enumerate(forecast.probs):
                crowd_probs[i] += prob
        
        # Normalize by number of forecasts
        num_forecasts = len(self.forecasts)
        if num_forecasts > 0:
            crowd_probs = [prob / num_forecasts for prob in crowd_probs]
        
        return crowd_probs

    @cached_property
    def unique_forecasters(self) -> List[str]:
        """Get list of unique forecasters for this problem."""
        return list(set(forecast.username for forecast in self.forecasts))

    @cached_property
    def option_payoffs(self) -> List[Tuple[int, float]]:
        """
        Obtain a sorted list of (option_idx, payoff) for this problem.
        The payoff is the (1 / odds) for a correct option, and 0 for an incorrect option.
        """
        if not self.has_odds:
            return []
        
        option_payoffs = []
        for option_idx, odd in enumerate(self.odds):
            payoff = 1 / odd if option_idx in self.correct_option_idx else 0
            option_payoffs.append((option_idx, payoff))
        
        return sorted(option_payoffs, key=lambda x: x[1], reverse=True)


class ForecastChallenge(BaseModel):
    """
    A collection of forecast problems with validation and computed properties.
    """
    title: str = Field(description="The title of the challenge")
    forecast_problems: List[ForecastProblem] = Field(description="The list of forecast problems")
    categories: Optional[List[str]] = Field(None, description="The categories of the challenge")

    @field_validator('forecast_problems')
    def validate_problems(cls, v):
        """Validate that there are problems and they have unique IDs."""
        if not v:
            raise ValueError("Challenge must have at least one problem")
        
        problem_ids = [p.problem_id for p in v]
        if len(problem_ids) != len(set(problem_ids)):
            raise ValueError("All problems must have unique IDs")
        return v

    @field_validator('categories')
    def validate_categories(cls, v, info):
        """Validate that categories are a list of strings."""
        if v is not None:
            v_set = set(v)
            if not len(v_set) == len(v):
                raise ValueError("All categories must be unique")
            # check all the `category` in the `forecast_problems` are in the `categories`
            for problem in info.data['forecast_problems']:
                if problem.category is not None and problem.category not in v_set:
                    raise ValueError(f"Category {problem.category} is not in the categories list")
        return v

    @cached_property
    def forecaster_map(self) -> Dict[str, List[ForecastEvent]]:
        """Map from forecaster username to their forecasts across all problems."""
        forecaster_map = {}
        for problem in self.forecast_problems:
            for forecast in problem.forecasts:
                if forecast.username not in forecaster_map:
                    forecaster_map[forecast.username] = []
                forecaster_map[forecast.username].append(forecast)
        return forecaster_map

    @cached_property
    def num_forecasters(self) -> int:
        """Total number of unique forecasters across all problems."""
        return len(self.forecaster_map)

    @cached_property
    def unique_forecasters(self) -> List[str]:
        """List of unique forecaster usernames."""
        return list(self.forecaster_map.keys())

    @cached_property
    def problem_option_payoffs(self) -> Dict[str, List[Tuple[int, float]]]:
        """Map from problem_id to a sorted list of (option_idx, payoff) for this problem."""
        return {problem.problem_id: problem.option_payoffs for problem in self.forecast_problems}

    def get_forecaster_problems(self, username: str) -> List[ForecastProblem]:
        """Get all problems that a specific forecaster participated in."""
        forecaster_problem_ids = {
            forecast.problem_id for forecast in self.forecaster_map.get(username, [])
        }
        return [p for p in self.forecast_problems if p.problem_id in forecaster_problem_ids]

    def get_problem_by_id(self, problem_id: str) -> Optional[ForecastProblem]:
        """Get a specific problem by its ID."""
        for problem in self.forecast_problems:
            if problem.problem_id == problem_id:
                return problem
        return None

    def get_problems(self, nums: int = -1) -> List[ForecastProblem]:
        """Get a list of problems. If nums is -1, return all problems."""
        if nums == -1:
            return self.forecast_problems
        return self.forecast_problems[:nums]

    def stream_problems(self, order: Literal["sequential", "random", "time"] = "sequential", increment: int = 100) \
        -> Iterator[List[ForecastProblem]]:
        """
        Stream the problems in the challenge. Either by random or by the problem end time.

        Args:
            order: The order in which to stream the problems.
            increment: The number of problems to stream in each iteration.

        Returns:
            An iterator of lists of problems.
        """
        full_problems = self.forecast_problems.copy()
        if order == "random":
            random.shuffle(full_problems)
        elif order == "time":
            full_problems.sort(key=lambda x: x.end_time.replace(tzinfo=None))

        for i in range(0, len(full_problems), increment):
            yield full_problems[i:i+increment]

    def stream_problems_over_time(
        self, increment_by: Literal["day", "week", "month"] = "day", min_bucket_size: int = 1,
    ) -> Iterator[Tuple[str, List["ForecastProblem"]]]:
        """Stream all problems in chronological buckets."""
        return self._stream_problems_over_time(
            problems=self.forecast_problems,
            increment_by=increment_by,
            min_bucket_size=min_bucket_size
        )

    @staticmethod
    def _stream_problems_over_time(
        problems: Optional[List[ForecastProblem]] = None, increment_by: Literal["day", "week", "month"] = "day",
        min_bucket_size: int = 1,
    ) -> Iterator[Tuple[str, List["ForecastProblem"]]]:
        """Stream all problems in chronological buckets.

        Each bucket covers a contiguous time window of length *increment_by* (day, week, or
        month).  If the window does **not** yet contain *min_bucket_size* problems, the
        window is repeatedly extended by another *increment_by* until the size
        requirement is met **or** no problems remain.  All problems whose ``end_time`` is
        **strictly after** the previous bucket boundary *and* **≤** the current bucket
        boundary are included.

        The timestamp returned for a bucket is the *inclusive* upper‐bound boundary
        expressed in ISO‑8601 (YYYY‑MM‑DD).

        Args:
            increment_by: The time interval to stream problems in a bucket.
            min_bucket_size: The minimum number of problems to stream in each bucket.

        Returns:
            An iterator where each element is a bucket of (timestamp, list of problems).
        """
        assert min_bucket_size > 0, "min_bucket_size must be greater than 0"
        if not problems:
            return  # Nothing to yield

        # 1. Sort once so we can consume the list with a monotone pointer.
        full_problems = sorted(problems, key=lambda p: p.end_time)

        # 2. Helper that advances a *date* by the requested interval. We consciously use *relativedelta* for months to avoid the "30‑day" hack.
        if increment_by == "day":
            step = lambda d: d + timedelta(days=1)
        elif increment_by == "week":
            step = lambda d: d + timedelta(weeks=1)
        elif increment_by == "month":
            step = lambda d: d + timedelta(days=30)

        # 3. Main sweep: maintain an index into *full_problems* and enlarge the current window until it holds ≥ *min_bucket_size* elements.
        n, idx = len(full_problems), 0

        # The lower boundary is exclusive. Start one microsecond before the first problem so that the first problem definitely falls into the first bucket.
        prev_boundary = full_problems[0].end_time.date()  # earliest date present
        prev_boundary = prev_boundary - timedelta(microseconds=1)

        while idx < n:
            # Upper boundary grows by *step* until bucket is large enough.
            upper_boundary = step(prev_boundary)
            bucket = []

            while idx < n and len(bucket) < min_bucket_size:
                # Consume problems whose date ≤ current upper_boundary.
                while idx < n and full_problems[idx].end_time.date() <= upper_boundary:
                    bucket.append(full_problems[idx])
                    idx += 1

                # If still not enough, advance the boundary again and repeat.
                if len(bucket) < min_bucket_size:
                    upper_boundary = step(upper_boundary)

            # Even if we exit because idx == n (no more problems), we yield what we have.
            if bucket:
                yield upper_boundary.isoformat(), bucket
                # Next loop: window starts after *upper_boundary*.
                prev_boundary = upper_boundary

    def fill_problem_with_fair_odds(self, force: bool = False) -> None:
        """
        Certain challenge do not have odds data, we can fill in fair/uniform odds for each problem.
        If `force` is True, we will not check whether the problem already has odds data.
        """
        for problem in self.forecast_problems:
            if problem.has_odds and not force:
                continue
            problem.odds = [1 / len(problem.options)] * len(problem.options)

class ChallengeLoader(ABC):
    """
    Abstract base class for loading forecast challenges from different data sources.
    This separates the loading logic from the data model.
    """
    
    @abstractmethod
    def load_challenge(self) -> ForecastChallenge:
        """Load and return a ForecastChallenge from the data source."""
        pass

    @abstractmethod
    def get_challenge_metadata(self) -> Dict[str, Any]:
        """Get metadata about the challenge without loading all data."""
        pass
        