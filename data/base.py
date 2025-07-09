"""
Defining the prediction market data structure and functions to load them 
from different types of data sources.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Iterator, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from functools import cached_property
import math, random


class ForecastEvent(BaseModel):
    """Individual forecast from a user for a specific problem."""
    problem_id: int = Field(description="The id of the problem")
    username: str = Field(description="The user name/id of the forecaster")
    timestamp: datetime = Field(description="The timestamp of the forecast")
    probs: List[float] = Field(description="The forecasted probabilities for each option")
    correct_prob: float = Field(description="The probability assigned to the correct answer")

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

    @field_validator('correct_prob')
    def validate_correct_probability(cls, v, info):
        """Validate that correct_prob matches one of the probabilities."""
        if info.data and 'probs' in info.data:
            probs = info.data['probs']
            if not any(math.isclose(v, p, abs_tol=1e-6) for p in probs):
                raise ValueError(f"correct_prob {v} must match one of the probabilities {probs}")
        return v


class ForecastProblem(BaseModel):
    """A prediction problem with multiple options and forecasts."""
    title: str = Field(description="The title of the problem")
    problem_id: int = Field(description="The id of the problem")
    options: List[str] = Field(description="The available options for the problem")
    correct_option: str = Field(description="The correct answer")
    forecasts: List[ForecastEvent] = Field(description="All forecasts for this problem")
    end_date: datetime = Field(description="The end date of the problem")
    num_forecasters: int = Field(description="The number of forecasters")
    url: str | None = Field(None, description="The URL of the problem")
    odds: List[float] | None = Field(None, description="The odds for each option")

    @field_validator('correct_option')
    def validate_correct_option(cls, v, info):
        """Validate that correct_option is in the options list."""
        if info.data and 'options' in info.data and v not in info.data['options']:
            raise ValueError(f"correct_option '{v}' must be one of the options: {info.data['options']}")
        return v

    @field_validator('forecasts')
    def validate_forecasts(cls, v, info):
        """Validate that all forecasts have correct number of probabilities."""
        if info.data and 'options' in info.data:
            expected_length = len(info.data['options'])
            for forecast in v:
                if len(forecast.probs) != expected_length:
                    raise ValueError(
                        f"Forecast by {forecast.username} has {len(forecast.probs)} probabilities, "
                        f"expected {expected_length}"
                    )
        return v

    @field_validator('odds')
    def validate_odds(cls, v, info):
        """Validate that odds match the number of options if provided."""
        if v is not None and info.data and 'options' in info.data:
            if len(v) != len(info.data['options']):
                raise ValueError(f"Number of odds ({len(v)}) must match number of options ({len(info.data['options'])})")
        return v

    @property
    def has_odds(self) -> bool:
        """Check if the problem has odds data."""
        return self.odds is not None and len(self.odds) > 0
    
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


class ForecastChallenge(BaseModel):
    """
    A collection of forecast problems with validation and computed properties.
    """
    title: str = Field(description="The title of the challenge")
    forecast_problems: List[ForecastProblem] = Field(description="The list of forecast problems")

    @field_validator('forecast_problems')
    def validate_problems(cls, v):
        """Validate that there are problems and they have unique IDs."""
        if not v:
            raise ValueError("Challenge must have at least one problem")
        
        problem_ids = [p.problem_id for p in v]
        if len(problem_ids) != len(set(problem_ids)):
            raise ValueError("All problems must have unique IDs")
        
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

    def get_forecaster_problems(self, username: str) -> List[ForecastProblem]:
        """Get all problems that a specific forecaster participated in."""
        forecaster_problem_ids = {
            forecast.problem_id for forecast in self.forecaster_map.get(username, [])
        }
        return [p for p in self.forecast_problems if p.problem_id in forecaster_problem_ids]

    def get_problem_by_id(self, problem_id: int) -> ForecastProblem | None:
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
            full_problems.sort(key=lambda x: x.end_date.replace(tzinfo=None))

        for i in range(0, len(full_problems), increment):
            yield full_problems[i:i+increment]


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
        