"""
Concrete implementations of ChallengeLoader for different data sources.
"""
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
from .base import ChallengeLoader, ForecastChallenge, ForecastProblem, ForecastEvent
from datetime import datetime
import math
from .utils import parse_json_or_eval


class GJOChallengeLoader(ChallengeLoader):
    """Load forecast challenges from GJO (Good Judgment Open) data format."""
    
    def __init__(self, predictions_df: Optional[pd.DataFrame] = None, predictions_file: Optional[str] = None, \
        metadata_file: Optional[str] = None, challenge_title: str = ""):
        """
        Initialize the GJOChallengeLoader. The challenge can be either loaded with a given `pd.DataFrame` or with \
            a combination of paths `predictions_file` and `metadata_file`.

        Args:
            predictions_df (pd.DataFrame): a pd.DataFrame containing the predictions. If provided, \
                `predictions_file` and `metadata_file` will be ignored.
            predictions_file (str): the path to the predictions file
            metadata_file (str): the path to the metadata file
            challenge_title (str): the title of the challenge
        """
        self.challenge_title = challenge_title
        # either predictions_file or prediction_df should be provided
        if predictions_df is None:
            assert predictions_file is not None and metadata_file is not None, \
                "Either predictions_df or (predictions_file and metadata_file) should be provided"

            self.predictions_file = Path(predictions_file)
            self.metadata_file = Path(metadata_file)

            if not self.predictions_file.exists():
                raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
            if not self.metadata_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        else:
            self.predictions_df = predictions_df
            
    def _get_filtered_df(self, predictions_df: pd.DataFrame, metadata_df: pd.DataFrame, forecaster_filter: int, problem_filter: int) \
        ->  Tuple[pd.DataFrame, pd.DataFrame]:
        # step 1: we group the problems by problem_id and calculate the number of events for each problem
        problem_event_counts = predictions_df.groupby('problem_id').size()

        # step 2: we filter the problems by the number of events
        filtered_metadata_df = metadata_df[metadata_df['problem_id'].isin(problem_event_counts[problem_event_counts >= problem_filter].index)]
        filtered_predictions_df = predictions_df[predictions_df['problem_id'].isin(filtered_metadata_df['problem_id'])]

        # step 3: we filter the forecasters by the number of events
        forecaster_event_counts = filtered_predictions_df.groupby('username').size()
        filtered_predictions_df = filtered_predictions_df[filtered_predictions_df['username'].isin(forecaster_event_counts[forecaster_event_counts >= forecaster_filter].index)]

        return filtered_predictions_df, filtered_metadata_df
    
    def load_challenge(self, forecaster_filter: int = 0, problem_filter: int = 0) -> ForecastChallenge:
        """Load challenge data from GJO format files.

        Args:
            forecaster_filter: minimum number of events for a forecaster to be included
            problem_filter: minimum number of events for a problem to be included

        Returns:
            ForecastChallenge: a ForecastChallenge object containing the forecast problems and events
        """
        if hasattr(self, 'predictions_df'):
            predictions_df = self.predictions_df
        else:
            predictions_df = pd.read_json(self.predictions_file)
            metadata_df = pd.read_json(self.metadata_file)
        
        # Filter the data
        if forecaster_filter > 0 or problem_filter > 0:
            filtered_predictions_df, filtered_metadata_df = self._get_filtered_df(predictions_df, metadata_df, forecaster_filter, problem_filter)
        else:
            filtered_predictions_df, filtered_metadata_df = predictions_df, metadata_df
        
        # Iterate over each row of the filtered prediction df to construct the forecast events for each problem
        problem_id_to_forecast_events = {}
        problem_id_to_correct_idx = {}

        for _, row in filtered_predictions_df.iterrows():
            problem_id: int = int(row['problem_id'])
            username: str = str(row['username'])
            # the original timestamp is in string format like "2024-09-10T19:22:23Z"
            timestamp: datetime = datetime.fromisoformat(str(row['timestamp']))
            probs: List[float] = list(row['prediction'])

            if problem_id not in problem_id_to_forecast_events:
                problem_id_to_forecast_events[problem_id] = []
                problem_meta_row = filtered_metadata_df[filtered_metadata_df['problem_id'] == problem_id].iloc[0]
                problem_id_to_correct_idx[problem_id] = problem_meta_row['options'].index(problem_meta_row['correct_answer'])
            
            forecast_event = ForecastEvent(
                problem_id=problem_id,
                username=username,
                timestamp=timestamp,
                probs=probs,
                correct_prob=probs[problem_id_to_correct_idx[problem_id]]
            )

            problem_id_to_forecast_events[problem_id].append(forecast_event)

        # Iterate over each row of the filtered metadata df to construct the forecast problems
        forecast_problems = []
        for _, row in filtered_metadata_df.iterrows():
            problem_id: int = int(row['problem_id'])
            problem_forecasts = problem_id_to_forecast_events[problem_id]
            forecast_problems.append(ForecastProblem(
                title=str(row['title']),
                problem_id=problem_id,
                options=list(row['options']),
                correct_option=str(row['correct_answer']),
                forecasts=problem_forecasts,
                end_date=datetime.fromisoformat(str(row['metadata']['end_date'])),
                num_forecasters=len(problem_forecasts),
                url=str(row['url']),
                odds=None
            ))
        
        # Create the forecast challenge
        forecast_challenge = ForecastChallenge(
            title=self.challenge_title,
            forecast_problems=forecast_problems
        )

        return forecast_challenge
    
    def get_challenge_metadata(self) -> Dict[str, Any]:
        """Get basic metadata about the GJO challenge."""
        if self.metadata_file is not None:
            with open(self.metadata_file, 'r') as f:
                metadata_df = pd.read_json(f)
        else:
            metadata_df = self.predictions_df[['problem_id', 'title', 'options', 'correct_answer', 'url']].drop_duplicates()
        
        if self.challenge_title is None:
            # set title to be the `xx` part of metadata file before `xx_metadata.json`
            self.challenge_title = self.metadata_file.stem.split('_')[0]
        
        return {
            'title': self.challenge_title,
            'num_problems': len(metadata_df),
            'predictions_file': str(self.predictions_file),
            'metadata_file': str(self.metadata_file)
        }


class ProphetArenaChallengeLoader(ChallengeLoader):
    """Load forecast challenges from Prophet Arena data format."""
    
    def __init__(self, predictions_df: Optional[pd.DataFrame] = None, predictions_file: Optional[str] = None, \
        challenge_title: str = "", use_bid_for_odds: bool = False):
        """
        Initialize the ProphetArenaChallengeLoader. The challenge can be either loaded with a given `pd.DataFrame` or with a path to a predictions file.

        Args:
            predictions_df (pd.DataFrame): a pd.DataFrame containing the predictions. If provided, `predictions_file` will be ignored.
            predictions_file (str): the path to the predictions file
            challenge_title (str): the title of the challenge
            use_bid_for_odds (bool): whether to use the `yes_bid` field for implied probability calculation
                if True, the implied probability will be calculated as the (yes_bid + no_bid) / 2
                if False, the implied probability will be simply `yes_ask` (normalized to sum to 1)
        """
        self.challenge_title = challenge_title
        self.use_bid_for_odds = use_bid_for_odds
        if predictions_df is None:
            assert predictions_file is not None, "Either predictions_df or predictions_file should be provided"
            self.predictions_file = Path(predictions_file)
            if not self.predictions_file.exists():
                raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
        else:
            self.predictions_df = predictions_df
    
    @staticmethod
    def _calculate_implied_probs_for_problem(market_info: dict, options: list, use_bid_for_odds: bool = False) -> list | None:
        """
        Calculate odds for each option from market_info dict.
        For multi-option, use yes_ask for each option and normalize to sum to 1 (implied probabilities).
        """
        asks = []
        for opt in options:
            info = market_info.get(opt, {})
            yes_ask = info.get('yes_ask', None)
            if yes_ask is not None and yes_ask > 0:
                if use_bid_for_odds:
                    if 'yes_bid' in info:
                        yes_bid = info['yes_bid']
                        asks.append((yes_bid + yes_ask) / 2)
                    else:
                        asks.append(yes_ask)
                else:
                    asks.append(yes_ask)
            else:
                print(f"Warning: {opt} has no odds info")
                asks.append(None)
        # normalize the implied probabilities to sum to 1
        implied_probs = [(a / 100.0) if a is not None else 0.0 for a in asks]
        total = sum(implied_probs)

        if total > 0:
            return [p / total for p in implied_probs]
        else:
            return None

    def load_challenge(self) -> ForecastChallenge:
        """
        Load challenge data from Prophet Arena data format.
        Group by submission_id, then for each group, build the list of forecasts, then the ForecastProblem.
        """
        if hasattr(self, 'predictions_df'):
            df = self.predictions_df
        else:
            df = pd.read_csv(self.predictions_file)

        forecast_problems = []
        problem_id_counter = 1
        grouped = df.groupby('submission_id')
        for submission_id, group in grouped:
            first_row = group.iloc[0]
            options = parse_json_or_eval(first_row['markets'], expect_type=list)
            market_info = parse_json_or_eval(first_row['market_info'], expect_type=dict)
            first_option_info = next(iter(market_info.values())) if market_info else {}
            title = first_option_info.get('title', submission_id)
            end_date_str = first_option_info.get('close_time', None)
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00')) if end_date_str else datetime.now()
            odds = self._calculate_implied_probs_for_problem(market_info, options, self.use_bid_for_odds)
            market_outcome = parse_json_or_eval(first_row['market_outcome'], expect_type=dict)
            
            correct_option = [opt for opt, val in (market_outcome or {}).items() if int(val) == 1][0]

            if correct_option not in options:
                continue

            forecasts = []
            for _, row in group.iterrows():
                username = str(row['predictor_name'])
                prediction: dict = parse_json_or_eval(row['prediction'], expect_type=dict)
                probs_dict = {d['market']: d['probability'] for d in prediction.get('probabilities', [])}
                probs = [probs_dict.get(opt, 0.0) for opt in options]
                # make sure the probs sum to 1
                if not math.isclose(sum(probs), 1.0, abs_tol=1e-6):
                    continue
                timestamp = datetime.now()
                correct_prob = probs[options.index(correct_option)] if correct_option in options else 0.0
                forecasts.append(ForecastEvent(
                    problem_id=problem_id_counter,
                    username=username,
                    timestamp=timestamp,
                    probs=probs,
                    correct_prob=correct_prob
                ))
            
            if len(forecasts) > 0:
                forecast_problems.append(ForecastProblem(
                    title=title,
                    problem_id=problem_id_counter,
                    options=options,
                    correct_option=correct_option,
                    forecasts=forecasts,
                    end_date=end_date if end_date else datetime.now(),
                    num_forecasters=len(forecasts),
                    url=None,
                    odds=odds
                ))
                problem_id_counter += 1

        forecast_challenge = ForecastChallenge(
            title=self.challenge_title or "Prophet Arena Challenge",
            forecast_problems=forecast_problems
        )
        return forecast_challenge

    def get_challenge_metadata(self) -> Dict[str, Any]:
        """
        Get basic metadata about the Prophet Arena challenge using pandas groupby (no full parsing).
        """
        if hasattr(self, 'predictions_df'):
            df = self.predictions_df
        else:
            df = pd.read_csv(self.predictions_file)
        num_problems = df['submission_id'].nunique()
        return {
            'title': self.challenge_title or "Prophet Arena Challenge",
            'num_problems': num_problems,
            'num_forecasters': df['predictor_name'].nunique(),
            'predictions_file': str(getattr(self, 'predictions_file', 'Loaded from pd.DataFrame'))
        }