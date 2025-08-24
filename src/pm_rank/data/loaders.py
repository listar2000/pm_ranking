"""
Concrete implementations of ChallengeLoader for different data sources.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
from .base import ChallengeLoader, ForecastChallenge, ForecastProblem, ForecastEvent, ProphetArenaForecastEvent
from datetime import datetime
import math
from .utils import parse_json_or_eval, get_logger


class GJOChallengeLoader(ChallengeLoader):
    """Load forecast challenges from GJO (Good Judgment Open) data format."""

    def __init__(self, predictions_df: Optional[pd.DataFrame] = None, predictions_file: Optional[str] = None,
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
        self.logger = get_logger(
            f"pm_rank.data.loaders.{self.__class__.__name__}")

        # either predictions_file or prediction_df should be provided
        if predictions_df is None:
            assert predictions_file is not None and metadata_file is not None, \
                "Either predictions_df or (predictions_file and metadata_file) should be provided"

            self.predictions_file = Path(predictions_file)
            self.metadata_file = Path(metadata_file)

            if not self.predictions_file.exists():
                raise FileNotFoundError(
                    f"Predictions file not found: {predictions_file}")
            if not self.metadata_file.exists():
                raise FileNotFoundError(
                    f"Metadata file not found: {metadata_file}")

            self.logger.info(
                f"Initialize challenge loader with predictions file {predictions_file} and metadata file {metadata_file}")
        else:
            self.predictions_df = predictions_df
            self.logger.info(f"Initialize challenge loader with pd.DataFrame")

    def _get_filtered_df(self, predictions_df: pd.DataFrame, metadata_df: pd.DataFrame, forecaster_filter: int, problem_filter: int) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        # step 1: we group the problems by problem_id and calculate the number of events for each problem
        problem_event_counts = predictions_df.groupby('problem_id').size()

        # step 2: we filter the problems by the number of events
        filtered_metadata_df = metadata_df[metadata_df['problem_id'].isin(
            problem_event_counts[problem_event_counts >= problem_filter].index)]
        filtered_predictions_df = predictions_df[predictions_df['problem_id'].isin(
            filtered_metadata_df['problem_id'])]

        # step 3: we filter the forecasters by the number of events
        forecaster_event_counts = filtered_predictions_df.groupby(
            'username').size()
        filtered_predictions_df = filtered_predictions_df[filtered_predictions_df['username'].isin(
            forecaster_event_counts[forecaster_event_counts >= forecaster_filter].index)]

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
            filtered_predictions_df, filtered_metadata_df = self._get_filtered_df(
                predictions_df, metadata_df, forecaster_filter, problem_filter)
        else:
            filtered_predictions_df, filtered_metadata_df = predictions_df, metadata_df

        # Iterate over each row of the filtered prediction df to construct the forecast events for each problem
        problem_id_to_forecast_events = {}
        problem_id_to_correct_idx = {}

        for _, row in filtered_predictions_df.iterrows():
            problem_id: str = str(row['problem_id'])
            username: str = str(row['username'])
            # the original timestamp is in string format like "2024-09-10T19:22:23Z"
            timestamp: datetime = datetime.fromisoformat(str(row['timestamp']))
            probs: List[float] = list(row['prediction'])

            if problem_id not in problem_id_to_forecast_events:
                problem_id_to_forecast_events[problem_id] = []
                problem_meta_row = filtered_metadata_df[filtered_metadata_df['problem_id']
                                                        == problem_id].iloc[0]
                problem_id_to_correct_idx[problem_id] = problem_meta_row['options'].index(
                    problem_meta_row['correct_answer'])

            forecast_event = ForecastEvent(
                forecast_id=f"{problem_id}-{username}",
                problem_id=problem_id,
                username=username,
                timestamp=timestamp,
                probs=probs,
                unnormalized_probs=probs,
            )

            problem_id_to_forecast_events[problem_id].append(forecast_event)

        # Iterate over each row of the filtered metadata df to construct the forecast problems
        forecast_problems = []
        for _, row in filtered_metadata_df.iterrows():
            problem_id: str = str(row['problem_id'])
            problem_forecasts = problem_id_to_forecast_events[problem_id]
            forecast_problems.append(ForecastProblem(
                title=str(row['title']),
                problem_id=problem_id,
                options=list(row['options']),
                correct_option_idx=problem_id_to_correct_idx[problem_id],
                forecasts=problem_forecasts,
                end_time=datetime.fromisoformat(
                    str(row['metadata']['end_date'])),
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
            metadata_df = self.predictions_df[[
                'problem_id', 'title', 'options', 'correct_answer', 'url']].drop_duplicates()

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

    def __init__(self, predictions_df: Optional[pd.DataFrame] = None, predictions_file: Optional[str] = None,
                 challenge_title: str = "", use_bid_for_odds: bool = False, use_open_time: bool = False):
        """Initialize the ProphetArenaChallengeLoader.

        The challenge can be either loaded with a given `pd.DataFrame` or with a path to a predictions file.

        :param predictions_df: A pd.DataFrame containing the predictions. If provided, `predictions_file` will be ignored.
        :param predictions_file: The path to the predictions file.
        :param challenge_title: The title of the challenge.
        :param use_bid_for_odds: Whether to use the `yes_bid` field for implied probability calculation.
                                If True, the implied probability will be calculated as the (yes_bid + no_bid) / 2.
                                If False, the implied probability will be simply `yes_ask` (normalized to sum to 1).
        :param use_open_time: Whether to use the `open_time` field for the `end_time` of the problem.
                                If True, the `end_time` will be the `open_time` of the problem.
                                If False, the `end_time` will be the `close_time` of the problem.
        """
        self.challenge_title = challenge_title
        self.use_bid_for_odds = use_bid_for_odds
        self.use_open_time = use_open_time
        self.logger = get_logger(
            f"pm_rank.data.loaders.{self.__class__.__name__}")
        if predictions_df is None:
            assert predictions_file is not None, "Either predictions_df or predictions_file should be provided"
            self.predictions_file = Path(predictions_file)
            if not self.predictions_file.exists():
                raise FileNotFoundError(
                    f"Predictions file not found: {predictions_file}")
            self.logger.info(
                f"Initialize challenge loader with predictions file {predictions_file}")
        else:
            self.predictions_df = predictions_df
            self.logger.info(f"Initialize challenge loader with pd.DataFrame")

    @staticmethod
    def _calculate_implied_probs_for_problem(market_info: dict, options: list, use_bid_for_odds: bool = False, \
        yes_contract: bool = True, logger: Optional[logging.Logger] = None) -> List:
        """
        Calculate odds for each option from market_info dict.
        For multi-option, use yes_ask for each option and normalize to sum to 1 (implied probabilities).
        """
        ask_str = 'yes_ask' if yes_contract else 'no_ask'
        bid_str = 'yes_bid' if yes_contract else 'no_bid'

        asks = []
        for opt in options:
            info = market_info.get(opt, {})
            yes_ask = info.get(ask_str, None)
            if info.get('liquidity', None) is not None and info.get('liquidity', None) < 100:
                asks.append(100)
            elif yes_ask is not None and yes_ask > 0:
                if use_bid_for_odds:
                    if bid_str in info:
                        yes_bid = info[bid_str]
                        asks.append((yes_bid + yes_ask) / 2)
                        # asks.append(max(yes_bid, yes_ask))
                    else:
                        asks.append(yes_ask)
                else:
                    asks.append(yes_ask)
            else:
                warning_msg = f"Warning: Option {opt} in market {market_info.get('title', 'Unknown Market')} does not have odds info"
                logger.warning(warning_msg) if logger is not None else print(
                    warning_msg)
                asks.append(None)

        implied_probs = [(a / 100.0) if a is not None else 1e-3 for a in asks]
        return implied_probs

    @staticmethod
    def _get_normalized_probs(unnormalized_probs: list) -> list:
        """
        Get normalized probabilities from unnormalized probabilities.
        """
        if not math.isclose(sum(unnormalized_probs), 1.0, abs_tol=1e-6):
            if sum(unnormalized_probs) > 0:
                return [p / sum(unnormalized_probs) for p in unnormalized_probs]
            else:
                return [1.0 / len(unnormalized_probs) for _ in unnormalized_probs]
        return unnormalized_probs

    def load_challenge(self, add_market_baseline: bool = False) -> ForecastChallenge:
        """
        Load challenge data from Prophet Arena data format.
        Group by submission_id, then for each group, build the list of forecasts, then the ForecastProblem.

        :param add_market_baseline: Whether to add the market baseline as a forecaster
        :return: A ForecastChallenge object containing the forecast problems and events.
        """
        if hasattr(self, 'predictions_df'):
            df = self.predictions_df
        else:
            df = pd.read_csv(self.predictions_file)

        forecast_problems = []
        categories = []
        grouped = df.groupby('event_ticker')

        if self.use_open_time:
            self.logger.warning(
                f"Currently, the Prophet Arena challenge is using the `open_time` in place of `close_time` for each problem.")

        for event_ticker, group in grouped:
            first_row = group.iloc[0]
            problem_id = str(event_ticker)
            market_info = parse_json_or_eval(
                first_row['market_info'], expect_type=dict)
            # skip this market if the market_info is empty
            if not market_info:
                continue
            first_option_info = next(iter(market_info.values()))
            title = first_option_info.get('title', event_ticker)

            if self.use_open_time:
                open_time = first_option_info.get('open_time', None)
                end_time = datetime.fromisoformat(open_time.replace(
                    'Z', '+00:00')) if open_time else datetime.now()
            else:
                if 'close_time' in first_row:
                    close_time = first_row['close_time']
                else:
                    close_time = first_option_info.get('close_time', None)
                end_time = datetime.fromisoformat(close_time.replace(
                    'Z', '+00:00')) if close_time else datetime.now()
            
            problem_option_keys = list(market_info.keys())

            odds = self._calculate_implied_probs_for_problem(
                market_info, problem_option_keys, self.use_bid_for_odds, True, self.logger)

            no_odds = self._calculate_implied_probs_for_problem(
                market_info, problem_option_keys, self.use_bid_for_odds, False, self.logger)

            market_outcome = parse_json_or_eval(
                first_row['market_outcome'], expect_type=dict)

            correct_option_idx = [i for i, key in enumerate(problem_option_keys) if market_outcome.get(key, 0) == 1]

            timestamp = datetime.now()

            forecasts = []

            if add_market_baseline:
                # we set the submission_id to be the submission_id of the first row in the group
                submission_id = str(first_row['submission_id'])
                # we will add a "market row" in this group, with (unnormalized) prob being simply the market odds
                forecasts.append(ProphetArenaForecastEvent(
                    forecast_id=f"{problem_id}-market-baseline",
                    problem_id=problem_id,
                    submission_id=submission_id,
                    username="market-baseline",
                    timestamp=timestamp,
                    probs=self._get_normalized_probs(odds),
                    unnormalized_probs=odds,
                ))

            category = first_row.get('category', None)
            if category is not None and category not in categories:
                categories.append(category)

            for i, row in group.iterrows():
                username = str(row['predictor_name'])
                prediction: dict = parse_json_or_eval(
                    row['prediction'], expect_type=dict)
                probs_dict = {d['market']: d['probability']
                              for d in prediction.get('probabilities', [])}

                # clip raw prob to be between 0 and 1
                unnormalized_probs = [max(0.0, min(1.0, probs_dict.get(opt, 0.0))) for opt in problem_option_keys]
                # make sure the probs sum to 1
                probs = self._get_normalized_probs(unnormalized_probs)

                # set `forecast_id` to be `prediction_id` if the column exists, otherwise construct one
                # using `username` and `problem_id`
                forecast_id = str(row['prediction_id']) if 'prediction_id' in row else f"{username}-{problem_id}-{i}"
                submission_id = str(row['submission_id']) if 'submission_id' in row else problem_id

                forecasts.append(ProphetArenaForecastEvent(
                    forecast_id=forecast_id,
                    problem_id=problem_id,
                    submission_id=submission_id,
                    username=username,
                    timestamp=timestamp,
                    probs=probs,
                    unnormalized_probs=unnormalized_probs,
                ))

            if len(forecasts) > 0:
                forecast_problems.append(ForecastProblem(
                    title=title,
                    problem_id=problem_id,
                    options=problem_option_keys,
                    correct_option_idx=correct_option_idx,
                    forecasts=forecasts,
                    end_time=end_time,
                    num_forecasters=len(forecasts),
                    url=None,
                    odds=odds,
                    no_odds=no_odds,
                    category=category
                ))

        forecast_challenge = ForecastChallenge(
            title=self.challenge_title or "Prophet Arena Challenge",
            forecast_problems=forecast_problems,
            categories=categories
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
