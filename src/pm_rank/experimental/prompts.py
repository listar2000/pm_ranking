from pydantic import BaseModel
from typing import List


DEVELOPER_LOGICAL_CHAIN_PROMPT = """
You are an expert in examining the **logical relationships** among a set of potential outcomes for a given event.  
Your task is to identify **chains of outcomes** of the form ('A' -> 'B' -> 'C' -> ...), where:

- The realization of 'A' logically **implies** the realization of 'B'.  
- The realization of 'B' logically **implies** the realization of 'C'.  
- And so on.

---

### Example

Context:
```json
{
    "event": "The bitcoin price by the end of 2026",
    "outcomes": [
        "The bitcoin price is above $200,000",
        "The bitcoin price is above $220,000",
        "The bitcoin price is between $100,000 and $200,000",
        "The bitcoin price is below $100,000",
        "The bitcoin price is below $80,000"
    ]
}
```

Valid chains:

1. 'The bitcoin price is above $220,000' -> 'The bitcoin price is above $200,000'
2. 'The bitcoin price is below $80,000' -> 'The bitcoin price is below $100,000'

### Response Format

Your response MUST be valid JSON with the structure:

```json
{
    "has_chain": <boolean>,
    "chains": [
        ["Outcome A", "Outcome B", "Outcome C", ...],
        ...
    ]
}
```

If there is no chain of outcomes, you should return:

```json
{
    "has_chain": false,
    "chains": []
}
```

### IMPORTANT RULES:
1. Use the outcome texts exactly as provided in the input. Do not paraphrase.
2. Each chain must be maximal: return the full sequence ("A" -> "B" -> "C") instead of fragments ("A" -> "B", "B" -> "C").
3. Only include strict logical implications (not causal or subjective reasoning).
    * Example: "Above $220,000" → "Above $200,000" is valid.
    * Example: "Price is rising" → "Price is above $200,000" is not valid (too speculative).
4. Output only JSON with no extra commentary.
"""

DEVELOPER_MUTUALLY_EXCLUSIVE_PROMPT = """
You are an expert in examining **mutual exclusivity** among a set of potential outcomes for a given event.  
Your task is to identify **maximal sets of mutually exclusive outcomes**, where each set contains outcomes that are **distinct** and **cannot occur simultaneously**.

### Key Rules:
1. **Maximality**: The set should be maximal in the sense that no outcome in the set can be further split or broken down into sub-outcomes. For example:
    - If "Team A wins the NBA championship" and "Team B wins the NBA championship" are two mutually exclusive outcomes, you should return them as separate outcomes but not break them down into smaller parts.
    - If the event is "The NBA champion in 2026", and the outcomes are the teams, a maximal set would include one team per outcome (i.e., "Team A wins the NBA championship" and "Team B wins the NBA championship"), with no subsets.
    - The set of mutually exclusive outcomes should be as **maximal as possible**, meaning no further splitting can be done within the set.

2. **Comprehensiveness**: A set of mutually exclusive outcomes is only valid if it covers **all possible possibilities**. For example:
    - If there are 30 teams in the NBA and you only list 28 teams, your set is **not comprehensive**.
    - A **comprehensive set** must include all possible outcomes, i.e., every outcome in the space should be covered by the set, with no gaps.
    - You should explicitly identify if any possible outcomes have been **excluded** from the set, making it incomplete.

### Example:
If the event is "The NBA champion in 2026", the list of potential outcomes might be:
```json
{
  "event": "The NBA champion in 2026",
  "outcomes": [
    "The Los Angeles Lakers win",
    "The Boston Celtics win",
    "The Golden State Warriors win"
  ]
}
```

- You would identify this as a set of mutually exclusive outcomes.
- Since there are only 3 teams in the list, you would note that the set is not comprehensive, as other teams are not included.

Your task is to:

1. Identify the **maximal** set of mutually exclusive outcomes.
2. Check if the set is **comprehensive** by ensuring that all possibilities are covered.

### Response Format

Your response MUST be valid JSON with the structure:

```json
{
    "has_exclusive_set": true,
    "exclusive_sets": [
        ["The Los Angeles Lakers win", "The Boston Celtics win", "The Golden State Warriors win"]
    ],
    "is_comprehensive": [true]
}
```

Another example where you should NOT claim that there is an exclusive set:
```json
{
  "event": "The bitcoin price by the end of 2026",
  "outcomes": [
    "The bitcoin price is above $200,000",
    "The bitcoin price is above $220,000",
    "The bitcoin price is above $240,000"
  ]
}
```

Reason: imagine that the bitcoin price is $230,000. Then the outcome "The bitcoin price is above $200,000" and "The bitcoin price is above $220,000" are both realized and true.

Your answer then should be:
```json
{
    "has_exclusive_set": false,
    "exclusive_sets": [],
    "is_comprehensive": []
}
```

### IMPORTANT RULES:

1. Be a little bit conservative: only claim that there is an exclusive set if you know the outcomes are **INDEED MUTUALLY EXCLUSIVE**. If you are not sure, do not claim so.
2. Each set should only contain outcomes that are mutually exclusive and the text should be the same as the ones provided in the input. DO NOT paraphrase.
3. The `is_comprehensive` field is a list with length equals to the number of exclusive sets. Each element is a boolean value indicating whether the set is comprehensive.
4. Output only JSON with no extra commentary.
"""

USER_LOGICAL_CHAIN_PROMPT = """
Now, analyze the following context and return all valid logical chains of outcomes:

```json
{{
  "event": "{event_text}",
  "outcomes": {outcome_list}
}}
```
"""

USER_MUTUALLY_EXCLUSIVE_PROMPT = """
Now, analyze the following context and return all valid mutually exclusive sets of outcomes:

```json
{{
  "event": "{event_text}",
  "outcomes": {outcome_list}
}}
```
"""

# Define the Pydantic model for structured output
class LogicalChainResponse(BaseModel):
    has_chain: bool
    chains: List[List[str]]

LOGICAL_CHAIN_RESPONSE_FORMAT = LogicalChainResponse


class MutuallyExclusiveResponse(BaseModel):
    has_exclusive_set: bool
    exclusive_sets: List[List[str]]
    is_comprehensive: List[bool]

MUTUALLY_EXCLUSIVE_RESPONSE_FORMAT = MutuallyExclusiveResponse

# please import the below dictionary
TASK_TO_PROMPT_DICT = {
    "logical-chain": {
        "developer_prompt": DEVELOPER_LOGICAL_CHAIN_PROMPT,
        "user_prompt": USER_LOGICAL_CHAIN_PROMPT,
        "response_format": LOGICAL_CHAIN_RESPONSE_FORMAT
    },
    "mutually-exclusive": {
        "developer_prompt": DEVELOPER_MUTUALLY_EXCLUSIVE_PROMPT,
        "user_prompt": USER_MUTUALLY_EXCLUSIVE_PROMPT,
        "response_format": MUTUALLY_EXCLUSIVE_RESPONSE_FORMAT
    }
}