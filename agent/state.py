from typing import TypedDict, Optional, List


class AgentState(TypedDict):
    property_input: dict          # raw user-supplied property details (may be incomplete/noisy)
    validated_input: Optional[dict]   # cleaned input with defaults filled in
    validation_errors: List[str]      # list of warnings about missing or invalid fields
    predicted_price: Optional[float]  # output from the ML model (₹)
    market_trends: Optional[str]      # retrieved RAG context
    report: Optional[str]             # final structured advisory report from LLM
