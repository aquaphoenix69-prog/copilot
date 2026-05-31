"""
State aggregator: combines interpretation + sentiment + recent transcript history
into the State object the planner consumes.

Holds an in-memory rolling history per conversation_id (last N chunks).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque

from app.core.schemas import Interpretation, Sentiment, State

_HISTORY: dict[str, Deque[str]] = defaultdict(lambda: deque(maxlen=8))
_SEQ: dict[str, int] = defaultdict(int)


def build_state(
    conversation_id: str,
    text: str,
    interp: Interpretation,
    sentiment: Sentiment,
) -> State:
    _HISTORY[conversation_id].append(text)
    _SEQ[conversation_id] += 1
    return State(
        conversation_id=conversation_id,
        seq=_SEQ[conversation_id],
        text=text,
        intent=interp.intent,
        customer_tag=interp.customer_tag,
        sentiment=sentiment,
        history=list(_HISTORY[conversation_id])[:-1],
    )


def reset(conversation_id: str) -> None:
    _HISTORY.pop(conversation_id, None)
    _SEQ.pop(conversation_id, None)
