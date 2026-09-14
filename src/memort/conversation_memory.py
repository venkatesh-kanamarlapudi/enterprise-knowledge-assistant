"""
Conversational memory.

Keeps a bounded window of recent user/assistant turns and uses it in two
ways:
  1. Query reformulation: rewrites a follow-up question (e.g. "What about
     carry-forward?") into a standalone question (e.g. "What is the
     carry-forward rule for earned leave?") using the LLM, so retrieval
     receives a self-contained query.
  2. Prompt context: the raw recent turns are also included in the final
     answer-generation prompt so the LLM keeps the broader conversational
     tone/context.
"""

from dataclasses import dataclass, field
from typing import List

from src.config import CONFIG

CONDENSE_PROMPT_TEMPLATE = """Given the conversation history and a follow-up question, \
rewrite the follow-up question to be a standalone question that contains all \
the context needed to answer it without seeing the history. If the follow-up \
question is already standalone, return it unchanged. Do not answer the \
question, only rewrite it.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""


@dataclass
class Turn:
    user: str
    assistant: str


@dataclass
class ConversationMemory:
    turns: List[Turn] = field(default_factory=list)

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        self.turns.append(Turn(user=user_message, assistant=assistant_message))
        max_turns = CONFIG.memory.max_turns
        if len(self.turns) > max_turns:
            self.turns = self.turns[-max_turns:]

    def clear(self) -> None:
        self.turns = []

    def as_history_text(self) -> str:
        if not self.turns:
            return "(no previous conversation)"
        lines = []
        for turn in self.turns:
            lines.append(f"User: {turn.user}")
            lines.append(f"Assistant: {turn.assistant}")
        return "\n".join(lines)

    def condense_question(self, question: str, llm) -> str:
        """
        Use the active LLM to rewrite a possibly context-dependent question
        into a standalone one, using recent conversation history.
        Falls back to the original question if there is no history yet,
        or if the LLM call fails for any reason.
        """
        if not self.turns:
            return question

        prompt = CONDENSE_PROMPT_TEMPLATE.format(
            history=self.as_history_text(), question=question
        )
        try:
            response = llm.invoke(prompt)
            standalone = getattr(response, "content", str(response)).strip()
            return standalone if standalone else question
        except Exception as exc:  # noqa: BLE001
            print(f"[memory] Query condensation failed, using raw question: {exc}")
            return question
