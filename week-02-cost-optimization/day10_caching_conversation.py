# conversation_manager.py
"""
A conversation manager that automatically compresses older turns into a summary
once the history grows past a threshold. This keeps recent turns raw for immediate
context fidelity and older turns compressed to save tokens and context window.
"""

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

SUMMARIZATION_SYSTEM = """You are compressing a conversation history into a concise summary 
that preserves all facts relevant to future turns: stated positions, preferences, numbers, 
decisions made, and open questions. Omit pleasantries and conversational filler entirely.

Write the summary as a set of factual statements, not prose narrative."""

def summarize_conversation(messages: list) -> str:
    """Compress a list of conversation turns into a short factual summary."""
    conversation_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}" for m in messages
    ])
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # cheap model — summarization is a bounded task
        max_tokens=300,
        temperature=0,
        messages=[
            {"role": "system", "content": SUMMARIZATION_SYSTEM},
            {"role": "user", "content": conversation_text}
        ]
    )
    return response.choices[0].message.content


class ManagedConversation:
    """
    A conversation manager that automatically summarizes older turns
    once the history grows past a threshold, keeping recent turns raw
    (for immediate context fidelity) and older turns compressed.
    """
    def __init__(self, system_prompt: str, keep_recent_turns: int = 6, 
                 summarize_after_turns: int = 12):
        self.system_prompt = system_prompt
        self.keep_recent_turns = keep_recent_turns
        self.summarize_after_turns = summarize_after_turns
        self.raw_messages = []
        self.summary = None
    
    def add_turn(self, role: str, content: str):
        self.raw_messages.append({"role": role, "content": content})
        
        if len(self.raw_messages) > self.summarize_after_turns:
            self._compress()
    
    def _compress(self):
        """Summarize everything except the most recent N turns."""
        turns_to_summarize = self.raw_messages[:-self.keep_recent_turns]
        recent_turns = self.raw_messages[-self.keep_recent_turns:]
        
        new_summary_content = summarize_conversation(turns_to_summarize)
        
        if self.summary:
            # Merge with any existing summary rather than losing it
            self.summary = self.summary + "\n" + new_summary_content
        else:
            self.summary = new_summary_content
        
        self.raw_messages = recent_turns
        print(f"  [Compressed] {len(turns_to_summarize)} turns → summary. "
              f"Keeping {len(recent_turns)} recent turns raw.")
    
    def get_messages_for_api(self) -> list:
        """Build the actual messages array to send, with summary injected as context."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if self.summary:
            # Inject the summary as a system-level fact, not a fake user turn
            summary_note = f"[Earlier conversation summary: {self.summary}]"
            messages.extend([
                {"role": "user", "content": summary_note},
                {"role": "assistant", "content": "Understood, I have that context."}
            ])
            
        messages.extend(self.raw_messages)
        return messages
    
    def chat(self, user_message: str) -> str:
        self.add_turn("user", user_message)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=500,
            messages=self.get_messages_for_api()
        )
        reply = response.choices[0].message.content
        self.add_turn("assistant", reply)
        return reply


# ── Test it with a long-ish conversation ──────────────────
if __name__ == "__main__":
    convo = ManagedConversation(
        system_prompt="You are a trading assistant for BTC and gold.",
        keep_recent_turns=4,
        summarize_after_turns=8
    )

    exchanges = [
        "I'm holding 0.5 BTC bought at $58,000.",
        "I also have a gold position: long from $2,340.",
        "My risk tolerance is 1% per trade.",
        "I prefer London session setups for gold.",
        "What's my unrealized P&L on BTC if price is now $62,000?",
        "Should I move my stop loss on the gold position?",
        "Remind me what my BTC entry price was.",   # tests whether old info survived compression
    ]

    for msg in exchanges:
        print(f"\nUSER: {msg}")
        reply = convo.chat(msg)
        print(f"ASSISTANT: {reply[:150]}...")
