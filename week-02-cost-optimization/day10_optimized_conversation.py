# optimized_conversation.py
"""
Combines all three context optimization techniques:
1. Cached system prompt (OpenAI caches long prefixes automatically)
2. Structured state for facts (overwritten, not appended)
3. Summarized + pruned conversational history (compressed as it grows)
"""

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

class OptimizedConversation:
    """
    Manages structured state, recent turns, and compressed older turns.
    Leverages OpenAI's automatic prompt caching for static context prefixes.
    """
    def __init__(self, system_prompt: str, keep_recent_turns: int = 4):
        self.system_prompt = system_prompt
        self.keep_recent_turns = keep_recent_turns
        self.state = {}                # structured facts
        self.raw_messages = []         # recent conversational turns
        self.summary = None            # compressed older turns
    
    def update_fact(self, key: str, value: str):
        self.state[key] = value
    
    def _build_state_context(self) -> str:
        if not self.state:
            return ""
        lines = [f"- {k}: {v}" for k, v in self.state.items()]
        return "Known facts:\n" + "\n".join(lines)
    
    def _maybe_compress(self):
        if len(self.raw_messages) > self.keep_recent_turns * 2:
            to_summarize = self.raw_messages[:-self.keep_recent_turns]
            recent = self.raw_messages[-self.keep_recent_turns:]
            
            conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in to_summarize])
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=200,
                temperature=0,
                messages=[
                    {"role": "system", "content": "Summarize this conversation into brief factual statements, omitting filler."},
                    {"role": "user", "content": conversation_text}
                ]
            )
            new_summary = response.choices[0].message.content
            self.summary = (self.summary + "\n" + new_summary) if self.summary else new_summary
            self.raw_messages = recent
    
    def chat(self, user_message: str) -> str:
        self.raw_messages.append({"role": "user", "content": user_message})
        self._maybe_compress()
        
        context_parts = []
        if self.state:
            context_parts.append(self._build_state_context())
        if self.summary:
            context_parts.append(f"Earlier conversation summary:\n{self.summary}")
        
        context_prefix = "\n\n".join(context_parts)
        
        # Build API messages: static system prompt + injected state/summary + raw turns
        full_messages = [{"role": "system", "content": self.system_prompt}]
        
        if context_prefix:
            full_messages.extend([
                {"role": "user", "content": context_prefix},
                {"role": "assistant", "content": "Understood, using this as context."}
            ])
            
        full_messages.extend(self.raw_messages)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=500,
            messages=full_messages
        )
        
        reply = response.choices[0].message.content
        self.raw_messages.append({"role": "assistant", "content": reply})
        
        # Log actual cache behavior for cost tracking (OpenAI includes this in usage)
        usage = response.usage
        cached = getattr(usage.prompt_tokens_details, 'cached_tokens', 0) if hasattr(usage, 'prompt_tokens_details') else 0
        fresh_input = usage.prompt_tokens - cached
        
        print(f"  [tokens] cache_hit={cached}, "
              f"fresh_input={fresh_input}, output={usage.completion_tokens}")
        
        return reply


# ── Usage ──────────────────────────────────────────────────
if __name__ == "__main__":
    convo = OptimizedConversation(
        system_prompt="You are a trading assistant for BTC and XAUUSD using ICT/SMC methodology. Be concise."
    )

    convo.update_fact("btc_position", "0.5 BTC @ $58,000")
    convo.update_fact("risk_per_trade", "1%")

    print("\nUSER: What's my risk exposure right now?")
    print(f"ASSISTANT: {convo.chat('What\'s my risk exposure right now?')}")
    
    print("\nUSER: If BTC drops to $55,000, what's my unrealized loss?")
    print(f"ASSISTANT: {convo.chat('If BTC drops to $55,000, what\'s my unrealized loss?')}")
