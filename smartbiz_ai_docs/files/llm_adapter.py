"""
Optional: let a language model polish the wording. The numbers stay safe because
Assistant._polish() throws away any answer that contains a number it did not get.

pip install anthropic     and set ANTHROPIC_API_KEY in your environment.
Check the current model names in Anthropic's documentation before you deploy.
"""
import anthropic

_client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY


def call_llm(prompt: str) -> str:
    resp = _client.messages.create(
        model='claude-sonnet-5-5',        # example name; confirm in the docs
        max_tokens=600,
        system=('You rewrite messages for small business owners in warm, simple English. '
                'Never change, add or remove a number. Never add advice or facts.'),
        messages=[{'role': 'user', 'content': prompt}],
    )
    return ''.join(b.text for b in resp.content if b.type == 'text')

# Usage:
#   from llm_adapter import call_llm
#   bot = Assistant(df, owner='Amina', call_llm=call_llm)
