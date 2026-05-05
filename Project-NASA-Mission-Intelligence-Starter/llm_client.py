import os
from typing import Dict, List, Optional

from openai import OpenAI

# Load variables from a local .env file if one is present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def generate_response(openai_key: Optional[str], user_message: str, context: str,
                     conversation_history: List[Dict], model: Optional[str] = None) -> str:
    """Generate response using OpenAI with context.

    If `openai_key` is not provided, falls back to the OPENAI_API_KEY env var.
    If `model` is not provided, falls back to OPENAI_CHAT_MODEL or 'gpt-3.5-turbo'.
    """

    # Resolve credentials and model from environment when not explicitly provided
    openai_key = openai_key or os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError(
            "OpenAI API key is missing. Set OPENAI_API_KEY in your environment "
            "or pass it explicitly to generate_response()."
        )
    model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")

    # Define system prompt for NASA expertise
    system_prompt = (
        "You are a NASA mission intelligence specialist with deep expertise in historic "
        "space missions including Apollo 11, Apollo 13, and the Challenger (STS-51L) "
        "missions. Use the provided context excerpts from official NASA transcripts and "
        "technical documents to answer the user's question accurately and concisely. "
        "If the context does not contain enough information to answer the question, "
        "say so clearly rather than fabricating details. When relevant, cite the "
        "mission and source document."
    )

    # Build messages list with system prompt
    messages: List[Dict] = [{"role": "system", "content": system_prompt}]

    # Inject the retrieved RAG context (if any) as an additional system message
    if context:
        messages.append({
            "role": "system",
            "content": f"Relevant context from NASA archives:\n\n{context}"
        })

    # Add prior conversation history
    if conversation_history:
        for msg in conversation_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    # Add the current user message
    messages.append({"role": "user", "content": user_message})

    # Create the OpenAI client (supports OpenAI direct or Vocareum proxy via base_url)
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    client_kwargs = {"api_key": openai_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    # Send the request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )

    # Return the assistant's response text
    return response.choices[0].message.content
