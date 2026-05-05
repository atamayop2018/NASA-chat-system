from typing import Dict, List
from openai import OpenAI


def generate_response(openai_key: str, user_message: str, context: str,
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt for NASA domain expertise
    system_prompt = (
        "You are a knowledgeable NASA space mission expert assistant. "
        "You specialize in the Apollo 11, Apollo 13, and Challenger missions. "
        "Use the provided context from official NASA documents and transcripts to "
        "answer the user's question as accurately as possible. "
        "If the context does not contain enough information to answer, say so clearly "
        "rather than fabricating details. Always cite the mission name when relevant, "
        "and prefer concise, factual answers grounded in the supplied context."
    )

    # Build messages list
    messages: List[Dict] = [{"role": "system", "content": system_prompt}]

    # Set context in messages (as an additional system-style message)
    if context:
        messages.append({
            "role": "system",
            "content": f"Relevant context retrieved from the NASA document corpus:\n\n{context}"
        })

    # Add prior chat history (limit to last ~10 turns to control token usage)
    if conversation_history:
        for msg in conversation_history[-10:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    # Create OpenAI client
    client = OpenAI(api_key=openai_key)

    # Send request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )

    # Return response text
    return response.choices[0].message.content.strip()
