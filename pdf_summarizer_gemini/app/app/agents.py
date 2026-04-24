import google.generativeai as genai


def get_model(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")


def safe_generate(model, prompt, fallback_text):
    """
    Wrapper to handle API quota errors gracefully.
    """
    try:
        response = model.generate_content(prompt)
        return response.text if response.text else fallback_text
    except Exception as e:
        print(f"⚠️ Gemini API failed: {e}")
        return fallback_text


def analyzer_agent(model, text: str) -> str:
    prompt = f"""
    Analyze the following document and identify:
    - Main topics
    - Document type
    - Overall structure

    Document text:
    {text[:3000]}
    """

    return safe_generate(
        model,
        prompt,
        fallback_text="Basic analysis: Document contains general informational content."
    )


def summarizer_agent(model, context: str) -> str:
    prompt = f"""
    You are a summarization assistant.

    Rules:
    - Use ONLY the provided context.
    - Do NOT add external information.

    Context:
    {context}

    Write a clear and concise summary:
    """

    return safe_generate(
        model,
        prompt,
        fallback_text=context[:1000]  # fallback summary
    )


def editor_agent(model, summary: str) -> str:
    prompt = f"""
    Improve the following summary:
    - Remove repetition
    - Improve clarity and flow

    Summary:
    {summary}
    """

    return safe_generate(
        model,
        prompt,
        fallback_text=summary.strip()
    )