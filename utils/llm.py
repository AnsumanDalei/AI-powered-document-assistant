def generate_answer(client, chunks, question):
    """Classify scope and answer using only the supplied PDF excerpts."""
    excerpts = "\n\n".join(
        f"[Source: {chunk['source']}, section {chunk['section']}]\n{chunk['text']}"
        for chunk in chunks
    )
    prompt = f"""You answer questions only from the supplied PDF excerpts. Do not use outside knowledge.

First decide whether the question is meaningfully related to the subject of the supplied excerpts.
- If it is unrelated (for example, asking weather when the excerpts discuss Python), output exactly:
OUT_OF_SCOPE
This question is outside the scope of the uploaded PDF. I can answer questions related to your document only. Please upload a relevant PDF for the topic you want to discuss.
- If it is related but the requested information is not stated in the excerpts, output exactly:
NOT_FOUND
I couldn't find this information in the uploaded PDF. Please upload a document containing this information if you want me to answer it.
- Otherwise output:
ANSWER
Then give a concise, clear answer supported only by the excerpts. Add the source PDF filename(s) in parentheses when practical. For summary questions, summarize only the provided excerpts and say it is a summary of the retrieved passages.

PDF excerpts:
{excerpts}

Question: {question}
Response:"""
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )
    text = (response.text or "").strip()
    if text.startswith("OUT_OF_SCOPE"):
        return "This question is outside the scope of the uploaded PDF. I can answer questions related to your document only. Please upload a relevant PDF for the topic you want to discuss."
    if text.startswith("NOT_FOUND"):
        return "I couldn't find this information in the uploaded PDF. Please upload a document containing this information if you want me to answer it."
    if text.startswith("ANSWER"):
        return text[len("ANSWER") :].strip() or "I couldn't find this information in the uploaded PDF. Please upload a document containing this information if you want me to answer it."
    # Fail closed if the model does not follow the required response format.
    return "I couldn't find this information in the uploaded PDF. Please upload a document containing this information if you want me to answer it."
