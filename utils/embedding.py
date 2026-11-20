def generate_embedding_batches(client, texts, batch_size=100):
    """Yield each batch of document vectors so callers can store them at once."""
    if not texts:
        raise ValueError("There is no document text to embed.")

    for start in range(0, len(texts), batch_size):
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts[start : start + batch_size],
        )
        vectors = [embedding.values for embedding in response.embeddings]
        expected_count = min(batch_size, len(texts) - start)
        if len(vectors) != expected_count:
            raise ValueError("Gemini returned an incomplete batch of document embeddings.")
        yield start, vectors


def generate_embeddings(client, texts, batch_size=100):
    """Create embeddings in batches (used by the learning scripts)."""
    vectors = []
    for _, batch in generate_embedding_batches(client, texts, batch_size):
        vectors.extend(batch)
    return vectors


def generate_query_embedding(client, query):
    """Create the embedding used to find chunks related to a question."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
    )
    if not response.embeddings:
        raise ValueError("Gemini did not return a question embedding.")
    return response.embeddings[0].values
