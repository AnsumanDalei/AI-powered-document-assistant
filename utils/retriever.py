import numpy as np


def retrieve_chunks(index, query_embedding, chunks, k=5, broad=False):
    """Return similar excerpts, plus a few spread-out excerpts for summaries."""
    if index.ntotal == 0 or not chunks:
        return []

    query_vector = np.array([query_embedding], dtype="float32")
    result_count = min(k, index.ntotal)
    _, indices = index.search(query_vector, result_count)
    selected = [int(i) for i in indices[0] if 0 <= i < len(chunks)]

    if broad and len(chunks) > result_count:
        # Include document-wide coverage so a summary is not based on one passage.
        target_count = min(10, len(chunks))
        if target_count > 1:
            for position in range(target_count):
                sample_index = round(position * (len(chunks) - 1) / (target_count - 1))
                if sample_index not in selected:
                    selected.append(sample_index)

    return [chunks[index] for index in selected[:10]]
