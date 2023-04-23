"""Taken from: https://docs.pinecone.io/docs/hybrid-search"""
import hashlib
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Extra, root_validator
from langchain.embeddings.base import Embeddings
from pinecone_text.hybrid import hybrid_convex_scale



def hash_text(text: str) -> str:
    return str(hashlib.sha256(text.encode("utf-8")).hexdigest())


def create_vectors(
    contexts: List[str],
    embeddings: Embeddings,
    sparse_encoder: Any,
    ids: Optional[List[str]] = None,
    meta_dict: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generates sparse-dense vectors for each input text and returns them in a list of dictionaries.
    Each dictionary contains the keys 'id', 'values', 'metadata', and 'sparse_values'.

    :param contexts: List of input texts.
    :param embeddings: Embeddings object used to generate dense vectors.
    :param sparse_encoder: BM25Encoder object used to generate sparse vectors.
    :param ids: Optional list of IDs to use for each input text. If not provided, unique IDs will be generated.
    :param meta_dict: Optional dictionary containing metadata to be added to each vector. The keys of the dictionary
                      will be used as the metadata field names, and the values will be used as the metadata values.
    :return: List of dictionaries, each containing a sparse-dense vector for each input text.
    """
    vectors = []

    if ids is None:
        ids = [hash_text(context) for context in contexts]

    for context, doc_id in zip(contexts, ids):
        # add metadata to each vector
        metadata = {"context": context}
        if meta_dict is not None:
            metadata.update(meta_dict)
        # create dense vector
        dense_embed = embeddings.embed_documents([context])[0]
        # create sparse vector
        sparse_embed = sparse_encoder.encode_documents([context])[0]
        # convert sparse values to floats
        sparse_embed["values"] = [float(val) for val in sparse_embed["values"]]
        # add vector to list of vectors
        vectors.append(
            {
                "id": doc_id,
                "values": dense_embed,
                "metadata": metadata,
                "sparse_values": sparse_embed,
            }
        )

    return vectors

class PineconeHybridVectorCreator(BaseModel):
    embeddings: Embeddings
    sparse_encoder: Any
    alpha: float = 0.5

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def generate_vectors(
        self,
        contexts: List[str],
        ids: Optional[List[str]] = None,
        meta_dicts: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        vectors = []

        if ids is None:
            ids = [hash_text(context) for context in contexts]

        for idx, (context, doc_id) in enumerate(zip(contexts, ids)):
            # add metadata to each vector
            metadata = {"context": context}
            if meta_dicts is not None:
                metadata.update(meta_dicts[idx])

            # create dense vector
            dense_embed = self.embeddings.embed_documents([context])[0]

            # create sparse vector
            sparse_embed = self.sparse_encoder.encode_documents([context])[0]
            sparse_embed["values"] = [float(val) for val in sparse_embed["values"]]

            # scale dense and sparse vectors using hybrid_convex_scale
            dense_embed, sparse_embed = hybrid_convex_scale(dense_embed, sparse_embed, self.alpha)

            # add vector to list of vectors
            vectors.append(
                {
                    "id": doc_id,
                    "values": dense_embed,
                    "metadata": metadata,
                    "sparse_values": sparse_embed,
                }
            )

        return vectors
