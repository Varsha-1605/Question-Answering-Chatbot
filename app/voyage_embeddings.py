from io import BytesIO
from typing import Literal

from langchain_core.embeddings import Embeddings
from PIL import Image

InputType = Literal["document", "query"]
# A 1600x1200 image is ~3.4k tokens; 16 per request stays far below the 320k limit.
IMAGE_BATCH_SIZE = 16


class VoyageEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
        batch_size: int,
        timeout: float,
        max_retries: int,
    ) -> None:
        import voyageai

        self._client = voyageai.Client(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._embed([[text] for text in batch], "document"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([[text]], "query")[0]

    def embed_image(self, image_bytes: bytes, text: str) -> list[float]:
        """Embed one image together with its description as a single document vector."""
        return self.embed_images([(image_bytes, text)])[0]

    def embed_images(self, items: list[tuple[bytes, str]]) -> list[list[float]]:
        """Embed (image, description) pairs, a few per request.

        Images cost ~1 token per 560 pixels, so batches stay small to keep each
        request under the 320k-token limit; only one batch is decoded at a time.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(items), IMAGE_BATCH_SIZE):
            batch = [_image_input(data, text) for data, text in items[start : start + IMAGE_BATCH_SIZE]]
            vectors.extend(self._embed(batch, "document"))
        return vectors

    def _embed(self, inputs: list[list], input_type: InputType) -> list[list[float]]:
        result = self._client.multimodal_embed(
            inputs=inputs,
            model=self._model,
            input_type=input_type,
            output_dimension=self._dimensions,
        )
        return [list(vector) for vector in result.embeddings]


def _image_input(image_bytes: bytes, text: str) -> list:
    with Image.open(BytesIO(image_bytes)) as image:
        loaded = image.copy()  # reads the pixels, so the buffer can be released
    return [text, loaded] if text else [loaded]
