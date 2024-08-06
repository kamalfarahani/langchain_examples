import json

import umap
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from langchain_core.embeddings import Embeddings

from paper_reader.datamanager import DBManager


class KeywordClusterer:
    def __init__(
        self,
        embeddings: Embeddings,
        db_manager: DBManager,
        n_components: int = 256,
    ) -> None:
        """
        Initializes the keyword clusterer.

        Args:
            embeddings: The embeddings to use.
            db_manager: The database manager to use.

        Returns:
            None
        """
        self.embeddings = embeddings
        self.db_manager = db_manager
        self.n_components = n_components

    def cluster_keywords(
        self,
        category_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Clusters the keywords.

        Args:
            category_name: The name of the category.
            embeddings: The embeddings to use.
            db_manager: The database manager to use.

        Returns:
            (np.ndarray, np.ndarray, np.ndarray): The raw keywords, embedded keywords, and labels.
        """
        raw_keywords, embedded_keywords = self.get_category_data(category_name)

        # This line is necessary for umap to work see [this issue](https://github.com/lmcinnes/umap/issues/201)
        n_components = min(self.n_components, len(embedded_keywords) // 2)
        embedded_keywords_reduced = reduce_dimensions(
            data=embedded_keywords,
            n_components=n_components,
        )

        dbscan = DBSCAN(min_samples=5, eps=0.5, n_jobs=-1)
        dbscan.fit(embedded_keywords_reduced)

        return raw_keywords, embedded_keywords, dbscan.labels_

    def get_category_data(
        self,
        category_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Gets keywords in a category and embeds them then
        returns the keywords and embedded keywords.

        Args:
            category_name: The name of the category.
            embeddings: The embeddings model.
            db_manager: The database manager.

        Returns:
            (np.ndarray, np.ndarray): The raw keywords and embedded keywords.
        """
        category = self.db_manager.get_category_by_name(name=category_name)

        raw_keywords = []
        embedded_keywords = []
        for paper in self.db_manager.get_papers_by_category(category_id=category.id):
            keywords = json.loads(paper.keywords)["keywords"]
            for keyword in keywords:
                raw_keywords.append(keyword)
                embedded_keywords.append(
                    np.array(self.embeddings.embed_query(keyword)),
                )

        raw_keywords = np.array(raw_keywords)
        embedded_keywords = np.array(embedded_keywords)

        return raw_keywords, embedded_keywords


def reduce_dimensions(
    data: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    """
    Reduces the dimensions of the data.

    Args:
        data: The data to reduce.
        n_components: The number of dimensions to reduce to.

    Returns:
        (np.ndarray): The reduced data.
    """
    reducer = umap.UMAP(
        n_neighbors=25,
        n_components=n_components,
        min_dist=0.01,
        metric="euclidean",
    )

    return reducer.fit_transform(data)
