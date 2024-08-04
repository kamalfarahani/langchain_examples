import json

import paper_reader.paper as P

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from paper_reader.datamanager.models import Paper, Category, PaperCategory, Base


class DBManager:
    def __init__(self, db_path: str) -> None:
        """
        Initializes the database manager.

        Args:
            db_path: The path to the database.

        Returns:
            None
        """
        self.engine = create_engine(db_path)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.create_tables()

    def create_tables(self) -> None:
        """
        Creates the tables.

        Args:
            None

        Returns:
            None
        """
        Base.metadata.create_all(self.engine)

    def add_paper(self, paper: P.Paper, keywords: list[str]) -> None:
        """
        Adds the paper to the database.

        Args:
            paper: The paper to add.

        Returns:
            None
        """
        authors_json = json.dumps(
            {
                "authors": paper.authors,
            }
        )

        keywords_json = json.dumps(
            {
                "keywords": keywords,
            }
        )

        paper = Paper(
            url=paper.url,
            title=paper.title,
            authors=authors_json,
            year=paper.year,
            abstract=paper.abstract,
            keywords=keywords_json,
        )
        self.session.add(paper)
        self.session.commit()

    def add_category(self, category: str):
        """
        Adds the category to the database.

        Args:
            category: The category to add.

        Returns:
            None
        """
        category = Category(name=category)
        self.session.add(category)
        self.session.commit()

    def add_paper_category(self, paper_url: str, category_name: str):
        """
        Adds the paper category to the database.

        Args:
            paper_url: The url of the paper.
            category_name: The name of the category.

        Returns:
            None
        """
        category_id = self.get_category_by_name(category_name).id
        paper_id = self.get_paper_by_url(paper_url).id
        paper_category = PaperCategory(paper_id=paper_id, category_id=category_id)
        self.session.add(paper_category)
        self.session.commit()

    def get_paper_by_id(self, paper_id: int) -> Paper:
        """
        Returns the paper by the id.

        Args:
            paper_id: The id of the paper.

        Returns:
            (Paper): The paper.
        """
        return self.session.query(Paper).filter_by(id=paper_id).first()

    def get_paper_by_url(self, url: str) -> Paper:
        """
        Returns the paper by the url.

        Args:
            url: The url of the paper.

        Returns:
            (Paper): The paper.
        """
        return self.session.query(Paper).filter_by(url=url).first()

    def get_papers_by_category(self, category_id: int) -> list[Paper]:
        """
        Returns the papers by the category id.

        Args:
            category_id: The id of the category.

        Returns:
            (list[Paper]): The papers.
        """
        return (
            self.session.query(Paper)
            .join(PaperCategory)
            .join(Category)
            .filter(Category.id == category_id)
            .all()
        )

    def get_categories(self) -> list[Category]:
        """
        Returns the categories.

        Args:
            None

        Returns:
            (list[Category]): The categories.
        """
        return self.session.query(Category).all()

    def get_category_by_id(self, category_id: int) -> Category:
        """
        Returns the category by the id.

        Args:
            category_id: The id of the category.

        Returns:
            (Category): The category.
        """
        return self.session.query(Category).filter_by(id=category_id).first()

    def get_category_by_name(self, name: str) -> Category:
        """
        Returns the category by the name.

        Args:
            name: The name of the category.

        Returns:
            (Category): The category.
        """
        return self.session.query(Category).filter_by(name=name).first()
