from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, JSON, ForeignKey
from sqlalchemy.orm import relationship


Base = declarative_base()


class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)

    papers = relationship(
        "Paper",
        secondary="paper_categories",
        back_populates="categories",
    )

    def __repr__(self):
        return f"Category(id={self.id}, name={self.name})"


class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True)
    url = Column(String)
    abstract = Column(String)
    keywords = Column(JSON)

    categories = relationship(
        "Category",
        secondary="paper_categories",
        back_populates="papers",
    )

    def __repr__(self):
        return f"Paper(id={self.id}, url={self.url})"


class PaperCategory(Base):
    __tablename__ = "paper_categories"

    paper_id = Column(
        Integer,
        ForeignKey("papers.id"),
        primary_key=True,
    )
    category_id = Column(
        Integer,
        ForeignKey("categories.id"),
        primary_key=True,
    )
