from datetime import datetime

from sqlalchemy import Float
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column

from citations_searcher.data import Paper


class SQLBase(DeclarativeBase, MappedAsDataclass):
    pass


class SQLPaper(SQLBase):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    arxiv_id: Mapped[str]
    date: Mapped[datetime]
    title: Mapped[str]
    authors: Mapped[str]
    abstract: Mapped[str]
    category_1: Mapped[str]
    category_2: Mapped[str] = mapped_column(nullable=True, default=None)

    # From Semantic Scholar
    semantic_id: Mapped[str] = mapped_column(nullable=True, default=None)
    citation_count: Mapped[int] = mapped_column(nullable=True, default=None)
    influential_citation_count: Mapped[int] = mapped_column(nullable=True, default=None)
    embedding_v1: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=True, default=None)
    embedding_v2: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=True, default=None)
    tldr: Mapped[str] = mapped_column(nullable=True, default=None)

    @classmethod
    def from_paper(
        cls,
        source_paper: Paper,
    ) -> "SQLPaper":
        return cls(
            **source_paper.to_dict(),
        )


class SQLReferences(SQLBase):
    __tablename__ = "source_paper_references"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)

    paper_arxiv_id: Mapped[str] = mapped_column()
    reference_arxiv_id: Mapped[str] = mapped_column()
