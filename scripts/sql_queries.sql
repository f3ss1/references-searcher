-- Create positive filtered entries
CREATE TABLE filtered_positive_references AS
SELECT
    DISTINCT source_paper_references.*
FROM source_paper_references
INNER JOIN papers
ON source_paper_references.paper_arxiv_id = papers.arxiv_id
WHERE papers.category_1 = 'cs.lg'
    AND papers.date > '2019-12-31'
    AND papers.date < '2022-11-01'
    AND papers.citation_count > 10
    AND EXISTS (
        SELECT 1
        FROM papers AS papers_2
        WHERE source_paper_references.reference_arxiv_id = papers_2.arxiv_id
    );

-- Create positive joined entries
CREATE TABLE joined_filtered_positive_references AS
SELECT
    paper.arxiv_id AS paper_arxiv_id,
    paper.title AS paper_title,
    paper.abstract AS paper_abstract,
    paper.tldr AS paper_tldr,
    reference.arxiv_id AS reference_arxiv_id,
    reference.date AS reference_date,
    reference.title AS reference_title,
    reference.authors AS reference_authors,
    reference.abstract AS reference_abstract,
    reference.category_1 AS reference_category_1,
    reference.category_2 AS reference_category_2,
    reference.citation_count as reference_citation_count,
    reference.influential_citation_count as reference_influential_citation_count,
    reference.tldr as reference_tldr
FROM papers AS paper
INNER JOIN filtered_positive_references ON paper.arxiv_id = filtered_positive_references.paper_arxiv_id
INNER JOIN papers AS reference ON filtered_positive_references.reference_arxiv_id = reference.arxiv_id;

-- Find all referenced articles and create negative entries
CREATE VIEW referenced_papers AS
SELECT
    DISTINCT reference_arxiv_id as arxiv_id
FROM filtered_positive_references;

CREATE TABLE filtered_negative_references AS
SELECT
    paper_arxiv_id,
    reference_arxiv_id
FROM (
    SELECT
        papers.paper_arxiv_id AS paper_arxiv_id,
        referenced_papers.arxiv_id as reference_arxiv_id,
        ROW_NUMBER() OVER (PARTITION BY papers.paper_arxiv_id ORDER BY RANDOM()) as rn
    FROM referenced_papers
    CROSS JOIN (
        SELECT DISTINCT paper_arxiv_id
        FROM filtered_positive_references
    ) as papers
    LEFT JOIN filtered_positive_references AS ref ON papers.paper_arxiv_id = ref.paper_arxiv_id AND referenced_papers.arxiv_id = ref.reference_arxiv_id
    WHERE ref.reference_arxiv_id IS NULL AND papers.paper_arxiv_id != referenced_papers.arxiv_id
) as sub
WHERE rn <= 20;

-- Create negative joined entries
CREATE TABLE joined_filtered_negative_references AS
SELECT
    paper.arxiv_id AS paper_arxiv_id,
    paper.title AS paper_title,
    paper.abstract AS paper_abstract,
    paper.tldr AS paper_tldr,
    reference.arxiv_id AS reference_arxiv_id,
    reference.date AS reference_date,
    reference.title AS reference_title,
    reference.authors AS reference_authors,
    reference.abstract AS reference_abstract,
    reference.category_1 AS reference_category_1,
    reference.category_2 AS reference_category_2,
    reference.citation_count as reference_citation_count,
    reference.influential_citation_count as reference_influential_citation_count,
    reference.tldr as reference_tldr
FROM papers AS paper
INNER JOIN filtered_negative_references ON paper.arxiv_id = filtered_negative_references.paper_arxiv_id
INNER JOIN papers AS reference ON filtered_negative_references.reference_arxiv_id = reference.arxiv_id;

-- Create all train references metadata
CREATE TABLE train_references_metadata AS
SELECT
    paper.arxiv_id AS arxiv_id,
    paper.date AS date,
    paper.title AS title,
    paper.authors AS authors,
    paper.abstract AS abstract,
    paper.category_1 AS category_1,
    paper.category_2 AS category_2,
    paper.citation_count as citation_count,
    paper.influential_citation_count as influential_citation_count,
    paper.tldr as tldr
FROM papers AS paper
INNER JOIN (
    SELECT DISTINCT reference_arxiv_id
    FROM filtered_positive_references
) as reference ON paper.arxiv_id = reference.reference_arxiv_id;

-- Create all total possible references metadata
CREATE TABLE train_references_metadata AS
SELECT
    paper.arxiv_id AS arxiv_id,
    paper.date AS date,
    paper.title AS title,
    paper.authors AS authors,
    paper.abstract AS abstract,
    paper.category_1 AS category_1,
    paper.category_2 AS category_2,
    paper.citation_count as citation_count,
    paper.influential_citation_count as influential_citation_count,
    paper.tldr as tldr
FROM papers AS paper
INNER JOIN (
    SELECT DISTINCT reference_arxiv_id
    FROM filtered_positive_references
) as reference ON paper.arxiv_id = reference.reference_arxiv_id

