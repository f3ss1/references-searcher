# Postgres table schemas

The project include 4 main tables to work with and 2 tables which are the derivatives
created for convenience and time optimization:

- `papers` -- the table where all the articles metadata is stored.

- `source_paper_references` -- the table where all raw (non-filtered) references
  are stored. These include only the arxiv references and only for a limited pool of papers
  (event though the pool is larger than the final pool).

- `filtered_positive_references` -- same as the `source_article_references`, but
  filtered further as stated in the project description.

- `filtered_negative_references` -- the negative references for papers. Each paper, included
  in the `filtered_positive_references` as the source paper, has a constant number of negative
  references in this table (20). The papers are selected randomly from the pool of all the
  papers, **referenced** in `filtered_positive_references`.

- `joined_filtered_positive_references` -- join of `articles` and `filtered_positive_references`
  to include metadata, suitable for model training (i.e. no "time travel" information about the source paper).

- `joined_filtered_negative_references` -- same as `joined_filtered_positive_references`, but
  for negative references (`filtered_negative_references`).

## `papers`

- `id`: integer

- `arxiv_id`: character varying

  The id of the paper on arxiv.

- `date`: timestamp without time zone

  The publication date, determined as the date of the latest update on arxiv.

- `title`: character varying

- `authors`: character varying

- `abstract`: character varying

- `category_1`: character varying

  The first category of the paper, listed on arxiv. Corresponds to the arxiv's notation.

- `category_2`: character varying

  The second category of the paper, listed on arxiv. Corresponds to the arxiv's notation.

- `semantic_id`: character varying

  The id of the paper at [Semantic Scholar](https://www.semanticscholar.org/)

- `citation_count`: integer 

  The number of citations of this paper, determined by [Semantic Scholar](https://www.semanticscholar.org/).

  This number **does not only include papers, published on arxiv**.

- `influential_citation_count`: integer

  The number of influential citations of this paper, determined by [Semantic Scholar](https://www.semanticscholar.org/).

- `embedding_v1`: ARRAY[float]

  The paper's embedding, obtained using the `SPECTER_v1` model. Obtained from [Semantic Scholar](https://www.semanticscholar.org/).

- `embedding_v2`: ARRAY[float]

  The paper's embedding, obtained using the `SPECTER_v2` model. Obtained from [Semantic Scholar](https://www.semanticscholar.org/).

- `tldr`: character varying

  A short description of the paper, obtained from [Semantic Scholar](https://www.semanticscholar.org/).

## `source_paper_references`

- `id`: integer

- `paper_arxiv_id`: character varying

  The `arxiv_id` of the paper, who is **referencing** another paper.

- `reference_arxiv_id`: character varying

  The `arxiv_id` of the paper, who is **being referencing**.

## `filtered_positive_references`

- `id`: integer

- `paper_arxiv_id`: character varying

  The `arxiv_id` of the paper, who is **referencing** another paper.

- `reference_arxiv_id`: character varying

  The `arxiv_id` of the paper, who is **being referencing**.

## `filtered_negative_references`

Please keep in mind, that these references are negative, meaning they did not actually occur in real life.

- `id`: integer

- `paper_arxiv_id`: character varying

  The `arxiv_id` of the paper, who is **(in fact, not) referencing** another paper.

- `reference_arxiv_id`: character varying

  The `arxiv_id` of the paper, who is **(in fact, not) being referencing**.

## `joined_filtered_positive_references`

- `paper_arxiv_id`: character varying

  The id of the paper on arxiv. It is not supposed to be used in training sequence
  and is provided for debugging and further development only.

- `paper_title`: character varying

- `paper_abstract`: character varying

- `paper_tldr`: character varying

  A short description of the paper, obtained from [Semantic Scholar](https://www.semanticscholar.org/).

- `reference_arxiv_id`: character varying

  The id of the reference on arxiv. It is not supposed to be used in training sequence
  and is provided for debugging and further development only. However, is theoretically
  available for usage (since references are already published).

- `reference_date`: timestamp without time zone

  The publication date, determined as the date of the latest update on arxiv.

- `reference_title`: character varying

- `reference_authors`: character varying

- `reference_abstract`: character varying

- `reference_category_1`: character varying

  The first category of the reference, listed on arxiv. Corresponds to the arxiv's notation.

- `reference_category_2`: character varying

  The second category of the reference, listed on arxiv. Corresponds to the arxiv's notation.

- `reference_citation_count`: integer

  The number of citations of this paper, determined by [Semantic Scholar](https://www.semanticscholar.org/).
  This number **does not only include papers, published on arxiv**.

- `reference_influential_citation_count`: integer
  The number of influential citations of this paper, determined by [Semantic Scholar](https://www.semanticscholar.org/).

- `reference_tldr`: character varying

  A short description of the reference, obtained from [Semantic Scholar](https://www.semanticscholar.org/).

## `joined_filtered_negative_references`

Please keep in mind, that these references are negative, meaning they did not actually occur in real life.

- `paper_arxiv_id`: character varying

  The id of the paper on arxiv. It is not supposed to be used in training sequence
  and is provided for debugging and further development only.

- `paper_title`: character varying

- `paper_abstract`: character varying

- `paper_tldr`: character varying

  A short description of the paper, obtained from [Semantic Scholar](https://www.semanticscholar.org/).

- `reference_arxiv_id`: character varying

  The id of the reference on arxiv. It is not supposed to be used in training sequence
  and is provided for debugging and further development only. However, is theoretically
  available for usage (since references are already published).

- `reference_date`: timestamp without time zone

  The publication date, determined as the date of the latest update on arxiv.

- `reference_title`: character varying

- `reference_authors`: character varying

- `reference_abstract`: character varying

- `reference_category_1`: character varying

  The first category of the reference, listed on arxiv. Corresponds to the arxiv's notation.

- `reference_category_2`: character varying

  The second category of the reference, listed on arxiv. Corresponds to the arxiv's notation.

- `reference_citation_count`: integer

  The number of citations of this paper, determined by [Semantic Scholar](https://www.semanticscholar.org/).
  This number **does not only include papers, published on arxiv**.

- `reference_influential_citation_count`: integer
  The number of influential citations of this paper, determined by [Semantic Scholar](https://www.semanticscholar.org/).

- `reference_tldr`: character varying

  A short description of the reference, obtained from [Semantic Scholar](https://www.semanticscholar.org/).
