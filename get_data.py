from references_searcher.data.sql import DatabaseInterface

# Assuming there are no tables in the database.
interface = DatabaseInterface()
interface.upload_papers_to_db()
interface.get_semantic_data(successful_requests_to_save=1)
