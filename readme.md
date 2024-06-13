# References Searcher

## `.env` file

The project is set up with secrets using the `.env` file you have to provide on your own. An example one is as follows:

```bash
POSTGRES_USER=postgres_user
POSTGRES_PASSWORD=postgres_password
POSTGRES_DB=database_name
FLOWER_USER=flower_user
FLOWER_PASSWORD=flower_password
SCHOLAR_API_KEY=scholar_api_key
DJANGO_SECRET_KEY=django_secret_key
```

## Running the project

To run the project, you first need to populate the database. For that, run only the postgres service using docker compose.

After it's up, you can use the `get_data.py` to run the process of extracting necessary data from Semantic Scholar.

Once populated, you can either train your model using `main.py` (configured with `config.yaml`), or use a pretrained one
if you have any to run the app. For that, run `docker compose up`. The app should be available at `localhost`.