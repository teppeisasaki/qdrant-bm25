# qdrant-bm25
BM25 Search in Qdrant

## Installation
You need to install packages in `requirements.txt` and fastembed which supports `disable_stemmer` option.
```bash
cd qdrant-bm25

# Install packages in requirements
pip install -r requirements.txt

# Clone FastEmbed
git clone git@github.com:qdrant/fastembed.git
cd fastembed

# Install Poetry
python -m pip install poetry
poetry install

# Build & Install FastEmbed
poetry build
pip install fastembed/dist/fastembed-0.4.2-py3-none-any.whl
```

## Quickstart
```bash
cd qdrant-bm25

# Launch Qdrant
docker compose up -d

# Execute script
python main.py
```

## Attribution
### [globis-university/aozorabunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean)
The dataset used in this project is provided by globis-university and is licensed under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).
