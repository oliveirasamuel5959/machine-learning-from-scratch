persist-nb:
    uv run jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.kernel_name=mlscratch \
    02_LDA.ipynb

ruff-check:
    uv run ruff check --select I --fix .

ruff-format:
    uv run ruff format .
