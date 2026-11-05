persist-nb:
    uv run jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.kernel_name=mlscratch \
    02_LDA.ipynb
