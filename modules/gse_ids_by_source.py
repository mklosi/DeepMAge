source_to_gse_ids = {
    "series_matrix": [
        "GSE102177",
        "GSE103911",
        "GSE105123",
        "GSE106648",
        "GSE107459",
        "GSE107737",
        "GSE112696",
        "GSE19711",
        "GSE20067",
        "GSE27044",
        "GSE30870",
        # "GSE34639",  # *
        # "GSE37008",  # *
        "GSE40279",
        "GSE41037",
        "GSE52588",
        "GSE53740",
        "GSE58119",
        # "GSE67530",  # a very slight negative value.
        "GSE77445",
        "GSE79329",
        # "GSE81961",  # *
        "GSE84624",
        "GSE87582",
        "GSE87640",
        "GSE97362",
        "GSE98876",
        "GSE99624",
    ],
    "matrix_normalized": ["GSE125105", "GSE128235"],
    "GSE59065_MatrixProcessed.csv.gz": ["GSE59065"],
    "GSE61496_Processed.csv.gz": ["GSE61496"],
    "GSE77696_MatrixProcessed.txt.gz": ["GSE77696"],
}

gse_id_to_source = {
    gse_id: source
    for source, gse_ids in source_to_gse_ids.items()
    for gse_id in gse_ids
}
