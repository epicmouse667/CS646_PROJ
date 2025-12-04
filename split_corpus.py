from retrieval.preprocess import split_corpus_to_100

split_corpus_to_100(
    in_corpus_file='./kr_data/wikipedia_2019_08_01.json',
    out_corpus_file='./kr_data/wikipedia_100.jsonl'
)