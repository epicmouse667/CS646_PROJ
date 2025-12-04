wiki_dataset=trex
echo conducting retreival on $wiki_dataset
python retrieval/main.py \
    --retriever bge \
    --corpus_path kr_data/wikipedia_100.jsonl \
    --topk 20 --wiki_dataset $wiki_dataset
