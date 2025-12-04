''' retrieval of vanilla rag baseline '''

from retrieval import (
    CorpusLoader,
    QueriesLoader,
    DenseRetrievalExactSearch as DR,
    DenseRetrievalExactSearchMultiDatasets as DRMD,
)
from retrievers import (
    BGE,
    DPR,
    Contriever,
)
from typing import List
import os
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


# KILT benchmark task-specific retrieval instructions
# Reference: https://github.com/facebookresearch/KILT
TASK_POOL = {
    # Natural Questions: open-domain QA from Google search queries
    'nq': {
        'retrieval_instruction': 'Given a factoid question, retrieve Wikipedia passages that contain the answer.'
    },
    # HotpotQA: multi-hop reasoning QA requiring information from multiple documents
    'hotpotqa': {
        'retrieval_instruction': 'Given a multi-hop question requiring reasoning over multiple facts, retrieve Wikipedia passages that provide supporting evidence.'
    },
    # ELI5: long-form QA with explanatory answers (Explain Like I\'m 5)
    'eli5': {
        'retrieval_instruction': 'Given an open-ended question seeking explanation, retrieve Wikipedia passages that help provide a comprehensive answer.'
    },
    # FEVER: fact verification - classify claims as SUPPORTED, REFUTED, or NOT ENOUGH INFO
    'fever': {
        'retrieval_instruction': 'Given a factual claim, retrieve Wikipedia passages that can be used to verify or refute the claim.'
    },
    # Wizard of Wikipedia: knowledge-grounded dialogue generation
    'wow': {
        'retrieval_instruction': 'Given a dialogue context about a topic, retrieve Wikipedia passages that provide relevant knowledge to continue the conversation.'
    },
    # T-REx: slot filling - extracting object entities given subject and relation
    'trex': {
        'retrieval_instruction': 'Given a subject entity and relation, retrieve Wikipedia passages that contain the object entity completing the fact.'
    },
    # Zero-Shot RE: relation extraction without training examples
    'zs-re': {
        'retrieval_instruction': 'Given a subject entity and relation description, retrieve Wikipedia passages that mention the related object entity.'
    },
}


def passages_retrieval_multiset(dataset_list: List[str],
                                corpus_file: str,
                                base_folder: str = './kr_data',
                                seed_id: str = '0',
                                model_name: str = 'dpr',
                                top_k: int = 5,
                                batch_size: int = 128,
                                corpus_chunk_size: int = 10000):
    '''
    retrieve relevant passages for shared context construction based on entities, 
    for multiple datasets at the same time
    '''
    corpus = CorpusLoader(corpus_path=corpus_file).load()
    queries_list, inputs_list, outputs_list = [], [], []
    for dataset in dataset_list:
        filename = os.path.join(base_folder, dataset, f'test_{seed_id}.jsonl')
        queries, inputs, outputs = QueriesLoader(
            data_path=filename, query_type='input', task_instruction=TASK_POOL[dataset]['retrieval_instruction']).load()
        queries_list.append(queries)
        inputs_list.append(inputs)
        outputs_list.append(outputs)
    
    if model_name == 'dpr':
        model = DPR((
            'facebook/dpr-question_encoder-multiset-base',
            'facebook/dpr-ctx_encoder-multiset-base'
        ))
        retriever = DRMD(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'bge-base' or model_name == 'bge':
        model = BGE('BAAI/bge-base-en-v1.5')
        retriever = DRMD(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    else:
        logging.error(f'Wrong retriever: {model_name}')

    results_list = retriever.search(corpus, queries_list, top_k=top_k, score_function="cos_sim", return_sorted=False)

    output_filename_list = [os.path.join(base_folder, dataset, f'test_{seed_id}_w_passages_{model_name}.jsonl') for dataset in dataset_list]
    for output_filename, results, inputs, outputs in zip(output_filename_list, results_list, inputs_list, outputs_list):
        # dict_keys(['_id', 'input', 'docs', 'output'])
        with open(output_filename, 'w') as file:
            for query_id, ranking_scores in results.items():
                sorted_scores = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
                docs = [{'title': corpus[doc_id].get('title'), 'text': corpus[doc_id].get('text'), 'score': score} for doc_id, score in sorted_scores]
                line = {'_id': query_id, 'input': inputs[query_id], 'docs': docs, 'output': outputs[query_id]}
                file.write(json.dumps(line) + '\n')
    
    return {
        'output_file_list': output_filename_list
    }


def main():
    # wiki_datasets = ['nq', 'hotpotqa', 'eli5', 'fever', 'wow', 'trex', 'zs-re']
    wiki_datasets = ['nq']
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever', required=True, type=str, help='The retriever model path.')
    parser.add_argument('--corpus_path', required=True, type=str, help='The corpus file path.')
    parser.add_argument('--topk', default=5, required=True, type=int)
    parser.add_argument('--wiki_dataset',required=True, type=str, help='The wiki dataset to process.')
    args = parser.parse_args()
    wiki_datasets = [args.wiki_dataset]

    if 'wiki' in args.corpus_path:
        outputs = passages_retrieval_multiset(
            base_folder = "./kr_data/kilt_tasks",
            dataset_list=wiki_datasets,
            corpus_file=args.corpus_path,
            model_name=args.retriever,
            top_k=args.topk,
            batch_size=4096,           # ↑ Larger batch for GPU utilization
            corpus_chunk_size=50000,   # ↑ Fewer similarity matrix computations
        )
        logging.info(outputs)


if __name__ == '__main__':
    main()
