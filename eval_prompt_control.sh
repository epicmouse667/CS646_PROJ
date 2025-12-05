# eli5 wow trex fever
task=fever
adaptive=True
#ck, trust_context, trust_parametric, balanced, context_if_confident
mode=trust_parametric
for mode in balanced context_if_confident; do
    echo eval rag on   $task, adaptive: $adaptive mode: $mode
    python eval_rag.py \
        --model_name ./checkpoints/Meta-Llama-3-8B-Instruct \
        --mode $mode --adaptive $adaptive \
        --input_file /proj/inf-scaling/physhuman/code/CK-PLUG/kr_data/kilt_tasks/$task/test_0_w_passages_bge.jsonl \
        --task $task
done