# eli5 wow trex fever
task=fever
adaptive=True
cad=False
cf=False
#ck, trust_context, trust_parametric, balanced, context_if_confident
mode=trust_context

echo eval rag on   $task, adaptive: $adaptive, cad: $cad, cf: $cf, mode: $mode
python eval_rag.py \
    --model_name ./checkpoints/Meta-Llama-3-8B-Instruct \
    --mode $mode --adaptive $adaptive --cad $cad --cf $cf \
    --input_file /proj/inf-scaling/physhuman/code/CK-PLUG/kr_data/kilt_tasks/$task/test_0_w_passages_bge.jsonl \
    --task $task