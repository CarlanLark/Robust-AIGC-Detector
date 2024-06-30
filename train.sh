# metric-based detectors: ["Log-Likelihood", "Log-Rank", "Entropy", "GLTR"]
# model-based detectors: ['bert-base', 'roberta-base', 'deberta-base', 'ChatGPT-Detector', 'flooding', 'rdrop', 'ranmask', 'scrn']

export DATASET_ABBR=mixed-source
export MODEL_ABBR=scrn
export BERT_MODEL=roberta-base # just used for huggingface wrapped model

python3 -u main.py  \
--do_train True \
--do_predict True \
--cache_dir .cache  \
--seed 2020 \
--save_total_limit 5 \
--learning_rate 1e-4 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 2.0 \
--max_seq_length 512 \
--num_labels 2 \
--logging_steps 50 \
--gradient_accumulation_steps 1 \
--metric_base_model_name_or_path gpt2 \
--model_name_or_path ${BERT_MODEL} \
--data_files ./data_in/${DATASET_ABBR}  \
--output_dir ./data_out/${MODEL_ABBR} \