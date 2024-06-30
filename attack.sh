export OUTPUT_DIR=./data_out/scrn_in-domain # path/model_dataset
export MODEL_TYPE=hf
export BASE_MODEL=roberta-base
export NUM_EXAMPLES=200
export ENSEMBLE_NUM=1
export MASK_PERCENTAGE=0.30
export TRANSFER_DATASET_ABBR=self
export ATTACK_CLASS=ai # [human, ai]
export ATTACK_RECIPE=deep-word-bug # [pwws, deep-word-bug, pruthi]

python3 -u run_attack.py \
--model_type ${MODEL_TYPE} \
--bert_name_or_path ${BASE_MODEL} \
--metric_base_model_name_or_path gpt2 \
--attack_class ${ATTACK_CLASS} \
--attack_recipe ${ATTACK_RECIPE} \
--transfer_dataset_abbr ${TRANSFER_DATASET_ABBR} \
--output_dir ${OUTPUT_DIR} \
--num_examples ${NUM_EXAMPLES} \
--ensemble_num ${ENSEMBLE_NUM} \
--mask_percentage ${MASK_PERCENTAGE} \