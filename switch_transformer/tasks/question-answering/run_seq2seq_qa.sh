model_name_or_path=google/switch-base-8
dataset_name=squad
context_column=context
question_column=question
answer_column=answers
do_train=True
do_eval=True

per_device_train_batch_size=8
per_device_eval_batch_size=8

learning_rate=3e-5
num_train_epochs=10
max_seq_length=384
doc_stride=128
eval_accumulation_steps=10

experts_embedding=False
use_hypernet=False

process_dim=128
hypernet_input=128
hypernetwork_bottleneck=128
layer_emb_dim=10
adapter_dim=64
experts_embedding_dim=128


output_dir=./checkpoints_moe/${model_name_or_path##*/}/${dataset_name}/${learning_rate}
#output_dir=./evaluation/checkpoints_moe/${model_name_or_path##*/}/${dataset_name}/${learning_rate}
dataloader_num_workers=16
evaluation_strategy=epoch
save_strategy=no
predict_with_generate=True

echo "${output_dir}"
mkdir -p ${output_dir}

echo    --model_name_or_path ${model_name_or_path} \
        --output_dir ${output_dir} \
        --dataset_name ${dataset_name} \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --num_train_epochs ${num_train_epochs} \
        --overwrite_output_dir \
        --do_train ${do_train} --do_eval ${do_eval} \
        --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
        --evaluation_strategy ${evaluation_strategy} \
        --save_strategy ${save_strategy} \
        --eval_accumulation_steps ${eval_accumulation_steps} \


python run_seq2seq_qa.py \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ${output_dir} \
        --dataset_name ${dataset_name} \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --num_train_epochs ${num_train_epochs} \
        --overwrite_output_dir \
        --experts_embedding ${experts_embedding} \
        --use_hypernet ${use_hypernet} \
        --process_dim ${process_dim} \
        --hypernetwork_bottleneck ${hypernetwork_bottleneck} \
        --hypernet_input ${hypernet_input} \
        --layer_emb_dim ${layer_emb_dim} \
        --adapter_dim ${adapter_dim} \
        --experts_embedding_dim ${experts_embedding_dim} \
        --do_train ${do_train} --do_eval ${do_eval} \
        --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
        --evaluation_strategy ${evaluation_strategy} \
        --save_strategy ${save_strategy} \
        --eval_accumulation_steps ${eval_accumulation_steps} --predict_with_generate ${predict_with_generate}
