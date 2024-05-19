TASK_NAME=qnli
dataset_config_name="en"


model_name_or_path=google/switch-base-8
do_train=True
do_eval=True

per_device_train_batch_size=8
per_device_eval_batch_size=8

experts_embedding=False
use_hypernet=False

process_dim=128
hypernet_input=128
hypernetwork_bottleneck=128
layer_emb_dim=10
adapter_dim=64
experts_embedding_dim=128

max_source_length=256
predict_with_generate=True
num_train_epochs=10
dataloader_num_workers=16
evaluation_strategy=epoch
save_strategy=no
weight_decay=0.1
learning_rate=1e-5
seed=42
compute_memory=True
output_dir=./checkpoints_moe/${model_name_or_path##*/}/${TASK_NAME}/${learning_rate}

echo "${output_dir}"
mkdir -p ${output_dir}

echo  --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --task_name ${TASK_NAME} \
      --n_experts ${n_experts} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --num_train_epochs ${num_train_epochs} \
      --weight_decay ${weight_decay} --learning_rate ${learning_rate} \
      --overwrite_output_dir \
      --do_train ${do_train}\
      --do_eval \
      --weight_decay ${weight_decay} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
      > ${output_dir}/config.txt

if [ ! -f ${output_dir}/log.out ];then
echo "The file doesn't exist."
else
rm -d ${output_dir}/log.out
fi

python s2s_glue.py \
      --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --task_name ${TASK_NAME} \
      --eval_dataset_name ${TASK_NAME} \
      --test_dataset_name ${TASK_NAME} \
      --dataset_config_name ${dataset_config_name} \
      --eval_dataset_config_name ${dataset_config_name} \
      --test_dataset_config_name ${dataset_config_name} \
      --predict_with_generate ${predict_with_generate} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --num_train_epochs ${num_train_epochs} \
      --experts_embedding ${experts_embedding} \
      --use_hypernet ${use_hypernet} \
      --process_dim ${process_dim} \
      --hypernet_input ${hypernet_input} \
      --hypernetwork_bottleneck ${hypernetwork_bottleneck} \
      --layer_emb_dim ${layer_emb_dim} \
      --adapter_dim ${adapter_dim} \
      --weight_decay ${weight_decay} --learning_rate ${learning_rate} \
      --overwrite_output_dir \
      --do_train ${do_train} \
      --do_eval ${do_eval} \
      --weight_decay ${weight_decay} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
      --max_source_length ${max_source_length} --compute_memory ${compute_memory}
