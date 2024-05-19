do_train=True
model_name_or_path=gpt2
dataset_name=wikitext
dataset_config_name=wikitext-2-raw-v1

per_device_train_batch_size=4
per_device_eval_batch_size=4

experts_embedding=True
use_hypernet=True

process_dim=128
hypernet_input=128
hypernetwork_bottleneck=128
layer_emb_dim=10
adapter_dim=64
experts_embedding_dim=128

use_moe=MoE

num_train_epochs=20
dataloader_num_workers=16
evaluation_strategy=epoch
save_strategy=no
k=1
n_experts=8
seed=42

log_out=log.out
learning_rate=1e-5

output_dir=./checkpoints_hmoe/${model_name_or_path##*/}/${dataset_name}/${use_moe}/${n_experts}_${k}_${learning_rate}

echo "${output_dir}"
mkdir -p ${output_dir}

echo  --use_moe ${use_moe} \
      --k ${k} \
      --n_experts ${n_experts} \
      --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --dataset_name ${dataset_name} \
      --dataset_config_name ${dataset_config_name} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --num_train_epochs ${num_train_epochs} \
      --overwrite_output_dir \
      --do_train ${do_train} \
      --do_eval \
      --seed ${seed} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
      --load_best_model_at_end True \
      --learning_rate ${learning_rate} \
      > ${output_dir}/config.txt


if [ ! -f ${output_dir}/log.out ];then
echo "The file doesn't exist."
else
rm -d ${output_dir}/${log_out}
fi

python run_clm.py \
      --use_moe ${use_moe} \
      --k ${k} \
      --n_experts ${n_experts} \
      --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --dataset_name ${dataset_name} \
      --dataset_config_name ${dataset_config_name} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --num_train_epochs ${num_train_epochs} \
      --overwrite_output_dir \
      --do_train ${do_train} \
      --do_eval \
      --seed ${seed} \
      --hypernet_input ${hypernet_input} \
      --experts_embedding ${experts_embedding} \
      --use_hypernet ${use_hypernet} \
      --process_dim ${process_dim} \
      --hypernetwork_bottleneck ${hypernetwork_bottleneck} \
      --layer_emb_dim ${layer_emb_dim} \
      --adapter_dim ${adapter_dim} \
      --experts_embedding_dim ${experts_embedding_dim} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
      --learning_rate ${learning_rate} \


      
