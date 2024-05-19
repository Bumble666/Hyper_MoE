do_train=True
do_eval=False
model_name_or_path=google/switch-base-8
per_device_train_batch_size=8
per_device_eval_batch_size=2
dataset_name=cnn_dailymail
num_train_epochs=10
weight_decay=0.1
learning_rate=3e-5

experts_embedding=True
use_hypernet=True

process_dim=128
hypernet_input=128
hypernetwork_bottleneck=128
layer_emb_dim=10
adapter_dim=64
experts_embedding_dim=128

dataloader_num_workers=16
evaluation_strategy=epoch
save_strategy=no
eval_accumulation_steps=10
predict_with_generate=False

source_prefix=summarize:
metric_for_best_model=rouge2
SAVE=./checkpoints_hmoe/${model_name_or_path##*/}/${dataset_name}/${learning_rate}

echo "${SAVE}"
mkdir -p ${SAVE}

echo  --model_name_or_path ${model_name_or_path} \
      --dataset_name ${dataset_name} \
      --do_train ${do_train} --do_eval ${do_eval} --overwrite_output_dir \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --output_dir ${SAVE} \
      --num_train_epochs ${num_train_epochs} --learning_rate ${learning_rate} \
      --weight_decay ${weight_decay} --metric_for_best_model ${metric_for_best_model} \
      --val_max_target_length 60 --max_eval_samples 1600 \
      --num_beams 6 --max_length 60 --min_length 10 --no_repeat_ngram_size 3 \
      --evaluation_strategy ${evaluation_strategy} --save_strategy ${save_strategy} \
      --eval_accumulation_steps ${eval_accumulation_steps}

if [ ! -f ${output_dir}/log.out ];then
echo "The file doesn't exist."
else
rm -d ${output_dir}/${log_out}
fi

python run_summarization.py \
      --model_name_or_path ${model_name_or_path} \
      --dataset_name ${dataset_name} \
      --do_train ${do_train} --do_eval ${do_eval} --overwrite_output_dir \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --output_dir ${SAVE} \
      --num_train_epochs ${num_train_epochs} --learning_rate ${learning_rate} \
      --weight_decay ${weight_decay} --metric_for_best_model ${metric_for_best_model} \
      --val_max_target_length 60 --max_eval_samples 1600 \
      --num_beams 6 --max_length 60 --min_length 10 --no_repeat_ngram_size 3 \
      --experts_embedding ${experts_embedding} \
      --use_hypernet ${use_hypernet} \
      --process_dim ${process_dim} \
      --hypernetwork_bottleneck ${hypernetwork_bottleneck} \
      --layer_emb_dim ${layer_emb_dim} \
      --adapter_dim ${adapter_dim} \
      --source_prefix ${source_prefix} \
      --experts_embedding_dim ${experts_embedding_dim} \
      --evaluation_strategy ${evaluation_strategy} --save_strategy ${save_strategy} \
      --eval_accumulation_steps ${eval_accumulation_steps}  --predict_with_generate ${predict_with_generate}
