cd /home/guizili/yitingw1/models/DeepLearningExamples/TensorFlow2/LanguageModeling/BERT
bash scripts/docker/launch.sh

# process data
create_pretraining_data.py

# pretrain LAMB是分布式训练的，单卡用Adam
bash scripts/run_pretraining_lamb.sh <train_batch_size_phase1> <train_batch_size_phase2> <eval_batch_size> <learning_rate_phase1> <learning_rate_phase2> <precision> <use_xla> <num_gpus> <warmup_steps_phase1> <warmup_steps_phase2> <train_steps> <save_checkpoint_steps> <num_accumulation_phase1> <num_accumulation_steps_phase2> <bert_model>
bash scripts/run_pretraining_lamb.sh 60 10 8 7.5e-4 5e-4 fp16 false 1 2000 200 7820 100 64 192 large |& tee run1.log # XLA will get wrong
bash scripts/run_pretraining_lamb.sh $(source scripts/configs/pretrain_config.sh && dgxa100_1gpu_fp16) |& tee run.log #same with below  default: Large
# bash scripts/run_pretraining_lamb.sh 312 40 8 8.12e-4 5e-4 fp16 false 1 2000 200 6416 100 32 96  # OOM!!!
# phase1
+ python /workspace/bert_tf2/run_pretraining.py '--input_files=data/tfrecord/lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/wikicorpus_en/training/*' --model_dir=/results/tf_bert_pretraining_lamb_large_fp16_gbs139936_gbs215360_230223083416/phase_1 --bert_config_file=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_config.json --train_batch_size=312 --max_seq_length=128 --max_predictions_per_seq=20 --num_steps_per_epoch=5774 --num_train_epochs=1 --steps_per_loop=96 --save_checkpoint_steps=96 --warmup_steps=2000 --num_accumulation_steps=128 --learning_rate=8.120000e-04 --optimizer_type=LAMB --use_fp16 --enable_xla
# phase2
+ python /workspace/bert_tf2/run_pretraining.py '--input_files=data/tfrecord/lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/wikicorpus_en/training/*' --init_checkpoint=/results/tf_bert_pretraining_lamb_large_fp16_gbs19984_gbs23840_230224062437/phase_1/pretrained/bert_model.ckpt-1 --model_dir=/results/tf_bert_pretraining_lamb_large_fp16_gbs19984_gbs23840_230224062437/phase_2 --bert_config_file=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_config.json --train_batch_size=40 --max_seq_length=512 --max_predictions_per_seq=80 --num_steps_per_epoch=1666 --num_train_epochs=1 --steps_per_loop=100 --save_checkpoint_steps=100 --warmup_steps=200 --num_accumulation_steps=96 --learning_rate=5.000000e-04 --optimizer_type=LAMB --use_fp16 --enable_xla

# the first: bash scripts/run_pretraining_lamb.sh 60 10 8 7.5e-4 5e-4 fp16 false 1 2000 200 7820 100 64 192 large |& tee run1.log
bash scripts/run_pretraining_lamb.sh 60 10 8 7.5e-4 5e-4 fp16 false 1 660 66 2600 100 64 192 large |& tee -a pretrain_lamb0224.log
# phase1
+ python /workspace/bert_tf2/run_pretraining.py '--input_files=data/tfrecord/lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/wikicorpus_en/training/*' --model_dir=/results/tf_bert_pretraining_lamb_large_fp16_gbs13840_gbs21920_230223104202/phase_1 --bert_config_file=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_config.json --train_batch_size=60 --max_seq_length=128 --max_predictions_per_seq=20 --num_steps_per_epoch=7038 --num_train_epochs=1 --steps_per_loop=100 --save_checkpoint_steps=100 --warmup_steps=2000 --num_accumulation_steps=64 --learning_rate=7.500000e-04 --optimizer_type=LAMB --use_fp16
# phase2
+ python /workspace/bert_tf2/run_pretraining.py '--input_files=data/tfrecord/lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/wikicorpus_en/training/*' --init_checkpoint=/results/tf_bert_pretraining_lamb_large_fp16_gbs13840_gbs21920_230224061736/phase_1/pretrained/bert_model.ckpt-1 --model_dir=/results/tf_bert_pretraining_lamb_large_fp16_gbs13840_gbs21920_230224061736/phase_2 --bert_config_file=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_config.json --train_batch_size=10 --max_seq_length=512 --max_predictions_per_seq=80 --num_steps_per_epoch=1564 --num_train_epochs=1 --steps_per_loop=100 --save_checkpoint_steps=100 --warmup_steps=200 --num_accumulation_steps=192 --learning_rate=5.000000e-04 --optimizer_type=LAMB --use_fp16


nohup bash scripts/run_pretraining_adam.sh |& tee pretrain_adam.log
bash scripts/run_pretraining_adam.sh <train_batch_size_per_gpu> <eval_batch_size> <learning_rate_per_gpu> fp16 false 1 <warmup_steps> <train_steps> 100


# fine-tune   run successfully 
bash scripts/run_squad.sh <num_gpus> <batch_size_per_gpu> <learning_rate_per_gpu> <precision> <use_xla> <bert_model> <squad_version> <epochs>
bash scripts/run_squad.sh 1 12 5e-6 fp16 true large 1.1 2 true |& tee finetune.log
# inference
bash scripts/finetune_inference_benchmark.sh large 1 fp16 |& tee inference.log