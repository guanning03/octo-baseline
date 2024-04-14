python ./examples/02_finetune_new_observation_action.py \
    --pretrained_path "/data1/zhuxiaopei/octo-small" \
    --data_dir "/data1/zhuxiaopei/aloha_sim_dataset/" \
    --save_dir "/data2/zhuxiaopei/runs-ft-aloha-octo/" \
    --batch_size 128 \
    --freeze_transformer False > example02.log