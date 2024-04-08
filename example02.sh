python ./examples/02_finetune_new_observation_action.py \
    --pretrained_path "./weights/octo-base/" \
    --data_dir "/root/autodl-fs/aloha_sim_dataset" \
    --save_dir "/root/autodl-fs/runs-ft-aloha/" \
    --batch_size 128 \
    --freeze_transformer False > example02.log