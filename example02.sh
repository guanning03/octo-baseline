python ./examples/02_finetune_new_observation_action.py \
    --pretrained_path "/root/autodl-tmp/octo-small" \
    --data_dir "/root/autodl-tmp/aloha_sim_dataset/" \
    --save_dir "runs/runs-ft-aloha-octo/" \
    --batch_size 128 \
    --freeze_transformer False > example03.log