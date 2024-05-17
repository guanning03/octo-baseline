timestamp=$(date +%Y%m%d_%H%M%S)
python ./scripts/finetune_cobot.py \
    --name mix \
    --params_json_path ./logs/all-params-$timestamp.json \
    > ./logs/stdout-$timestamp.ans \
    2> ./logs/errout-$timestamp.err
