timestamp=$(date +%Y%m%d_%H%M%S)
python ./scripts/finetune_cobot.py \
    --name cobot_long \
    --params_json_path ./logs/all-params-$timestamp.json \
    --debug True > ./logs/stdout-$timestamp.ans \
    2> ./logs/errout-$timestamp.err
