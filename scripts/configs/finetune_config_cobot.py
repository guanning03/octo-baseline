import os
os.environ['CURL_CA_BUNDLE'] = ''
from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
import os
from datetime import datetime

CURRENT_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def get_config(config_string="full,language_conditioned"):
    mode, task = config_string.split(",")
    
    ### task中是否包含文字/图片指令
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    
    ### 训练的参数包括哪个部分
    assert mode in ["full", "head_only", "head_mlp_only"]

    # Fill this in for your own dataset!

    # There should be two image keys
    # first image key should be the third-person view (None if not used)
    # and second image key should be the wrist view (None if not used)

    ### 这个就是传给 make_single_dataset 的第一个参数
    FINETUNING_KWARGS = {
        "name": "cobot_magic",
        "data_dir": "/root/autodl-tmp/",
        "train_ratio": 0.85,
        "image_obs_keys": {"primary": "cam_high", 
                           "wrist_left": "cam_left_wrist", 
                           "wrist_right": "cam_right_wrist"},
        "state_obs_keys": ["qpos", "qvel"],
        "language_key": "instruction",
        "action_proprio_normalization_type": "normal",
        # All actions are relative deltas, except for the last one (gripper) which is absolute
        # Specifying this is only necessary if you want to predict > 1 step into the future
        "absolute_action_mask": [True, True, True, True, True, True, True,
                                 True, True, True, True, True, True, True],
        # standardize_fn is dynamically loaded from a file
        # for example: "experiments/kevin/custom_standardization_transforms.py:aloha_dataset_transform"
        "standardize_fn": "octo/data/cobot/standardize.py:standardize_fn",
        # If the default data loading speed is too slow, try these:
        # "num_parallel_reads": 8,  # for reading from disk / GCS
        # "num_parallel_calls": 16,  # for initial dataset construction
        "action_normalization_mask":[True, True, True, True, True, True, False,
                                     True, True, True, True, True, True, False],
        "proprio_normalization_mask":[True, True, True, True, True, True, False,
                                      True, True, True, True, True, True, False,
                                      True, True, True, True, True, True, True,
                                      True, True, True, True, True, True, True]
    }

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    elif mode == "head_mlp_only":
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )
    elif mode == "frozen_transformer":
        frozen_keys = ("octo_transformer.BlockTransformer_0.*",)
    else:
        raise ValueError("Invalid mode")

    max_steps = FieldReference(50000)
    window_size = FieldReference(default=2)

    config = dict(
        pretrained_path='/root/autodl-tmp/octo-small',
        pretrained_step=270000,
        batch_size=16,
        shuffle_buffer_size=5000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=5000,
        save_interval=5000,
        save_dir=os.path.join('/root/autodl-tmp/'),
        seed=42,
        wandb=dict(
            project="octo_cobot", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=FINETUNING_KWARGS,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=2000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,  # if you are using grad accumulation, you need to adjust max_steps accordingly
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        viz_kwargs=dict(
            eval_batch_size=128,
            trajs_for_metrics=100,
            trajs_for_viz=8,
            samples_per_state=8,
        ),
    )

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    traj_transform_kwargs = dict(
        window_size=window_size,
        future_action_window_size=4,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
        # If the default data loading speed is too slow, try these:
        # num_parallel_calls=16,  # for less CPU-intensive ops
    )
    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    wrist_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (320, 240),  # workspace (3rd person) camera is at 256x256
            "wrist_left": (320, 240),  # wrist camera is at 128x128
            "wrist_right": (320, 240)
        },
        image_augment_kwargs=[
            workspace_augment_kwargs,
            wrist_augment_kwargs,
        ],
    )
    # If the default data loading speed is too slow, try these:
    config[
        "frame_transform_threads"
    ] = 16  # for the most CPU-intensive ops (decoding, resizing, augmenting)

    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs
    return ConfigDict(config)
