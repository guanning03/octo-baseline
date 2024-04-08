"""
Contains trajectory transforms used in the octo data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory
length).
"""
import logging

import tensorflow as tf


def chunk_act_obs(
    traj: dict,
    window_size: int,
    future_action_window_size: int = 0,
) -> dict:
    """Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
    observations from the past and the current observation. "action" is given a new axis (at index 1) of size
    `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current
    action, and `future_action_window_size` actions from the future. "pad_mask" is added to "observation" and
    indicates whether an observation should be considered padding (i.e. if it would have come from a timestep
    before the start of the trajectory).
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    
    ### 这里为observation和action添加了编号。
    ### observation的编号从 -window_size + 1 到 0，重复traj_len次;
    ### 然后observation += traj_len * tf.range(traj_len)[:, None]，表示了每个timestep的observation编号。意思是，随着timestep的增加，observation的编号也在增加。
    ### action的编号从 -window_size + 1 到 1 + future_action_window_size，重复traj_len次
    ### 同理加上时间戳编号
    chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1), [traj_len, window_size]
    ) + tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, window_size])

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(traj_len)[:, None],
        [traj_len, window_size + future_action_window_size],
    )

    ### floored_chunk_indices截取了大于等于0的chunk_indices
    ### 一开始只有1个observation，之后每个timestep会有一个新的observation
    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    ### 在timestep不存在的情况下, goal_timestep = traj_len - 1，重复traj_len次
    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len - 1)

    ### floored_action_chunk_indice截取了介于0和goal_timestep之间的action_chunk_indices
    floored_action_chunk_indices = tf.minimum(
        tf.maximum(action_chunk_indices, 0), goal_timestep[:, None]
    )

    traj["observation"] = tf.nest.map_structure(
        lambda x: tf.gather(x, floored_chunk_indices), traj["observation"]
    )
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # if no absolute_action_mask was provided, assume all actions are relative
    if "absolute_action_mask" not in traj and future_action_window_size > 0:
        logging.warning(
            "future_action_window_size > 0 but no absolute_action_mask was provided. "
            "Assuming all actions are relative for the purpose of making neutral actions."
        )
    absolute_action_mask = traj.get(
        "absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool)
    )
    neutral_actions = tf.where(
        absolute_action_mask[:, None, :],
        traj["action"],  # absolute actions are repeated (already done during chunking)
        tf.zeros_like(traj["action"]),  # relative actions are zeroed
    )

    # actions past the goal timestep become neutral
    
    ### GPT："neutral actions"：这可能是指不会改变环境状态的动作。在这段代码中，相对动作被置零以生成中性动作，这可能是因为在某些情况下，我们希望生成一个动作序列，但不希望这个动作序列对环境产生任何影响。例如，在训练一个强化学习模型时，我们可能希望生成一个动作序列来评估模型的性能，但不希望这个动作序列改变环境的状态。
    action_past_goal = action_chunk_indices > goal_timestep[:, None]
    traj["action"] = tf.where(
        action_past_goal[:, :, None], neutral_actions, traj["action"]
    )
    return traj


def subsample(traj: dict, subsample_length: int) -> dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)
    return traj


def add_pad_mask_dict(traj: dict) -> dict:
    """Adds a dictionary indicating which elements of the observation/task should be treated as padding.

    traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]
    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            ### 如果不为空字符串，则pad_mask_key为True，表示不是padding
            if traj[key][subkey].dtype == tf.string:
                # handles "language_instruction", "image_*", and "depth_*"
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0
            else:
                # all other keys should not be treated as padding
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)
        traj[key]["pad_mask_dict"] = pad_mask_dict
    return traj
