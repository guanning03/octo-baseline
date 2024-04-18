def standardize_fn(traj: dict) -> dict:
    traj['observation']['qpos'] = traj['qpos']
    traj['observation']['qvel'] = traj['qvel']
    del traj['qpos']
    del traj['qvel']
    return traj