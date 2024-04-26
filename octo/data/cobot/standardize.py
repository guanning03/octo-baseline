from octo.data.utils.format import channel_transform

def standardize_fn(traj: dict) -> dict:
    traj['observation']['qpos'] = traj['qpos']
    traj['observation']['qvel'] = traj['qvel']
    del traj['qpos']
    del traj['qvel']
    for key in traj['observation']:
        if key.startswith('image_'):
            for i in range(len(traj['observation'][key])):
                traj['observation'][key][i] = channel_transform(traj['observation'][key][i])
    return traj