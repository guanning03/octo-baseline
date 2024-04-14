## octo预期的数据格式

### dataset

- 类别是`dlimp.dataset.DLataset`
- 是迭代器，可以`for ... in ...`遍历
- 其中每个元素是一个字典，包含了所有的数据

例如：
```json
{
    "absolute_action_mask": "Shape: (400, 14)",
    "action": "Shape: (400, 50, 14)",
    "dataset_name": "Shape: (400,)",
    "observation": {
        "image_primary": "Shape: (400, 1, 256, 256, 3)",
        "pad_mask": "Shape: (400, 1)",
        "pad_mask_dict": {
            "image_primary": "Shape: (400, 1)",
            "proprio": "Shape: (400, 1)",
            "timestep": "Shape: (400, 1)"
        },
        "proprio": "Shape: (400, 1, 14)",
        "timestep": "Shape: (400, 1)"
    },
    "task": {
        "language_instruction": "Shape: (400,)",
        "pad_mask_dict": {
            "language_instruction": "Shape: (400,)"
        }
    }
}
```