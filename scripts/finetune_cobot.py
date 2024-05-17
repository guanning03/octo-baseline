import datetime
from functools import partial
import imp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 3'
import json
import h5py
from absl import app, flags, logging
import flax
from flax.traverse_util import flatten_dict
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags, ConfigDict
import optax
import tensorflow as tf
import tqdm
import wandb
from octo.data.utils.format import standardize_pytree
import pdb
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.data.dataset import make_single_dataset
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_callbacks import (
    RolloutVisualizationCallback,
    SaveCallback,
    ValidationCallback,
    VisualizationCallback,
)
import itertools
from octo.utils.train_utils import (
    check_config_diff,
    create_optimizer,
    format_name_with_config,
    merge_params,
    process_text,
    Timer,
    TrainState,
)
from octo.model.components.action_heads import *
try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass
import random
from configs.finetune_config_cobot import load_rename_map
import io
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)") ### 默认设成False，便于wandb显示

flags.DEFINE_string("params_json_path", "./all_param.json", "Path to save all params' shape")

default_config_file = os.path.join(
    os.path.dirname(__file__), "configs/finetune_config_cobot.py"
)
config_flags.DEFINE_config_file(
    "config",
    default_config_file,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    logging.set_verbosity(logging.INFO) ### 可以调整日志输出的级别
    initialize_compilation_cache()
    devices = jax.devices()
    logging.info(
        f"""
        Octo Finetuning Script
        ======================
        Pretrained model: {FLAGS.config.pretrained_path}
        Finetuning Dataset: {FLAGS.config.dataset_kwargs.name_train}
        Data dir: {FLAGS.config.dataset_kwargs.data_dir}
        Task Modality: {FLAGS.config.modality}
        Finetuning Mode: {FLAGS.config.finetuning_mode}

        # Devices: {jax.device_count()}
        Batch size: {FLAGS.config.batch_size} ({FLAGS.config.batch_size // len(devices) } per device)
        # Steps: {FLAGS.config.num_steps}
    """
    )

    #########
    #
    # Setup Jax Data Parallelism
    #
    #########

    assert (
        FLAGS.config.batch_size % len(devices) == 0
    ), f"Batch size ({FLAGS.config.batch_size}) must be divisible by the number of devices ({len(devices)})"
    # assert (
    #     FLAGS.config.viz_kwargs.eval_batch_size % len(devices) == 0
    # ), f"Eval batch size ({FLAGS.config.viz_kwargs.eval_batch_size}) must be divisible by the number of devices ({len(devices)})"

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # Our batches will be data-parallel sharded -- each device will get a slice of the batch
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    # Our model will be replicated across devices (we are only doing data parallelism, not model parallelism)
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    #########
    #
    # Setup WandB
    #
    #########
    wandb.login(key = '256879fdda25bc1fb8ee4f0310e71615e92f75c9')
    name = format_name_with_config(
        FLAGS.name,
        FLAGS.config.to_dict(),
    )
    wandb_id = "{name}_{time}".format(
        name=name,
        time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    wandb.init(
        config=FLAGS.config.to_dict(),
        id=wandb_id,
        name=name,
        mode="disabled" if FLAGS.debug else None,
        **FLAGS.config.wandb,
    )

    #########
    #
    # Load Pretrained model + optionally modify config
    #
    #########

    pretrained_model = OctoModel.load_pretrained(
        FLAGS.config.pretrained_path,
        step=FLAGS.config.pretrained_step,
    )
    flat_config = flax.traverse_util.flatten_dict(
        pretrained_model.config, keep_empty_nodes=True
    )
    for d_key in flax.traverse_util.flatten_dict(
        FLAGS.config.get("config_delete_keys", ConfigDict()).to_dict()
    ):
        for c_key in list(flat_config.keys()):
            if ".".join(c_key).startswith(".".join(d_key)):
                del flat_config[c_key]

    ### FLAG.config是finetune的config
    ### 这个config是pretrained model的config
    config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
    config.update(FLAGS.config.get("update_config", ConfigDict()))
    config = config.to_dict()
    
    ### 检查模型config因为update_config和config_delete_keys的影响
    check_config_diff(config, pretrained_model.config)

    #########
    #
    # Setup Data Loader
    #
    #########

    # create text processor
    if config["text_processor"] is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.instantiate(config["text_processor"])()

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch
    
    # load standardize_fn from `path/to/file.py:fn_name` format
    ### 把standardize_fn函数解码出来，便于之后传参数给make_single_dataset
    if (
        standardize_fn := FLAGS.config["dataset_kwargs"].get("standardize_fn", None)
    ) is not None:
        path, name = standardize_fn.split(":")
        # imp is deprecated, but it's also what ml_collections uses
        standardize_fn = getattr(imp.load_source("standardize_fn", path), name)
        del FLAGS.config["dataset_kwargs"]["standardize_fn"]
        FLAGS.config["dataset_kwargs"]["standardize_fn"] = standardize_fn

    logging.info("Loading dataset, Please be patient...")
    
    def get_hdf5_files(directory):
        # 使用 os.walk 遍历目录和子目录
        hdf5_files = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(directory)
            for file in files if file.endswith('.hdf5')
        ]
        return hdf5_files
    
    dataset_list = get_hdf5_files(FLAGS.config.dataset_kwargs.data_dir)
    
    logging.info(f'There are totally {len(dataset_list)} files in the dataset.')
    
    class RandomFileIterator:
        def __init__(self, file_list, file_batch_size = 8):
            self.file_list = file_list
            self.file_batch_size = file_batch_size
            
        def __iter__(self):
            return self
        
        def __next__(self):
            if len(self.file_list) >= self.file_batch_size:
                return random.sample(self.file_list, self.file_batch_size)
            else:
                raise StopIteration
    
    file_iter = RandomFileIterator(dataset_list)
    
    def filebatch_to_databatch(file_batch, batch_size, text_tokenizer):  
        
        def pad_and_resize(image, target_size):
            original_size = image.size
            ratio = float(target_size) / max(original_size)
            new_size = tuple([int(x * ratio) for x in original_size])
            
            resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
            new_image = Image.new("RGB", (target_size, target_size))
            new_image.paste(resized_image, ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2))

            return new_image

        def bytes_image_to_jnp(image_bytes, image_size=128):
            image = Image.open(io.BytesIO(image_bytes))
            image = pad_and_resize(image, image_size)
            image_array = jnp.array(image)
            image_array = image_array[:,:,[2,1,0]]
            return image_array
    
        input_ids = []
        attention_mask = []
        primary = []
        wrist_left = []
        wrist_right = []
        action = []
        proprio = []
        timestep = []
        item_per_file = batch_size / len(file_batch)
        
        for filename in file_batch:
            
            file = h5py.File(filename, 'r')
            traj_len = file['action'].shape[0]
            text_token = text_tokenizer.encode([str(file['instruction'])])
            input_ids.extend([text_token['input_ids'] for _ in range(int(item_per_file))])
            attention_mask.extend([text_token['attention_mask'] for _ in range(int(item_per_file))])
            start_points = [random.randint(0, traj_len - 34) for _ in range(int(item_per_file))]
            timestep.append(start_points)
            
            for start_point in start_points:
                action.append(file['action'][start_point:start_point+34])
                proprio.append(file['observations']['qpos'][start_point:start_point+2])
                primary.append(file['observations']['images']['cam_high'][start_point:start_point+34])
                wrist_left.append(file['observations']['images']['cam_left_wrist'][start_point:start_point+34])
                wrist_right.append(file['observations']['images']['cam_right_wrist'][start_point:start_point+34])
                
        action = jnp.stack(action, axis=0)
        proprio = jnp.stack(proprio, axis=0)
        input_ids = jnp.stack(input_ids, axis=1).squeeze(0)
        attention_mask = jnp.stack(attention_mask, axis=1).squeeze(0)
        
        batch = {}
        batch['action'] = action
        batch['task'] = {}
        batch['task']['language_instruction'] = {}
        batch['task']['language_instruction']['input_ids'] = input_ids
        batch['task']['language_instruction']['attention_mask'] = attention_mask
        batch['observation'] = {}
        batch['observation']['proprio'] = proprio
        
        true_pad_mask = jnp.array([[True for _ in range(2)] for _ in range(batch_size)]).reshape((batch_size, 2))
        batch['task']['pad_mask_dict'] = {'language_instruction': jnp.array([True for _ in range(batch_size)])}
        timestep = jnp.array(timestep).reshape((batch_size, 1))
        increment = jnp.arange(2).reshape((1, 2))
        timestep = timestep + increment
        batch['observation']['timestep'] = timestep
        
        batch['observation']['pad_mask_dict'] = {
            'image_primary': true_pad_mask,
            'image_wrist_left': true_pad_mask,
            'image_wrist_right': true_pad_mask,
        }
        
        batch['observation']['pad_mask'] = true_pad_mask
        
        for i in range(len(primary)):
            primary[i] = jnp.stack([bytes_image_to_jnp(primary[i][j], image_size=256) for j in range(2)], axis=0)
            wrist_left[i] = jnp.stack([bytes_image_to_jnp(wrist_left[i][j], image_size=128) for j in range(2)], axis=0)
            wrist_right[i] = jnp.stack([bytes_image_to_jnp(wrist_right[i][j], image_size=128) for j in range(2)], axis=0)
            
        primary = jnp.stack(primary, axis=0)
        wrist_left = jnp.stack(wrist_left, axis=0)
        wrist_right = jnp.stack(wrist_right, axis=0)
        batch['observation']['image_primary'] = primary
        batch['observation']['image_wrist_left'] = wrist_left
        batch['observation']['image_wrist_right'] = wrist_right
        batch['absolute_action_mask'] = jnp.ones((batch_size, 14))
        
        return batch
    
    class DataLoader:
        def __init__(self, iterator, batch_size, text_tokenizer, prefetch_count=50):
            self.iterator = iterator
            self.batch_size = batch_size
            self.text_tokenizer = text_tokenizer
            self.prefetch_count = prefetch_count
            self.executor = ThreadPoolExecutor(max_workers=prefetch_count)
            self.futures = []
            self._preload_batches()
            
        def _preload_batches(self):
            while len(self.futures) < self.prefetch_count:
                try:
                    file_batch = next(self.iterator)
                    future = self.executor.submit(filebatch_to_databatch, file_batch, self.batch_size, self.text_tokenizer)
                    self.futures.append(future)
                except StopIteration:
                    break
            
        def __iter__(self):
            return self
        
        def __next__(self):
            if not self.futures:
                raise StopIteration
            
            future = self.futures.pop(0)
            self._preload_batches()
            return future.result()
    
    train_data_iter = DataLoader(file_iter, FLAGS.config.batch_size, text_processor)
                          
    # dataset = make_single_dataset(
    #     FLAGS.config.dataset_kwargs,
    #     traj_transform_kwargs=FLAGS.config.traj_transform_kwargs,
    #     frame_transform_kwargs=FLAGS.config.frame_transform_kwargs,
    #     train=True,
    # )
    
    # # 假设dataset已经定义并可以访问
    # def count_dataset_samples(dataset):
    #     # 这个函数将遍历数据集中的所有样本来计算总数
    #     return sum(1 for _ in dataset)

    # # 在您的数据迭代器之前计算样本数
    # total_samples = count_dataset_samples(dataset.unbatch())
    # logging.info(f'There are totally {total_samples} samples in the dataset.')
    
    # logging.info('Dataset loaded successfully. Start batching, please be patient ...')
    # train_data_iter = (
    #     dataset.repeat()
    #     .unbatch()
    #     .shuffle(FLAGS.config.shuffle_buffer_size)
    #     .batch(FLAGS.config.batch_size)
    #     .iterator()
    # )
    # train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)
    logging.info('Training data iteration loaded successfully')
    #########
    #
    # Load Pretrained Model
    #
    #########
    
    #TODO: 需要在这里把模型结构按照所提供的数据形式进行修改
    
    if FLAGS.config.change_model_config:
        config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
            LowdimObsTokenizer,
            n_bins=256,
            bin_type="normal",
            low=-2.0,
            high=2.0,
            obs_keys=["proprio"],
        )

        config["model"]["observation_tokenizers"]["wrist_left"] = config["model"]["observation_tokenizers"]["wrist"]
        config["model"]["observation_tokenizers"]["wrist_right"] = config["model"]["observation_tokenizers"]["wrist"]
        del config["model"]["observation_tokenizers"]["wrist"]
        
        config["model"]["heads"]['action'] = ModuleSpec.create(
            DiffusionActionHead,
            readout_key="readout_action",
            use_map = False,
            pred_horizon = FLAGS.config.traj_transform_kwargs["future_action_window_size"],
            action_dim = 14
        )
        
        # config['model']['heads']['action'] = ModuleSpec.create(
        #     MSEActionHead,
        #     pred_horizon=FLAGS.config.traj_transform_kwargs["future_action_window_size"],
        #     action_dim=14,
        #     readout_key="readout_action",
        # )
    
    print(f'model config: {json.dumps(config["model"], indent = 4)}')
    
    logging.info("Updating model for new observation & action spaces...")

    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, init_rng = jax.random.split(rng)
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        rng=init_rng,
        verbose = True,
        dataset_statistics=None,
    )
    
    if FLAGS.config.change_model_config:
        merged_params = merge_params(model.params, pretrained_model.params, load_rename_map(FLAGS.config.rename_map_path))
    else:
        merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model

    #########
    #
    # Setup Optimizer and Train State
    #
    #########

    logging.info('Setting optimizer and training state...')
    params = model.params
    if FLAGS.config.optimizer.frozen_keys is None:
        FLAGS.config.optimizer.frozen_keys = model.config["optimizer"]["frozen_keys"]

    with open(FLAGS.params_json_path, 'w') as f:
        f.write(standardize_pytree(params))
    
    tx, lr_callable, param_norm_callable = create_optimizer(
        params,
        **FLAGS.config.optimizer.to_dict(),
    )
    train_state = TrainState.create(
        model=model,
        tx=tx,
        rng=rng,
    )

    #########
    #
    # Save all metadata
    #
    #########

    if FLAGS.config.save_dir is not None:
        save_dir = tf.io.gfile.join(
            FLAGS.config.save_dir,
            FLAGS.config.wandb.project,
            FLAGS.config.wandb.group or "",
            wandb_id,
        )
        wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
        logging.info("Saving to %s", save_dir)
        save_callback = SaveCallback(save_dir)

        # Add window_size to top of config, to make eval easier
        new_config = ConfigDict(model.config)
        new_config["window_size"] = example_batch["observation"]["pad_mask"].shape[1]
        model = model.replace(config=new_config)

        # Save finetuning config since it's not saved by SaveCallback, i.e. as part of model.save_pretrained()
        with open(
            tf.io.gfile.join(save_dir, "finetune_config.json"), "w"
        ) as config_file:
            config_file.write(FLAGS.config.to_json_best_effort())
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        logging.warning("save_dir not passed in, not saving checkpoints")

    example_batch_spec = jax.tree_map(
        lambda arr: (arr.shape, str(arr.dtype)), example_batch
    )
    wandb.config.update(
        dict(example_batch_spec=example_batch_spec), allow_val_change=True
    )

    #########
    #
    # Define loss, train_step, and eval_step
    #
    #########

    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        
        ### TODO: 有时间从这里逆向回去看模型输入输出是否与预期相符
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # Action head knows to pull out the action readout_key
            batch["action"],
            pad_mask=batch["observation"]["pad_mask"],
            train=train,
        )
        return action_loss, action_metrics
    
    def real_loss_fn(params, batch, rng, train = False):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["pad_mask"],
            train=train,
        )
                
        norm_actions = bound_module.heads['action'].predict_action( # [batch, 32, 14]
            transformer_embeddings,
            train = False,
            rng = jax.random.PRNGKey(0)
        )
        
        #### FIXME: 下面有一些硬编码
        # action_mean = jnp.array(dataset_statistics['action']['mean'])
        # action_std = jnp.array(dataset_statistics['action']['std'])
        # mean_expanded = action_mean.reshape((1, 1, 14))
        # std_expanded = action_std.reshape((1, 1, 14))
        action_pred = norm_actions
        action_gt = batch['action'][:,2:, :]
        
        ### TODO: 将norm_actions重新变成原来的action, 然后和batch["action"]算loss
        return jnp.mean((action_pred - action_gt) ** 2), \
          {'real_mse': jnp.mean((action_pred - action_gt) **2)}

    # Data parallelism
    # Model is replicated across devices, data is split across devices
    @partial(
        jax.jit,
        in_shardings=[replicated_sharding, dp_sharding], 
        ### 数据的分片策略（replicated_sharding:模型参数复制，dp_sharding:训练数据分布式）
    )
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        # Gradient Metrics  ###
        grad_norm = optax.global_norm(grads)
        updates, _ = state.tx.update(grads, state.opt_state, state.model.params)
        update_norm = optax.global_norm(updates)
        info.update(
            {
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "param_norm": param_norm_callable(state.model.params),
                "learning_rate": lr_callable(state.step),
            }
        )
        # End Debug Metrics #

        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    #########
    #
    # Build validation & visualization callbacks
    #
    #########
    
    logging.info('Building validation and visualization callbacks...')

    if FLAGS.config.modality == "image_conditioned":
        modes_to_evaluate = ["image_conditioned"]
    elif FLAGS.config.modality == "text_conditioned":
        modes_to_evaluate = ["text_conditioned"]
    elif FLAGS.config.modality == "multimodal":
        modes_to_evaluate = ["image_conditioned", "text_conditioned"]
    else:
        modes_to_evaluate = ["base"]

    dataset_kwargs_list = [FLAGS.config.dataset_kwargs]

    # 直接关闭验证集
    val_callback = ValidationCallback(
        loss_fn=real_loss_fn, ### TODO: 这里的loss_fn需要改成real_loss_fn
        process_batch_fn=process_batch,
        text_processor=text_processor,
        val_dataset_kwargs_list=dataset_kwargs_list,
        dataset_kwargs=FLAGS.config,
        modes_to_evaluate=modes_to_evaluate,
        **FLAGS.config.val_kwargs,
    )

    # viz_callback = VisualizationCallback(
    #     text_processor=text_processor,
    #     val_dataset_kwargs_list=dataset_kwargs_list,
    #     dataset_kwargs=FLAGS.config,
    #     modes_to_evaluate=modes_to_evaluate,
    #     **FLAGS.config.viz_kwargs,
    # )

    #########
    #
    # Optionally build visualizers for sim env evals
    #
    #########

    # if "rollout_kwargs" in FLAGS.config:
    #     rollout_callback = RolloutVisualizationCallback(
    #         text_processor=text_processor,
    #         history_length=FLAGS.config["window_size"],
    #         model_pred_horizon=config["model"]["heads"]["action"]["kwargs"].get(
    #             "pred_horizon", 1
    #         ),
    #         **FLAGS.config.rollout_kwargs.to_dict(),
    #     )
    # else:
    #     rollout_callback = None

    #########
    #
    # Train loop
    #
    #########
    
    logging.info('Starting training!!!')

    def wandb_log(info, step):
        wandb.log(flatten_dict(info, sep="/"), step=step)

    timer = Timer()
    for i in tqdm.tqdm(
        range(0, int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps),
        dynamic_ncols=True,
    ):
        timer.tick("total")

        with timer("dataset"):
            batch = next(train_data_iter)

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()}, step=i
            )

        ### 直接关闭验证集
        if (i + 1) % FLAGS.config.eval_interval == 0 or i == 0:
            logging.info("Evaluating...")

            with timer("val"):
                val_metrics = val_callback(train_state, i + 1)
                wandb_log(val_metrics, step=i)

            # with timer("visualize"):
            #     viz_metrics = viz_callback(train_state, i + 1)
            #     wandb_log(viz_metrics, step=i)

            # if rollout_callback is not None:
            #     with timer("rollout"):
            #         rollout_metrics = rollout_callback(train_state, i + 1)
            #         wandb_log(rollout_metrics, step=i)

        if ((i + 1) % FLAGS.config.save_interval == 0 or i == 0) and save_dir is not None :
            logging.info("Saving checkpoint...")
            save_callback(train_state, i + 1)


if __name__ == "__main__":
    app.run(main)
