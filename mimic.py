from brax import envs
import functools
import diffmimic.brax_lib.agent_apg as apg
import numpy as np
import jax.numpy as jnp
from absl import flags, app, logging
import yaml
import os

from diffmimic.utils import AttrDict
from diffmimic.mimic_envs import register_mimic_env
from brax.training.agents.apg import networks as apg_networks

register_mimic_env()

FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'configs/aist_example.yaml', help='Experiment configuration.')
flags.DEFINE_boolean('tfboard', True, help='use tfboard')


def main(argv):
    with open(FLAGS.config, 'r') as f:
        args = AttrDict(yaml.safe_load(f))

    logdir = "logs/exp"
    for k, v in args.items():
        if k == 'ref':
            logdir += f"_{v.split('/')[-1].split('.')[0]}"
        else:
            logdir += f"_{v}"

    if FLAGS.tfboard:
        import tensorflow as tf
        file_writer = tf.summary.create_file_writer(logdir)
        file_writer.set_as_default()

        def progress_fn(it, metrics):
            if it % 10 == 0:
                logging.info(os.path.abspath(logdir))
            num_steps = it * args.num_envs * (args.ep_len-1)
            for k in metrics:
                tf.summary.scalar(k, data=np.array(metrics[k]), step=num_steps)
    else:
        def progress_fn(it, metrics):
            if it % 10 == 0:
                logging.info(os.path.abspath(logdir))

    model_fn = functools.partial(apg_networks.make_apg_networks, hidden_layer_sizes=(512, 256))

    demo_traj = jnp.array(np.load(args.ref))
    demo_len = demo_traj.shape[0]
    args.cycle_len = min(args.cycle_len, demo_len)
    args.ep_len = min(args.ep_len, demo_len)
    args.ep_len_eval = min(args.ep_len_eval, demo_len)


    train_env = envs.get_environment(
        env_name="humanoid_mimic_train",
        system_config=args.system_config,
        reference_traj=demo_traj,
        cycle_len=args.cycle_len,
        early_termination=args.early_termination,
        demo_replay_mode=args.demo_replay_mode,
        err_threshold=args.threshold,
        replay_rate=args.replay_rate,
        reward_scaling=args.reward_scaling,
        vel_weight=args.vel_weight
    )

    eval_env = envs.get_environment(
        env_name="humanoid_mimic",
        system_config=args.system_config,
        reference_traj=demo_traj,
        cycle_len=args.cycle_len,
        vel_weight=args.vel_weight
    )

    make_inference_fn, params, _ = apg.train(
        seed=args.seed,
        environment=train_env,
        eval_environment=eval_env,
        episode_length=args.ep_len-1,
        eval_episode_length=args.ep_len_eval-1,
        num_envs=args.num_envs,
        num_eval_envs=args.num_eval_envs,
        learning_rate=args.lr,
        num_evals=args.max_it+1,
        max_gradient_norm=args.max_grad_norm,
        network_factory=model_fn,
        normalize_observations=args.normalize_observations,
        save_dir=logdir,
        progress_fn=progress_fn,
        use_linear_scheduler=args.use_lr_scheduler,
        truncation_length=args.truncation_length
    )


if __name__ == '__main__':
    app.run(main)
