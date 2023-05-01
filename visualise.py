import numpy as np
from brax import envs
from brax.io import html
import streamlit.components.v1 as components
import streamlit as st
import glob

from diffmimic.utils.io import deserialize_qp
from diffmimic.mimic_envs import register_mimic_env

register_mimic_env()

traj_dir = st.text_input('Motion directory')
fname_dict = {fname.split('/')[-1]: fname for fname in glob.glob("{}/*.npy".format(traj_dir))}
ref_motion = st.text_input('Reference motion', list(fname_dict.keys())[0])

demo_traj = np.load(fname_dict[ref_motion])

if len(demo_traj.shape) == 3:
    demo_traj = demo_traj[:, 1]  # vis env 0

init_qp = deserialize_qp(demo_traj[0])
demo_qp = [deserialize_qp(demo_traj[i]) for i in range(demo_traj.shape[0])]

env = envs.create(env_name='humanoid_mimic',
                  system_config='smpl',
                  reference_traj=demo_traj,
                  )
components.html(html.render(env.sys, demo_qp), height=500)
