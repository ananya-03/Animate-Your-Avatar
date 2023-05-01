import numpy as np
from brax import envs
from brax.io import html
import streamlit.components.v1 as components
import streamlit as st

from diffmimic.utils.io import deserialize_qp
from diffmimic.mimic_envs import register_mimic_env

register_mimic_env()

# Ask for the path of the reference motion file
ref_motion_path = st.text_input('Enter path to reference motion file:')
try:
    demo_traj = np.load(ref_motion_path)
except FileNotFoundError:
    st.warning('Reference motion file not found.')

# Load reference motion file and extract the initial QP state and demo QP states
if len(demo_traj.shape) == 3:
    demo_traj = demo_traj[:, 1]  # vis env 0
init_qp = deserialize_qp(demo_traj[0])
demo_qp = [deserialize_qp(demo_traj[i]) for i in range(demo_traj.shape[0])]

# Create environment and render HTML visualization
env = envs.create(env_name='humanoid_mimic',
                  system_config='smpl',
                  reference_traj=demo_traj,
                  )
components.html(html.render(env.sys, demo_qp), height=500)
