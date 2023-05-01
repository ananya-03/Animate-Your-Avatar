import numpy as np
from brax import envs
from brax.io import html
import streamlit.components.v1 as components
import streamlit as st
import glob

from diffmimic.utils.io import deserialize_qp
from diffmimic.mimic_envs import register_mimic_env

register_mimic_env()

file_mapping = {
    'sit': 'data/demo_aist/sit.npy',
    'walk': 'data/demo_aist/walk.npy',
    'run': 'data/demo_aist/run.npy',
    'jump': 'data/demo_aist/jump.npy',
    'punch': 'data/demo_aist/punch.npy',
    'dance': 'data/demo_aist/dance.npy',
    'boxing': 'data/demo_aist/boxing.npy',
    'basketball': 'data/demo_amass/Basketball.npz.npy',
    'billiards': 'data/demo_amass/Billiards.npz.npy',
    'bmx': 'data/demo_amass/BMX.npz.npy',
    'boxing': 'data/demo_amass/Boxing.npz.npy',
    'dancing': 'data/demo_amass/Dancing.npz.npy',
    'hiking': 'data/demo_amass/Hiking.npz.npy',
    'horse_riding': 'data/demo_amass/HorseRiding.npz.npy',
    'parkour': 'data/demo_amass/Parkour.npz.npy',
    'polevault': 'data/demo_amass/PoleVault.npz.npy',
    'rowing': 'data/demo_amass/Rowing.npz.npy',
    'skiing': 'data/demo_amass/Skiing.npz.npy',
    'surfing': 'data/demo_amass/Surfing.npz.npy',
    'volleyball': 'data/demo_amass/Volleyball.npz.npy',
    'windsurfing': 'data/demo_amass/Windsurfing.npz.npy',
}

selected_file = None

text_input = st.text_input('Enter a motion', value='')
if text_input:
    if text_input not in file_mapping:
        st.warning('Motion not found')
    else:
        selected_file = file_mapping[text_input]
        st.success(f'Selected file: {selected_file}')

if selected_file:
    demo_traj = np.load(selected_file)

    if len(demo_traj.shape) == 3:
        demo_traj = demo_traj[:, 1]  # vis env 0

    init_qp = deserialize_qp(demo_traj[0])
    demo_qp = [deserialize_qp(demo_traj[i]) for i in range(demo_traj.shape[0])]

    env = envs.create(env_name='humanoid_mimic',
                      system_config='smpl',
                      reference_traj=demo_traj,
                      cycle_len = 100)
    components.html(html.render(env.sys, demo_qp), height=500)
