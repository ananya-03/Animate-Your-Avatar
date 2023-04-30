import streamlit as st
import numpy as np
import pybullet as p

# Load .npy file
file = 'example.npy'
data = np.load(file)

# Convert data to pybullet format
mesh = p.createCollisionShape(p.GEOM_MESH, vertices=data['vertices'], 
                               indices=data['indices'], meshScale=data['scale'])

# Export to FBX
output_file = 'output.fbx'
p.exportMesh(mesh, output_file, 'obj', meshScale=data['scale'], flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)

# Display output file
st.write('Output file:', output_file)
