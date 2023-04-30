import streamlit as st
import pythreejs as p3
import numpy as np


# Define a function to load the 3D model from a .npy file
def load_model(file):
    # Load the .npy file using numpy.load
    data = np.load(file)
    # Extract the vertices and faces from the data array
    vertices = data["vertices"]
    faces = data["faces"]
    return vertices, faces

# Create a file uploader using Streamlit's built-in function
uploaded_file = st.file_uploader("Upload a .npy file")

if uploaded_file is not None:
    # Call the load_model function to load the 3D model from the uploaded file
    vertices, faces = load_model(uploaded_file)

    # Use Pythreejs to create a 3D mesh object from the vertices and faces
    geometry = p3.BufferGeometry(
        attributes={
            "position": p3.BufferAttribute(vertices),
        },
        index=p3.BufferAttribute(faces.ravel()),
    )
    material = p3.MeshStandardMaterial(color="lightgray", metalness=0.5, roughness=0.5)
    mesh = p3.Mesh(geometry=geometry, material=material)

    # Use Streamlit to display the 3D mesh object
    st.write(p3.Renderer(camera="perspective", scene=p3.Scene(children=[mesh]), width=400, height=400))
