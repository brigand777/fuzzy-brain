import streamlit as st
import base64

def get_base64_of_file(file_path):
    """Reads a file and returns its base64 encoded string."""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def display_video(video_path, height=600):
    """
    Displays a video from a local file using HTML.
    
    Parameters:
      video_path (str): Path to the MP4 file.
      height (int): The height of the video display area.
    """
    video_base64 = get_base64_of_file(video_path)
    video_html = f"""
    <video width="100%" autoplay muted loop controls>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """
    st.components.v1.html(video_html, height=height)
