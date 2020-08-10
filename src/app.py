import streamlit as st
import music_generator

st.title("Create Cool New Music")

if st.button("Get Music"):
    file_ = music_generator.get_music()
    audio_file = open(file_, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/ogg")
