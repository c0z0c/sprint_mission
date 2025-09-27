from time import sleep
import streamlit as st

st.title("간단한 앱")
if st.button("클릭"):
    st.write("버튼 눌림")
    sleep(0.5)
    st.success("완료")