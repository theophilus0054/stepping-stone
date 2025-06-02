import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("Tabel Interaktif Dua Nilai per Sel")

# Ukuran tabel (ubah sesuai kebutuhan)
rows = 3
cols = 4

# Inisialisasi data dua dimensi (setiap sel = [sub_value, main_value])
data = [[{'sub': '', 'main': ''} for _ in range(cols)] for _ in range(rows)]

# Fungsi untuk menampilkan dan mengedit sel individual
def render_table(data):
    for i in range(rows):
        cols_list = st.columns(cols)
        for j in range(cols):
            with cols_list[j]:
                st.markdown(f"**Sel ({i+1},{j+1})**")
                sub_val = st.text_input(f"Sub-{i}-{j}", value=data[i][j]['sub'], label_visibility="collapsed")
                main_val = st.text_input(f"Main-{i}-{j}", value=data[i][j]['main'], label_visibility="collapsed")
                data[i][j]['sub'] = sub_val
                data[i][j]['main'] = main_val
                st.markdown(f"<div style='position: relative; height: 60px; border: 1px solid white; padding: 4px; background: black; color: white;'>"
                            f"<div style='position: absolute; top: 2px; left: 4px; font-size: 14px;'>{sub_val}</div>"
                            f"<div style='position: absolute; bottom: 2px; right: 4px; font-size: 16px;'>{main_val}</div>"
                            f"</div>", unsafe_allow_html=True)

render_table(data)
