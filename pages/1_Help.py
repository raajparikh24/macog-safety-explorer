import streamlit as st
from pathlib import Path

st.set_page_config(page_title='Filter guide', layout='wide')
st.title('Filter guide')

doc_path = Path(__file__).parent.parent / 'FILTERS.md'
if doc_path.exists():
    st.markdown(doc_path.read_text(encoding='utf-8'))
else:
    st.error(f'FILTERS.md not found at {doc_path}')
