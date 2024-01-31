import streamlit as st
from main import user_input
from Utiles import utils
from main import user_input

st.set_page_config('chat with muliple pdf')

#title of page
st.header('Multiple PDF Data Extraction')

##input quation
user_quation=st.text_input('Ask your quation for file')

output=st.button('submit')
if output:
    responce=user_input(user_quation)
    st.write(responce)

### for sidebar
with st.sidebar:
    st.title('upload file:')
    pdf_docs=st.file_uploader('Upload your PDF and click on submit and progcess',type=['pdf'])
    if st.button('Submit and Process'):
        with st.spinner('Processing....'):
            raw_text=utils.get_pdf_text(pdf_docs)
            text_chunks=utils.get_text_chunks(raw_text)
            utils.get_vector_store(text_chunks)
            st.success('Done')
