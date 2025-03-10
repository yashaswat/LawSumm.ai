import os
from io import StringIO
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import streamlit as st 
from streamlit_extras.customize_running import center_running
from streamlit_lottie import st_lottie
from streamlit_file_browser import st_file_browser

import spacy
from spacy_streamlit import visualize_ner

@st.cache_data
def model_init():
    model = AutoModelForSeq2SeqLM.from_pretrained('C:/LawSumm.ai/nlp-indian-legal-led-base', low_cpu_mem_usage = True)
    tokenizer = AutoTokenizer.from_pretrained('C:/LawSumm.ai/nlp-indian-legal-tokenizer', low_cpu_mem_usage = True)
    return model, tokenizer

@st.cache_data
def model_summarization(text):
    model, tokenizer = model_init()
    
    input_tokenized = tokenizer.encode(string_data, return_tensors='pt',
                                       padding='max_length',pad_to_max_length=True, 
                                       max_length=16384,truncation=True)
        
    summary_ids = model.generate(input_tokenized,
                                  num_beams=4,
                                  no_repeat_ngram_size=3,
                                  length_penalty=2,
                                  min_new_tokens=250,
                                  max_new_tokens=1000)
    
    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]

    return summary

col1, col2 = st.columns([0.25, 0.75])

with col1:
    st.lottie("https://lottie.host/f784db1c-ad9c-44b6-98c6-b5fe973801d1/QYy3yuWV5y.json", 
                     loop=True, height=150, width=150, quality='medium')
with col2:
    st.title('LawSumm: Indian Legal Text Summarizer')
    
st.sidebar.header('Past Summaries')

directory = 'prev_summaries'
files = os.listdir(directory)

for file in files:
    
    dropdown = st.sidebar.expander('Case: {0}'.format(file))

    with open(os.path.join(directory, file), "r") as f:
        file_text = f.read()
    
    dropdown.write(file_text)
    
    dropdown.download_button('Download Case: {0}'.format(file), 
                       data = file_text,
                       file_name='{0}'.format(file))

st.divider()
st.write('''**LawSumm.ai** is an AI-powered Natural Language Processing application that can automatically summarize any Indian 
            legal document, such as court judgments, statutes, contracts, and more. It uses state-of-the-art deep learning 
            models that are trained on a large corpus of Indian legal text, collected from various sources and domains.''')

st.write('''Our LLM model can extract the most relevant and important information from any legal document, such as the facts, arguments,
            reasoning, cited laws, cited judgments, and outcome. it can also generate concise and coherent summaries that capture the 
            essence and context of the legal document.''')
            
st.write('''This app is designed to help lawyers, judges, researchers, students, and anyone interested in Indian law to access and 
            comprehend legal information faster and easier.''')
            
st.write('''**LawSumm.ai** is the ultimate AI legal document summarizer for India. Try it now and see for yourself!''')                 
st.divider()

input_method = st.segmented_control('CHOOSE YOUR INPUT METHOD:', options=['File', 'Text'])
st.write('\n')
got_input = False

file, text_area = None, None
string_data = ''

if input_method == 'File':
    
    file = st.file_uploader('Upload your legal document: ', accept_multiple_files=False, type='txt')
    if file:
        got_input = True

elif input_method == 'Text':
    
    query_title = st.text_input('Enter a title for your query: ',
                                placeholder='This title will be used to save summary in the History Sidebar')
    text_area = st.text_area('Enter your legal text below: ', placeholder='Type or paste text here...', height=680)
    
    col1, col2 = st.columns([0.75, 0.25])
    if col2.button('Submit data', use_container_width=True):
        got_input = True

if got_input:
    
    if file is not None:
        bytes_data = file.getvalue()
        string_data = StringIO(bytes_data.decode("utf-8")).read()
        file_name = str(file.name).replace('.txt', '')
        
    elif text_area is not None:
        string_data = text_area
        file_name = query_title
    
    st.markdown('### Your Summarized Case :pencil: \n')
    
    summary = model_summarization(string_data)
    
    with open(os.path.join(directory, "{0}_summary.txt".format(file_name)), "w") as f:
        f.write(summary)

    # tab1, tab2 = st.tabs([' Summary', ':book: Original Document'])
    
    # tab1.write('\n')
    # tab1.write(summary)
    # tab2.write('\n')
    # tab2.write(string_data)
    
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(summary)
    visualize_ner(doc, labels=nlp.get_pipe("ner").labels, show_table=False, title=False)
        
    doc_tokens = len(string_data.split())
    summary_tokens = len(summary.split())
    
    token_change = doc_tokens - summary_tokens
    change = -(token_change/doc_tokens)*100
    change = round(change, 2)
    
    st.write('\n')
    col1, col2 = st.columns(2)
    col1.metric('Words in Original Document', doc_tokens)
    col2.metric('Words in Summary', summary_tokens, delta = '{0}%'.format(change))
        
    st.download_button('Download your summarized case file',
                    data=summary, 
                    file_name='{0}_summary.txt'.format(file_name))
