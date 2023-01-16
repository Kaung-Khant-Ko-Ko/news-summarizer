import streamlit as st
import helper

st.set_page_config(page_title="News Summarizer", page_icon="📜", layout="wide")

hide_menu_style = """
    <style>
        #MainMenu {visibility : hidden}
        footer {visibility : hidden}
    </style>
"""

st.markdown(hide_menu_style, unsafe_allow_html=True)


def main():
    header = st.container()
    model_load = st.container()
    input_doc = st.container()
    predict_sum = st.container()
    output_sum = st.container()

    news, summary = "", ""

    with header:
        st.markdown(
            "<h1 style = 'text-align : center'> News Summarizer </h1>", unsafe_allow_html=True)

    with model_load:
        model, tokenizer = helper.load_model()
        st.write(type(model))

    with input_doc:
        st.write("📰 Input News to Summarize")
        news = st.text_area("📰 Input News to Summarize", label_visibility = "collapsed")

    with predict_sum:
        if news != "":
            summary = helper.predict_summary(model, tokenizer, news)

    with output_sum:
        if summary != "":
            st.markdown("---")
            st.write("📋 Summarized News")
            st.text_area("📋 Summarized News", value = summary, disabled=True, label_visibility = "collapsed")
#        st.write(summary)


if __name__ == "__main__":
    main()
