import streamlit as st
from Projects.VisaChatbot.git_project.LG2 import app


st.set_page_config(page_title="Visa Assistant", layout="centered")
st.title("🛂 Visa Assistant")

query = st.text_input("Ask me a visa-related question:")

from langchain_core.messages import HumanMessage, AIMessage

if st.button("Submit") and query:
    with st.spinner("Thinking..."):
        try:
            # Call your agent
            response = app.invoke({
        "messages": [
        AIMessage(content="You are a visa assistant that checks credible government sites."),
        HumanMessage(content=query)
        ]
        })
            messages = response.get("messages", [])
            human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
            if human_messages:
                answer = human_messages[-1].content  # get the last HumanMessage content
            else:
                answer = "No answer found."
            st.success("Answer:")
            paragraph_text = answer.replace('\n', '\n\n')
            st.markdown(paragraph_text)
        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
