import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Llm_pipeline.pipeline import create_qa_chain, qa_bot_answer, load_llm, custom_prompt
from Topic_Router.topic_router import topic_to_response
import asyncio


# Initialize the QA chain and store it in Streamlit's session state
@st.cache_resource
def initialize_qa_chain():
    return create_qa_chain(load_llm=load_llm, custom_prompt=custom_prompt)

# Streamlit app
def main():
    st.title("CampusQuest Chatbot")
    st.write("Welcome! Ask me anything about Salem State University admissions.")

    # Initialize the QA chain and store it in session state
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = initialize_qa_chain()

    # User input
    user_input = st.text_input("Your Question:", key="user_input")

    if user_input:
        # Process the input and generate a response
        with st.spinner("Generating response..."):
            # Retrieve the QA chain from session state
            qa_chain = st.session_state.qa_chain

            # Call the chain asynchronously
            response = asyncio.run(qa_chain.ainvoke({"input": user_input}))

            # Use qa_bot_answer to handle the final response
            bot_response = asyncio.run(qa_bot_answer(user_input, qa_chain, response, None))

        # Display the response
        st.write("### Response:")
        st.write(bot_response["result"])

if __name__ == "__main__":
    main()