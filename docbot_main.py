import streamlit as st
import tempfile
from threading import Thread
from queue import Queue
from docbot import DocBot
import tempfile

# Global variables for multithreading
input_queue = Queue()
output_queue = Queue()
docbot_thread = None

def streamlit_app_docbot(document, openai_api_key):
    query = st.text_input('What do you want to ask with the document?')
    if query:
        st.subheader('Showing query response')
        input_queue.put(query)

        # Get the response from the output queue (if available)
        while not output_queue.empty():
            response = output_queue.get()
            st.write(response)

def process_docbot():
    # Create an instance of DocBot
    bot = DocBot(data=temp_file.name)

    # Set the OpenAI API key

    bot.set_openai_api_key(openai_api_key)

    # Process queries continuously
    while True:
        query = input_queue.get()
        response = bot.process_query(query)
        output_queue.put(response)

def main():
    st.title('Document Bot')

    selected_app = st.selectbox('Select an app', ['DOCUMENT BOT', 'Other App'])

    if selected_app == "DOCUMENT BOT":
        st.title('ENTERED INTO DOCUMENT BOT')
        uploaded_file = st.file_uploader("Upload your Document", type=["pdf", "doc"])
        if uploaded_file:
            file_contents = uploaded_file.read()
            if file_contents is not None:
                openai_api_key = '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'  # Replace with your OpenAI API key

                # Create a temporary file to store the document content
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file_contents)
                    temp_file.seek(0)  # Reset the file position to the beginning

                    # Start the DocBot thread if not already running
                    global docbot_thread
                    if docbot_thread is None or not docbot_thread.is_alive():
                        docbot_thread = Thread(target=process_docbot)
                        docbot_thread.start()

                    streamlit_app_docbot(document=file_contents, openai_api_key=openai_api_key)
    else:
        st.write('Other App is selected')

if __name__ == "__main__":
    main()
