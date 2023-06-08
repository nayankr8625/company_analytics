from langchain.embeddings.openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from services import logger, measure_time


class DocBot:

    def __init__(self,query,data):

        self._query = query
        self._data = data

    @measure_time
    def processor(self,openai_api_key):
        logger.debug('Document loaded Processing the document')
        reader = PdfReader(self._data)
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        logger.debug('Spliting Documents into chunks')
        text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len
        )
        texts = text_splitter.split_text(raw_text)
        logger.debug('Generating the Embeddings')
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        docsearch = FAISS.from_texts(texts, embeddings)
        chain = load_qa_chain(OpenAI(openai_api_key=openai_api_key),
                                    chain_type="stuff",
                                    )
        docs = docsearch.similarity_search(self._query)
        response = chain.run(input_documents=docs, question=self._query)

        return response
    

if __name__ == '__main__':

    ob = DocBot(query=input('Enter your query:- '), data ='lt-csr-annual-report-2020-21.pdf' )
    print(ob.processor(openai_api_key='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'))
        