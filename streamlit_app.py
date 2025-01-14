import streamlit as st
import os
import tempfile
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
#from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

class CloudRAGApp:
    def __init__(self):
        """Initialize the application state"""
        if 'chain' not in st.session_state:
            st.session_state.chain = None
        if 'pdf_loaded' not in st.session_state:
            st.session_state.pdf_loaded = False
        
        # Initialize the embeddings model once
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Use temporary directory for vector store
        self.temp_dir = tempfile.mkdtemp()
        self.index_directory = os.path.join(self.temp_dir, "vector_store")

    def load_pdf(self, pdf_content):
        """Load and process PDF content"""
        try:
            # Save uploaded content to temporary file
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_pdf.write(pdf_content)
            temp_pdf.close()

            # Process PDF
            loader = PyMuPDFLoader(temp_pdf.name)
            docs = loader.load()
            
            if not docs:
                st.error("No documents found in the PDF.")
                return None

            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,  # Reduced chunk size for better cloud performance
                chunk_overlap=100
            )
            texts = text_splitter.split_documents(docs)

            # Create vector store
            db = FAISS.from_documents(texts, self.embeddings)
            
            # Create retriever with search parameters
            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Define prompt template with improved context handling
            template = """Use the following pieces of context to answer the question. 
            If you cannot find the answer in the context, say "I cannot find the answer in the provided document."
            
            Context: {context}
            
            Question: {question}
            
            Answer: """
            
            prompt = ChatPromptTemplate.from_template(template)
            
            # Initialize model with specific parameters
            llm = Ollama(
                model="llama2",
                temperature=0.7,
                timeout=120  # Increased timeout for cloud environment
            )
            # llm = ChatOpenAI(
            #     api_key=os.getenv("OPENAI_API_KEY"),
            #     model_name="gpt-3.5-turbo",
            #     temperature=0.7
            #     )
            # Create chain with error handling
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # Cleanup temporary file
            os.unlink(temp_pdf.name)
            
            return chain

        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None

    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="RAG PDF Chatbot",
            page_icon="📄",
            layout="wide"
        )

        st.title("📄 RAG PDF Chatbot")
        
        # Sidebar for PDF upload
        with st.sidebar:
            st.header("📂 PDF Upload")
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type="pdf",
                help="Upload a PDF document to chat with"
            )

            if uploaded_file:
                with st.spinner("Processing PDF..."):
                    pdf_content = uploaded_file.getvalue()
                    st.session_state.chain = self.load_pdf(pdf_content)
                    if st.session_state.chain:
                        st.success("PDF processed successfully!")
                        st.session_state.pdf_loaded = True
                    
            # Add system information
            st.sidebar.markdown("---")
            st.sidebar.info("""
            💡 **Tips:**
            - Upload a PDF file to start
            - Ask specific questions
            - Questions are answered based on PDF content only
            """)

        # Main chat interface
        st.header("💬 Chat with Your PDF")
        
        if not st.session_state.pdf_loaded:
            st.info("👆 Please upload a PDF file to start chatting.")
        else:
            # Chat interface
            user_question = st.text_input(
                "Ask a question about your PDF:",
                placeholder="Enter your question here..."
            )

            if user_question:
                if st.session_state.chain:
                    with st.spinner("Generating response..."):
                        try:
                            result = st.session_state.chain.invoke(user_question)
                            st.write("🤖 Answer:")
                            st.markdown(result)
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
                            st.info("Please try rephrasing your question or uploading the PDF again.")

if __name__ == "__main__":
    app = CloudRAGApp()
    app.run()
 
