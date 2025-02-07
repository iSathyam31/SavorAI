import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from typing import List
import os

# Initialize session state
def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = None
    if 'diet_preference' not in st.session_state:
        st.session_state['diet_preference'] = 'All'
    if 'meal_time' not in st.session_state:
        st.session_state['meal_time'] = 'Any'

# Constants and configurations
PDF_STORAGE_PATH = 'Data/'
EMBEDDING_MODEL = OllamaEmbeddings(model="llama3.1")
LANGUAGE_MODEL = OllamaLLM(model="llama3.1")

MENU_RELATED_KEYWORDS = [
    'menu', 'food', 'dish', 'meal', 'eat', 'drink', 'recommend', 'suggestion',
    'order', 'specialty', 'cuisine', 'appetite', 'hungry', 'restaurant',
    'vegetarian', 'vegan', 'spicy', 'dessert', 'appetizer', 'main course',
    'dinner', 'lunch', 'breakfast', 'serve', 'portion', 'price', 'cost'
]

MENU_ANALYSIS_PROMPT = """
You are an expert restaurant assistant with deep knowledge of cuisine and dietary preferences. 
Use the provided menu context to help customers make informed dining choices.

Current user preferences:
🍽️ Dietary preference: {diet_preference}
⏰ Meal time: {meal_time}

Previous conversation:
{chat_history}

Current menu context: {context}

User query: {question}

Please provide recommendations based on the following guidelines:
1. Consider the user's dietary preferences (vegetarian/non-vegetarian)
2. Focus on menu items appropriate for the selected meal time
3. Suggest complementary dishes when appropriate
4. Highlight special or popular items from the menu
5. Explain key ingredients or preparation methods if relevant
6. Consider portion sizes and meal combinations

Response:
"""

CASUAL_CHAT_PROMPT = """
You are a friendly restaurant assistant engaging in casual conversation. 
Respond naturally to the user's message without analyzing the menu.

Current user preferences:
🍽️ Dietary preference: {diet_preference}
⏰ Meal time: {meal_time}

Previous conversation:
{chat_history}

User message: {question}

Respond in a friendly, conversational manner while maintaining context of the previous discussion.
"""

# File handling functions
def save_uploaded_files(uploaded_files) -> List[str]:
    """Save multiple uploaded files and return their paths"""
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    return file_paths

def load_multiple_pdfs(file_paths: List[str]):
    """Load multiple PDF documents"""
    documents = []
    for file_path in file_paths:
        document_loader = PDFPlumberLoader(file_path)
        documents.extend(document_loader.load())
    return documents

def chunk_documents(raw_documents):
    """Split documents into chunks"""
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

# Vector store operations
def index_documents(document_chunks):
    """Create vector store from document chunks"""
    vector_store = InMemoryVectorStore(embedding=EMBEDDING_MODEL)
    vector_store.add_documents(document_chunks)
    st.session_state['vector_store'] = vector_store

def find_related_documents(query: str):
    """Find relevant documents for a query"""
    if st.session_state['vector_store'] is None:
        return []
    return st.session_state['vector_store'].similarity_search(query)

# Text processing functions
def combine_documents(docs: list) -> str:
    """Combine multiple documents into a single string"""
    return "\n\n".join([doc.page_content for doc in docs])

def format_chat_history(messages: List[dict]) -> str:
    """Format chat history for prompt context"""
    formatted_history = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted_history.append(f"{role}: {content}")
    return "\n".join(formatted_history[-6:])

def should_analyze_menu(query: str) -> bool:
    """Determine if the query requires menu analysis"""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in MENU_RELATED_KEYWORDS)

# Response generation functions
def generate_casual_response(user_query: str) -> str:
    """Generate a casual response without menu analysis"""
    chat_history = format_chat_history(st.session_state['messages'])
    prompt = ChatPromptTemplate.from_template(CASUAL_CHAT_PROMPT)
    chain = prompt | LANGUAGE_MODEL
    
    return chain.invoke({
        "chat_history": chat_history,
        "question": user_query,
        "diet_preference": st.session_state['diet_preference'],
        "meal_time": st.session_state['meal_time']
    })

def generate_menu_response(user_query: str, context_documents: list) -> str:
    """Generate a response with menu analysis"""
    chat_history = format_chat_history(st.session_state['messages'])
    prompt = ChatPromptTemplate.from_template(MENU_ANALYSIS_PROMPT)
    chain = prompt | LANGUAGE_MODEL
    
    return chain.invoke({
        "chat_history": chat_history,
        "context": combine_documents(context_documents),
        "question": user_query,
        "diet_preference": st.session_state['diet_preference'],
        "meal_time": st.session_state['meal_time']
    })

def generate_answer(user_query: str, context_documents: list) -> str:
    """Generate appropriate response based on query type"""
    if should_analyze_menu(user_query):
        return generate_menu_response(user_query, context_documents)
    else:
        return generate_casual_response(user_query)

# Main UI function
def main():
    # Initialize session state
    init_session_state()
    
    # UI Configuration
    st.set_page_config(
        page_title="Smart Menu Assistant",
        page_icon="🍽️",
        layout="wide"
    )
    
    # Main title with emojis
    st.title("🍽️ SavorAI 🤖")
    st.markdown("### 👨‍🍳 Your Personal Restaurant Guide 🌟")
    st.markdown("---")
    
    # Sidebar with enhanced emojis
    with st.sidebar:
        st.markdown("### 🎯 Your Dining Journey")
        
        # Welcome message
        st.markdown("""
        👋 Welcome, food lover! 
        
        Let me help you discover 
        the perfect dining options! 🌟
        
        Use these filters to create your 
        perfect dining experience! ✨
        """)
        
        # Diet preference section
        st.markdown("#### 🥗 Dietary Preference")
        diet_choice = st.radio(
            "What's your food preference? 🍴",
            ["All 🍽️", "Vegetarian 🥬", "Non-Vegetarian 🍗"],
            help="Choose your dietary preference to get personalized recommendations",
            key="diet_radio"
        )
        st.session_state['diet_preference'] = diet_choice.split()[0]  # Remove emoji from choice
        
        # Meal time section
        st.markdown("#### ⏰ Meal Time")
        meal_time = st.selectbox(
            "When are you planning to eat? 🕐",
            ["Any ⏰", "Breakfast 🌅", "Lunch 🌞", "Dinner 🌙"],
            help="Select a meal time for specific menu suggestions",
            key="meal_select"
        )
        st.session_state['meal_time'] = meal_time.split()[0]  # Remove emoji from choice
        
        # Helpful tips
        st.markdown("---")
        st.markdown("""
        💡 **Pro Tips:**
        
        1. 📑 Upload multiple menus to compare options
        2. 🔍 Use filters to find perfect matches
        3. 💬 Ask specific questions about dishes
        4. ⭐ Look for chef's recommendations
        """)
        
        # Clear chat button at bottom
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", help="Remove all previous messages"):
            st.session_state['messages'] = []
            st.session_state['vector_store'] = None
            st.rerun()
    
    # File Upload Section with emojis
    uploaded_pdfs = st.file_uploader(
        "📤 Upload Restaurant Menus (PDF)",
        type="pdf",
        help="Select one or more restaurant menus in PDF format",
        accept_multiple_files=True
    )

    if uploaded_pdfs:
        try:
            with st.spinner("🔄 Processing your menus..."):
                saved_paths = save_uploaded_files(uploaded_pdfs)
                raw_docs = load_multiple_pdfs(saved_paths)
                processed_chunks = chunk_documents(raw_docs)
                index_documents(processed_chunks)
                st.success(f"✅ Successfully processed {len(uploaded_pdfs)} menu(s)! 🎉\n\n💭 Ask me anything about the menu!")
        except Exception as e:
            st.error(f"❌ Error processing menus: {str(e)}")

    # Add a separator before chat
    st.markdown("---")
    st.markdown("### 💬 Chat with Your Menu Assistant")

    # Display chat history with enhanced styling
    for message in st.session_state['messages']:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "👨‍🍳"):
            st.write(message["content"])

    # Chat input with emoji
    user_input = st.chat_input("🤔 How can I assist you with the menu today?")

    if user_input:
        # Add user message to chat history
        st.session_state['messages'].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.write(user_input)
        
        try:
            with st.spinner("🧠 Thinking..."):
                # Only fetch relevant docs if menu analysis is needed
                relevant_docs = find_related_documents(user_input) if should_analyze_menu(user_input) else []
                ai_response = generate_answer(user_input, relevant_docs)
                
            # Add assistant response to chat history
            st.session_state['messages'].append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant", avatar="👨‍🍳"):
                st.write(ai_response)
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")

if __name__ == "__main__":
    main()