import os
import streamlit as st
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import AsyncGenerator
import time

# Load environment variables
load_dotenv()

#---------------------------------------------
# RAG Backend Implementation
#---------------------------------------------
class RAGBackend:
    def __init__(self, markdown_file_path=None, markdown_content=None):
        """
        Initialize the RAG system with product data.
        Either provide a file path or markdown content directly.
        """
        # Create async OpenAI client
        self.client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                                  api_key=os.getenv("OPENROUTER_API_KEY"))
        
        self.markdown_file_path = markdown_file_path
        if markdown_content:
            self.product_data = markdown_content
        elif markdown_file_path:
            self.product_data = self._load_markdown_file()
        else:
            self.product_data = ""
    
    def _load_markdown_file(self):
        """Load and read the markdown file."""
        try:
            with open(self.markdown_file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error loading markdown file: {e}")
            return ""
    
    def get_system_prompt(self, user_question=None):
        """Generate the system prompt with product data and instructions."""
        if not self.product_data:
            return "Error: No product data available."
        
        return f"""
        You are the roofing product and company information assistant. Below is the product catalog information and some company information:

        ==<|STARTOF_ROOFING_DATA|>==
        
        {self.product_data}

        ==<|ENDOF_ROOFING_DATA|>==
        
        Instructions for answering:
        1. Answer questions only based on the information provided above.
        2. If asked about a specific product, provide all available details for that product.
        3. For every query about produtcs, be explicit about in-stock status and the website link (URL).
        4. When mentioning prices, always include the currency symbol.
        5. If information is not available in the provided data, politely state that you don't have that information and headover to anton online store "https://roofing.lk/".
        6. Keep responses concise and focused on the question asked.
        7. Format the response in a clear, readable way.
        8. Do not make up or assume any product information not present in the data.
        9. Use markdown formatting when appropriate to make your response more readable.
        """

    
    async def query(self, user_question):
        """
        Query the product information based on user question.
        Uses OpenAI API to generate a response based on the product data.
        """
        if not self.product_data:
            return "Error: No product data available. Please check the markdown file."
        
        # Ensure user_question is not None and is a string
        if user_question is None:
            user_question = ""
        else:
            user_question = str(user_question)
            
        system_prompt = self.get_system_prompt()
        
        # Ensure system_prompt is not None
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        
        try:
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model="google/gemini-2.0-flash-001",  
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                temperature=0.1  # Lower temperature for more factual responses
            )
            
            # Return the assistant's response
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error querying API: {e}")
            return f"Error processing your request: {str(e)}"
    
    async def stream_query(self, user_question) -> AsyncGenerator[str, None]:
        """
        Stream the response from API for a given user question.
        """
        # Print debugging information
        print(f"Debug - user_question type: {type(user_question)}")
        print(f"Debug - user_question value: '{user_question}'")
        
        # Handle empty product data
        if not self.product_data:
            yield "Error: No product data available. Please check the markdown file."
            return
        
        # Force user_question to be a non-empty string
        if user_question is None or user_question == "":
            user_question = "Hello"  # Default fallback question
        
        # Ensure it's a string
        user_question = str(user_question).strip()
        
        # Double-check we have content
        if not user_question:
            user_question = "Hello"  # Another fallback

        # Get and validate system prompt
        system_prompt = self.get_system_prompt()
        if not system_prompt or system_prompt == "Error: No product data available.":
            system_prompt = "You are a helpful assistant with knowledge about roofing Product Information."
        
        # Debug the message payload
        print(f"Debug - system_prompt length: {len(system_prompt)}")
        print(f"Debug - Final user_question: '{user_question}'")
        
        # Create messages with extra validation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
        
        try:
            # Call OpenAI API with streaming
            stream = await self.client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=messages,
                temperature=0.1,
                stream=True
            )
            
            # Yield each chunk as it arrives
            async for chunk in stream:
                if (hasattr(chunk.choices[0].delta, 'content') and 
                    chunk.choices[0].delta.content is not None):
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            print(f"Error streaming from OpenAI API: {e}")
            yield f"Error processing your request: {str(e)}"

#---------------------------------------------
# Streamlit UI Implementation
#---------------------------------------------

def load_markdown_content(file_path):
    """Load content from a markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        st.error(f"Error loading markdown file: {e}")
        return None

def initialize_rag_backend(markdown_content):
    """Initialize the RAG backend with markdown content."""
    return RAGBackend(markdown_content=markdown_content)

def main():
    # Page config - Using centered layout with collapsed sidebar
    st.set_page_config(
        page_title="Product Information Assistant",
        page_icon="🛎️",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Apply custom CSS for styling with darker theme
    st.markdown("""
    <style>
        .stChat {
            border-radius: 10px;
            padding: 10px;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
        }
        .chat-message.user {
            background-color: #2b313e;
        }
        .chat-message.assistant {
            background-color: #475063;
        }
        .app-title {
            text-align: center;
            margin-bottom: 2rem;
        }
        .stButton button {
            border-radius: 20px;
            padding: 0.5rem 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    if "context_loaded" not in st.session_state:
        # Load markdown content
        markdown_content = load_markdown_content("product_catalog.md")
        if not markdown_content:
            st.error("Failed to load markdown content.")
            st.stop()
        
        # Store the context in session state
        st.session_state.markdown_context = markdown_content
        st.session_state.context_loaded = True
        
        # Initialize RAG backend
        st.session_state.rag_backend = initialize_rag_backend(markdown_content)
    
    # Title area - simplified layout
    # st.markdown(
    #     """
    #     <div style="display: flex; justify-content: center;">
    #         <img src="roofing-logo.png" width="500">
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )
    st.markdown("<h1 class='app-title'>Product Information Assistant</h1>", unsafe_allow_html=True)
    
    # Top controls row with logo and new chat button
    col1, col2, col3 = st.columns([2, 5, 2])
        
    with col2:
        try:
            st.image("roofing-logo.png",width=325)
        except Exception as e:
            print(f"Error loading logo: {e}")
    
    with col3:
        new_chat_button = st.button(
            "New Chat", 
            use_container_width=True
        )
        if new_chat_button:
            st.session_state.messages = []
            st.session_state.processing = False
            st.success("Started a new conversation!")
            time.sleep(0.5)  # Short delay for the success message to be seen
            st.rerun()
    
    # Display chat messages - simple divider
    st.divider()
    
    # Display chat messages using Streamlit's built-in chat message component
    message_container = st.container()
    with message_container:
        if not st.session_state.messages:
            # Empty state message
            st.markdown("""
            <div style="text-align: center; padding: 20px; color: #888;">
                <h3>Welcome to Product Information Assistant!</h3>
                <p>Have a question about our products? Ask away!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.messages:
                role = message["role"]
                content = message["content"]
                # Use Streamlit's built-in chat message component
                st.chat_message(role).write(content)
    
    # Another divider before input
    st.divider()
    
    # Chat input
    user_input = st.chat_input("Ask a question about our products...", disabled=st.session_state.processing)
    
    if user_input and not st.session_state.processing:
        # Validate user input
        print(f"Debug - Raw user input: '{user_input}'")
        user_input = str(user_input).strip()
        print(f"Debug - Processed user input: '{user_input}'")
        
        if not user_input:
            st.warning("Please enter a non-empty message.")
            st.stop()
        
        # Mark as processing to prevent multiple requests
        st.session_state.processing = True
        
        # Add user message to messages list
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Rerun to show user message immediately
        st.rerun()
    
    # Process the message after showing user message
    if st.session_state.processing and len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        # Get the last user message
        user_message = st.session_state.messages[-1]["content"]
        print(f"Debug - Processing user message: '{user_message}'")
        
        # Extra validation to ensure we have a valid user message
        if not user_message or user_message.strip() == "":
            st.error("Cannot process empty message")
            st.session_state.processing = False
            st.rerun()
        
        # Create status indicator
        status = st.status("Generating response...", expanded=False)
        
        # Add assistant message placeholder
        assistant_container = st.chat_message("assistant")
        message_placeholder = assistant_container.empty()
        
        # Add an empty assistant message to our history
        assistant_msg = {"role": "assistant", "content": ""}
        st.session_state.messages.append(assistant_msg)
        
        # Reference to update the last message
        last_idx = len(st.session_state.messages) - 1
        
        try:
            # Generate response using the RAG backend with validated user message
            async def run_stream_query():
                full_response = ""
                try:
                    async for text_chunk in st.session_state.rag_backend.stream_query(user_message):
                        if text_chunk:  # Check if chunk is not empty
                            full_response += text_chunk
                            # Update the placeholder with the current response
                            message_placeholder.write(full_response)
                            
                            # Update the message in session state
                            st.session_state.messages[last_idx]["content"] = full_response
                            status.update(label="Generating response...", state="running")
                            time.sleep(0.01)  # Slight delay for UI updates
                    
                    # If we got an empty response, provide a fallback
                    if not full_response:
                        full_response = "I'm having trouble generating a response right now. Please try again."
                        message_placeholder.write(full_response)
                        st.session_state.messages[last_idx]["content"] = full_response
                        
                except Exception as e:
                    print(f"Stream query error: {e}")
                    full_response = f"Sorry, there was an error generating a response: {str(e)}"
                    message_placeholder.write(full_response)
                    st.session_state.messages[last_idx]["content"] = full_response
                
                # Final update with complete response
                status.update(label="Response complete!", state="complete")
            
            # Run the async function
            asyncio.run(run_stream_query())
            
            # Close status after a short delay
            time.sleep(0.5)
            status.update(state="complete", expanded=False)
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            message_placeholder.write(error_msg)
            st.session_state.messages[last_idx]["content"] = error_msg
            status.update(label="Error", state="error")
        
        # Reset processing flag
        st.session_state.processing = False
        st.rerun()

    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; color: #888;">
        <p>Product Information Assistant</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
