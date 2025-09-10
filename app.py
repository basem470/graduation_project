import streamlit as st
import datetime
import json
#import router here

# --- Page Configuration ---
st.set_page_config(
    page_title="Helios Dynamics ERP",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for minimal styling ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    .chat-message {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    .user-message {
        background-color: #f8f9fa;
        border-left-color: #007bff;
    }
    
    .assistant-message {
        background-color: #f0f8f0;
        border-left-color: #28a745;
    }
    
    .timestamp {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 0.25rem;
    }
    
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .message-content{
            color: #363636;
            }
    .sidebar-section {
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize session state ---
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session" not in st.session_state:
    st.session_state.current_session = None
if "session_counter" not in st.session_state:
    st.session_state.session_counter = 0

# --- Response Function (easily modifiable) ---
def get_assistant_response(user_message): 
    """
    Simple echo response - MODIFY THIS FUNCTION to integrate with your backend/LangChain
    
    Args:
        user_message (str): The user's input message
        
    Returns:
        str: The assistant's response
    """
    # Simple echo response - replace this with your actual logic
    return f"Echo: {user_message}" <- #rutern router.invoke use messages

# --- Chat History Management ---
def create_new_session():
    """Create a new chat session"""
    st.session_state.session_counter += 1
    session_id = f"session_{st.session_state.session_counter}"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    st.session_state.chat_sessions[session_id] = {
        "name": f"Chat {st.session_state.session_counter}",
        "created": timestamp,
        "messages": []
    }
    st.session_state.current_session = session_id
    return session_id

def delete_session(session_id):
    """Delete a chat session"""
    if session_id in st.session_state.chat_sessions:
        del st.session_state.chat_sessions[session_id]
        if st.session_state.current_session == session_id:
            st.session_state.current_session = None

def get_current_messages():
    """Get messages from current session"""
    if st.session_state.current_session and st.session_state.current_session in st.session_state.chat_sessions:
        return st.session_state.chat_sessions[st.session_state.current_session]["messages"]
    return []

def add_message(role, content):
    """Add message to current session"""
    if st.session_state.current_session:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        st.session_state.chat_sessions[st.session_state.current_session]["messages"].append(message)

# --- Sidebar for Chat History ---
with st.sidebar:
    st.markdown("### 🚀 Helios Dynamics ERP")
    st.markdown("**by Team 15**")
    
    # New chat button
    if st.button("➕ New Chat", use_container_width=True):
        create_new_session()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Chat History")
    
    # List all chat sessions
    if st.session_state.chat_sessions:
        for session_id, session_data in st.session_state.chat_sessions.items():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Session button
                if st.button(
                    f"💬 {session_data['name']}", 
                    key=f"select_{session_id}",
                    use_container_width=True,
                    type="primary" if session_id == st.session_state.current_session else "secondary"
                ):
                    st.session_state.current_session = session_id
                    st.rerun()
            
            with col2:
                # Delete button
                if st.button("🗑️", key=f"delete_{session_id}", help="Delete chat"):
                    delete_session(session_id)
                    st.rerun()
            
            # Show creation time
            st.caption(f"📅 {session_data['created']}")
            
            # Show message count
            message_count = len(session_data['messages'])
            st.caption(f"💬 {message_count} messages")
            
            st.markdown("---")
    else:
        st.markdown("*No chat history yet*")
        st.markdown("Click 'New Chat' to start!")
    
    # Export chat history
    if st.session_state.chat_sessions:
        st.markdown("### Export")
        if st.button("💾 Export All Chats", use_container_width=True):
            # Create downloadable JSON
            export_data = {
                "export_date": datetime.datetime.now().isoformat(),
                "sessions": st.session_state.chat_sessions
            }
            st.download_button(
                label="📄 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"helios_chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# --- Main Chat Interface ---
st.markdown("# Helios Dynamics ERP Assistant")

# Check if we have a current session
if not st.session_state.current_session:
    st.info("👈 Click 'New Chat' in the sidebar to start a conversation")
    st.stop()

# Get current session data
current_messages = get_current_messages()
session_name = st.session_state.chat_sessions[st.session_state.current_session]["name"]

# Display current session info
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown(f"**Current Chat:** {session_name}")
with col2:
    st.markdown(f"**Messages:** {len(current_messages)}")
with col3:
    # Rename session
    if st.button("✏️ Rename"):
        new_name = st.text_input("New name:", value=session_name, key="rename_input")
        if new_name:
            st.session_state.chat_sessions[st.session_state.current_session]["name"] = new_name
            st.rerun()

# --- Chat Display ---
st.markdown("---")

# Chat messages container
with st.container():
    if current_messages:
        for message in current_messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong><span class="message-content">👤 You:</span></strong><br>
                    <span class="message-content">{message["content"]}</span>
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong><span class="message-content"><strong>🤖 Assistant:</span></strong><br>
                    <span class="message-content">{message["content"]}</span>
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("*Start typing below to begin the conversation...*")

# --- Input Area ---
st.markdown("---")

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message:",
            placeholder="Ask me anything about ERP operations...",
            key="message_input",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("<div style='margin-top: 0px;'></div>", unsafe_allow_html=True)
        send_button = st.form_submit_button("Send 🚀", use_container_width=True)

# Handle form submission
if send_button and user_input:
    # Add user message
    add_message("user", user_input)
    
    # Get assistant response (modify the get_assistant_response function above)
    with st.spinner("Thinking..."):
        assistant_response = get_assistant_response(user_input)
    
    # Add assistant message
    add_message("assistant", assistant_response)
    
    # Refresh the page to show new messages
    st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
    Helios Dynamics ERP Assistant | Team 15 | 
    <em>Modify the `get_assistant_response()` function to integrate with your backend</em>
</div>
""", unsafe_allow_html=True)