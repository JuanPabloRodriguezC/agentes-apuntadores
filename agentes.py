import streamlit as st
import os
import time
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("apuntes-ia")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create vector store
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",
    namespace="parrafos"
)

# Initialize web search
search_wrapper = DuckDuckGoSearchAPIWrapper()

# System prompt
SYSTEM_PROMPT = """Eres un experto asistente de inteligencia artificial que ayuda a los estudiantes con sus preguntas sobre notas de clase.

Tienes acceso a las siguientes herramientas:

- search_database: usa esto para encontrar información de las notas de clase
- web_search: usa esto para buscar en la web

Instrucciones importantes:
1. SIEMPRE usa 'search_database' primero para responder preguntas sobre las notas de clase
2. SOLO usa 'web_search' si el usuario explícitamente pide buscar en internet
3. Cuando uses información de search_database, SIEMPRE incluye las fuentes en tu respuesta
4. Si no encuentras información relevante en las notas, dile al usuario
5. Responde en español de manera clara y concisa"""

# Define tools
@tool
def search_database(query: str) -> str:
    """Busca información en las notas de clase. Usa esta herramienta cuando el usuario pregunta sobre el contenido de las clases."""
    
    results = vectorstore.similarity_search_with_score(query, k=3)
    
    if not results or results[0][1] > 0.5:  # Lower score is better
        return "No se encontraron documentos relevantes en las notas de clase."
    
    # Format results with sources
    formatted_results = []
    for i, (doc, score) in enumerate(results, 1):
        content = doc.page_content
        metadata = doc.metadata
        
        # Create source citation
        source_info = f"\n**Fuente {i}:**"
        if 'source' in metadata:
            source_info += f" {metadata['source']}"
        if 'page' in metadata:
            source_info += f" (página {metadata['page']})"
        
        formatted_results.append(f"{content}{source_info}")
    
    return "\n\n---\n\n".join(formatted_results)


@tool
def web_search(query: str) -> str:
    """Busca información en la web. Usa esta herramienta SOLO cuando el usuario explícitamente pida buscar en internet."""
    
    try:
        results = search_wrapper.run(query)
        return f"Resultados de búsqueda web:\n\n{results}"
    except Exception as e:
        return f"Error al buscar en la web: {str(e)}"


# Initialize model
model = init_chat_model(
    model="gpt-3.5-turbo",
    model_provider="openai",
    temperature=0.3,
)

# Create checkpointer for memory
checkpointer = InMemorySaver()

# Create agent
agent = create_agent(
    model=model,
    tools=[search_database, web_search],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer
)

# Streamlit UI
st.set_page_config(page_title="Asistente IA - Notas de Clase", page_icon="🤖")
st.title("Asistente de Notas de Clase")
st.caption("Pregúntame sobre tus notas de clase")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Puedo ayudarte con preguntas sobre tus notas de clase. ¿Qué te gustaría saber?"}
    ]

# Thread ID for conversation memory
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "thread_1"

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Invoke agent with proper format
            response = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": st.session_state.thread_id}}
            )
            
            # Extract the assistant's response
            # The response contains a 'messages' list, get the last message
            if "messages" in response:
                full_response = response["messages"][-1].content
            else:
                full_response = str(response)
            
            # Simulate typing effect
            displayed_response = ""
            for chunk in full_response.split():
                displayed_response += chunk + " "
                time.sleep(0.03)
                message_placeholder.markdown(displayed_response + "▌")
            
            # Final response without cursor
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            full_response = f"Lo siento, ocurrió un error: {str(e)}"
            message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar
with st.sidebar:    
    if st.button("Limpiar conversación"):
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! Puedo ayudarte con preguntas sobre tus notas de clase. ¿Qué te gustaría saber?"}
        ]
        # Reset thread ID for new conversation
        st.session_state.thread_id = f"thread_{int(time.time())}"
        st.rerun()