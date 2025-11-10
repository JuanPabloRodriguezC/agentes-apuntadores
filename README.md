# agentes-apuntadores
Se desarrollan agentes con el framework de LangChain para montar sobre una interfaz de chat de streamlit con el fin de responder preguntas basadas en el curso de Inteligencia Artificial. La base de datos consiste en representaciones de los apuntes de los estudiantes.

Cree un ambiente de conda: 
```
 conda create --name my_env --python=3.10
```

Deberá instalar los siguientes paquetes utilizando ``` pip install ```

- streamlit
- langchain
- langchain-openai
- langchain-pinecone
- langchain-community
- langgraph
- pinecone-client
- python-dotenv
- duckduckgo-search
- openai

Cree un archivo .env con las siguientes variables:
- OPENAI_API_KEY
- PINECONE_API_KEY

Ejecute la aplicación con 
```
streamlit run agentes.py
```
