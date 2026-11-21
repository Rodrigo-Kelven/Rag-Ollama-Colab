"""
Assistente RAG de Políticas Internas (RH)

Arquitetura:
- Streamlit: Interface local
- LangChain: Orquestração
- FAISS: Banco vetorial local
- Ollama: Google Colab
- Ngrok: Comunicação entre Ubuntu e Colab
"""

import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# =====================================================
# 1. CONFIGURAÇÕES
# =====================================================

st.set_page_config(
    page_title="Assistente RAG - RH",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Assistente RAG — Políticas Internas")

st.caption("Assistente inteligente para consulta de políticas internas.")

# Endereço do Ngrok gerado no Google Colab
OLLAMA_URL = ""

# Modelos
MODELO_EMBEDDING = "qwen3-embedding:0.6b"
MODELO_LLM = "qwen3.5:0.8b"

# Documento PDF
PDF_PATH = Path(__file__).parent / "politica_rh.pdf"


# =====================================================
# 2. CARREGAMENTO E PROCESSAMENTO DO PDF
# =====================================================

def carregar_pdf():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo PDF não encontrado: {PDF_PATH}"
        )

    pdf_loader = PyPDFLoader(str(PDF_PATH))

    documentos = pdf_loader.load()

    if not documentos:
        raise ValueError("O PDF não possui conteúdo extraível.")

    return documentos


def separar_em_blocos(documentos):

    separador = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    lista_blocos = separador.split_documents(documentos)

    if not lista_blocos:
        raise ValueError("Não foi possível gerar os chunks.")

    return lista_blocos


# =====================================================
# 3. CRIAÇÃO DO BANCO VETORIAL
# =====================================================

@st.cache_resource(show_spinner=False)
def criar_banco_vetores(caminho_pdf, modificacao):

    st.write("📄 Carregando documento PDF...")

    documentos = carregar_pdf()

    st.write("✂️ Dividindo documento em chunks...")

    lista_blocos = separar_em_blocos(documentos)

    st.write(f"📚 Total de chunks: {len(lista_blocos)}")

    st.write("🧠 Gerando embeddings através do Ollama...")

    ferramenta_embedding = OllamaEmbeddings(
        model=MODELO_EMBEDDING,
        base_url=OLLAMA_URL
    )

    banco_vetores = FAISS.from_documents(
        lista_blocos,
        ferramenta_embedding
    )

    return banco_vetores


# =====================================================
# 4. CRIAÇÃO DO AGENTE RAG
# =====================================================

def criar_chain_agente(banco_vetores):

    prompt_template = ChatPromptTemplate.from_template(
        """
        Você é um assistente de RH que responde perguntas
        sobre políticas internas da empresa.

        Utilize APENAS as informações presentes no contexto.

        Se não encontrar a resposta no contexto,
        informe claramente que não sabe responder.

        Responda em português do Brasil,
        de maneira clara e objetiva.

        Contexto:
        {context}

        Pergunta:
        {question}

        Resposta:
        """
    )

    buscador_contexto = banco_vetores.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatOllama(
        model=MODELO_LLM,
        base_url=OLLAMA_URL,
        reasoning=False
    )

    chain = (
        {
            "context": buscador_contexto,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return chain


# =====================================================
# 5. INICIALIZAÇÃO DO RAG
# =====================================================

if "chain" not in st.session_state:

    try:

        with st.spinner(
            "Preparando o assistente e conectando ao Ollama..."
        ):

            banco_vetores = criar_banco_vetores(
                str(PDF_PATH),
                PDF_PATH.stat().st_mtime
            )

            chain = criar_chain_agente(banco_vetores)

            st.session_state["chain"] = chain

        st.success("Assistente inicializado com sucesso!")

    except Exception as erro:

        st.error("Não foi possível inicializar o assistente.")

        st.exception(erro)

        st.info(
            "Verifique se o Ollama está ativo no Google Colab, "
            "se o túnel ngrok está funcionando e se os modelos "
            "estão disponíveis."
        )

        st.stop()


# =====================================================
# 6. HISTÓRICO DE CONVERSAÇÃO
# =====================================================

if "messages" not in st.session_state:

    st.session_state["messages"] = []


for msg in st.session_state["messages"]:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])


# =====================================================
# 7. INTERFACE DO CHAT
# =====================================================

prompt = st.chat_input(
    "Pergunte sobre as políticas da empresa..."
)


if prompt:

    # Mensagem do usuário
    st.session_state["messages"].append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):

        st.write(prompt)


    # Resposta do assistente
    with st.chat_message("assistant"):

        with st.spinner("Consultando documentos..."):

            try:

                answer = st.session_state["chain"].invoke(prompt)

                st.write(answer)

                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as erro:

                st.error(f"Erro ao gerar resposta: {erro}")


# =====================================================
# 8. BARRA LATERAL
# =====================================================

with st.sidebar:

    st.header("Configurações")

    st.write("**Modelo LLM:**")
    st.code(MODELO_LLM)

    st.write("**Modelo Embedding:**")
    st.code(MODELO_EMBEDDING)

    st.write("**Banco vetorial:**")
    st.write("FAISS")

    if st.button("Limpar conversa"):

        st.session_state["messages"] = []

        st.rerun()

    if st.button("Reinicializar banco vetorial"):

        criar_banco_vetores.clear()

        st.session_state.pop("chain", None)

        st.rerun()
