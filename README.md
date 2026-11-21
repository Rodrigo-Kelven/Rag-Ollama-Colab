# Rag-Ollama-Colab

Assistente de perguntas e respostas sobre políticas internas de RH, usando **RAG** (Retrieval-Augmented Generation) com execução **local** da interface e dos embeddings/vetores, e inferência do LLM em **Google Colab** através de um túnel **ngrok**.

---

## 1. Visão geral da arquitetura

```
┌──────────────────────────┐        HTTPS (ngrok)        ┌──────────────────────────┐
│  Máquina local (Ubuntu)  │ ──────────────────────────▶ │  Google Colab (T4 GPU)   │
│                          │                             │                          │
│  Streamlit (interface)   │   /api/embeddings           │  Ollama serve :11434     │
│  PyPDFLoader             │   /api/chat                 │   ├─ qwen3-embedding:0.6b │
│  RecursiveTextSplitter   │                             │   └─ qwen3.5:0.8b         │
│  FAISS (índice em memória)│                            │  ngrok http 11434        │
│  LangChain (orquestração)│                             │                          │
└──────────────────────────┘                             └──────────────────────────┘
```

A ideia central é separar as responsabilidades:

| Camada | Tecnologia | Onde roda | Função |
|---|---|---|---|
| Interface | Streamlit | Local | Chat, histórico, sidebar, botões |
| Orquestração | LangChain (LCEL) | Local | Monta o fluxo `recuperar → prompt → LLM → string` |
| Ingestão | `PyPDFLoader` | Local | Lê o PDF e extrai texto por página |
| Chunking | `RecursiveCharacterTextSplitter` | Local | Divide o texto em blocos de 1000 chars com overlap de 200 |
| Banco vetorial | FAISS (`faiss-cpu`) | Local | Armazena e busca vetores por similaridade |
| Embeddings | `OllamaEmbeddings` | Colab (via ngrok) | Transforma chunks e perguntas em vetores |
| Geração | `ChatOllama` | Colab (via ngrok) | Responde com base no contexto recuperado |
| Tunelamento | ngrok | Colab | Expõe a porta 11434 do Ollama para a internet |

**Por que essa divisão?** O PDF, o índice FAISS e os dados da empresa ficam na máquina local (não são enviados para o Colab — apenas os textos dos chunks e a pergunta, para gerar embeddings/resposta). O Colab fornece uma GPU Tesla T4 gratuita para rodar os modelos, que seriam lentos em CPU.

---

## 2. Estrutura do repositório

| Arquivo | Descrição |
|---|---|
| `rag_ollama_colab.py` | Aplicação Streamlit completa (301 linhas): configuração, ingestão, FAISS, chain RAG, chat e sidebar. |
| `ollama_run_colab.ipynb` | Notebook para o Google Colab: instala o Ollama, instala `aiohttp`/`pyngrok`, configura o authtoken e sobe `ollama serve` + túnel ngrok. |
| `politica_rh.pdf` |_base de conhecimento_: manual de 8 páginas da "Nexus Tecnologia S.A." (10 capítulos: jornada, remuneração, férias, reembolsos, ética, segurança da informação, avaliação, comunicação, bem-estar e desligamento). |
| `requirements.txt` | Dependências Python da aplicação local (sem pins de versão). |
| `README.md` | Este documento. |

> O diretório `.venv/` (Python 3.13) existe localmente, mas está vazio de pacotes — as dependências ainda precisam ser instaladas nele.

---

## 3. Como funciona, passo a passo

### 3.1. Lado do Colab (`ollama_run_colab.ipynb`)

1. **Célula 1** — `apt-get install zstd` e `curl -fsSL https://ollama.com/install.sh | sh`: instala o binário do Ollama no ambiente Colab.
2. **Célula 2** — `pip install aiohttp pyngrok`: dependências para o cliente HTTP e para o túnel.
3. **Célula 3** — código Python assíncrono que:
   - define `LD_LIBRARY_PATH=/usr/lib64-nvidia` para encontrar as bibliotecas NVIDIA;
   - registra o authtoken do ngrok (`ngrok config add-authtoken ...`);
   - inicia `ollama serve` (escuta em `127.0.0.1:11434`);
   - inicia `ngrok http 11434 --host-header localhost:11434`, expondo o serviço numa URL pública `https://<id>.ngrok-free.app`.
   - usa `asyncio.gather` para manter os três processos rodando: a célula fica "executando" enquanto o assistente estiver em uso.

> **Atenção:** o notebook **não contém** uma célula `!ollama pull ...`. Os modelos precisam ser baixados manualmente (ver §5.2). Além disso, a URL do ngrok muda a cada reinício do runtime — ela precisa ser copiada para `OLLAMA_URL` no arquivo Python.

### 3.2. Lado local (`rag_ollama_colab.py`)

**a) Configuração (linhas 30–48)**
- Define título/ícone da página, modelos (`qwen3-embedding:0.6b` e `qwen3.5:0.8b`) e o caminho do PDF relativo ao script.
- `OLLAMA_URL` está vazio (`""`) — **deve ser preenchido** com a URL do túnel.

**b) Ingestão e chunking (linhas 55–84)**
- `carregar_pdf()` verifica a existência do arquivo, usa `PyPDFLoader` e valida que há texto extraível.
- `separar_em_blocos()` aplica `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`, preservando contexto entre blocos.

**c) Banco vetorial (linhas 91–116)**
- `criar_banco_vetores()` é decorado com `@st.cache_resource(show_spinner=False)` — o índice é construído uma vez por sessão e reaproveitado.
- Cria `OllamaEmbeddings(model=..., base_url=OLLAMA_URL)` e chama `FAISS.from_documents(...)`.
- O parâmetro `modificacao` recebe `PDF_PATH.stat().st_mtime`: se o PDF mudar, o hash de cache muda e o índice é recriado. (`caminho_pdf` é aceito mas não usado internamente.)

**d) Chain RAG (linhas 123–168)**
- `ChatPromptTemplate` instrui o modelo a responder **apenas** com o contexto, em português do Brasil, e a dizer que não sabe quando a resposta não estiver no texto.
- `as_retriever(search_kwargs={"k": 4})` recupera os 4 chunks mais similares por questão.
- `ChatOllama(..., reasoning=False)` desliga o modo de raciocínio do Qwen 3.5 (respostas mais diretas).
- Fluxo LCEL:
  ```
  {"context": retriever, "question": RunnablePassthrough()} → prompt → llm → StrOutputParser()
  ```
- A pergunta do usuário vai direto para `question` e, ao mesmo tempo, é usada pelo retriever para buscar o `context`.

**e) Inicialização e chat (linhas 175–269)**
- Na primeira execução, tenta montar o banco e a chain dentro de `st.spinner`; em caso de erro, mostra `st.exception`, uma dica de troubleshooting e chama `st.stop()`.
- `st.session_state["messages"]` guarda o histórico; cada mensagem é renderizada com `st.chat_message`.
- A cada `st.chat_input`, a pergunta é anexada ao histórico, `chain.invoke(prompt)` gera a resposta e esta é exibida e armazenada.

**f) Sidebar (linhas 276–301)**
- Mostra os modelos em uso e o banco (FAISS).
- **Limpar conversa**: zera `messages` e faz `st.rerun()`.
- **Reinicializar banco vetorial**: chama `criar_banco_vetores.clear()` (limpa o cache), remove `chain` da sessão e recarrega a página — útil após trocar o PDF ou a URL do ngrok.

### 3.3. Fluxo completo de uma pergunta

```
Usuário digita pergunta
   → pergunta virá vetor (OllamaEmbeddings no Colab)
   → busca dos 4 chunks mais próximos no FAISS (local)
   → chunks + pergunta preenchem o prompt
   → prompt enviado ao ChatOllama (Colab)
   → resposta em texto → StrOutputParser → exibida no chat
```

---

## 4. Pré-requisitos

- Python 3.10+ (o `.venv` existente usa 3.13).
- Conta no Google Colab (de preferência com GPU **T4** — `Runtime → Alterar tipo de runtime → T4 GPU`).
- Conta no [ngrok](https://dashboard.ngrok.com/get-started/your-authtoken) e um **authtoken** gratuito.
- ~2 GB de espaço para os modelos (`qwen3-embedding:0.6b` ≈ 600 MB, `qwen3.5:0.8b` ≈ 750 MB + modelo de especulação/visão).
- Conexão de internet estável (o Colab e a máquina local precisam se enxergar via ngrok).

---

## 5. Como executar

### 5.1. Preparar o ambiente local

```bash
cd ~/Downloads/test-Rag
python3 -m venv .venv                 # se ainda não existir
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Se o `.venv` atual não tiver `pip`, recrie-o: `rm -rf .venv && python3 -m venv .venv`.

### 5.2. Preparar o Colab

1. Abra `ollama_run_colab.ipynb` no [Google Colab](https://colab.research.google.com/).
2. Selecione **T4 GPU** em `Runtime → Alterar tipo de runtime`.
3. Substitua `'your-auth-token'` na célula 3 pelo seu token do ngrok.
4. (Recomendado) adicione uma célula antes da célula 3 para baixar os modelos:
   ```bash
   !ollama pull qwen3-embedding:0.6b
   !ollama pull qwen3.5:0.8b
   ```
   > O download é demorado na primeira vez. Você pode acompanhar com `!ollama list`.
5. Execute as células em ordem (1 → 2 → 3). A última célula fica em execução — não a interrompa.
6. No output da célula 3, localize a linha `started tunnel ... url=https://XXXX.ngrok-free.app` e copie essa URL.

### 5.3. Configurar e subir a aplicação

1. Edite `rag_ollama_colab.py`, linha 41:
   ```python
   OLLAMA_URL = "https://XXXX.ngrok-free.app"   # sem barra final
   ```
2. Inicie o Streamlit:
   ```bash
   streamlit run rag_ollama_colab.py
   ```
3. Acesse `http://localhost:8501`. Na primeira carga, o assistente lê o PDF, gera os chunks e os embeddings (isso pode levar alguns minutos na primeira execução).
4. Faça uma pergunta de teste, por exemplo: *"Quantos dias de home office são permitidos por semana?"*

### 5.4. Alternativa mais simples: Ollama na própria máquina

Se você tiver Ollama instalado localmente, o ngrok/Colab não são necessários:

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.5:0.8b
ollama serve
# em rag_ollama_colab.py:
OLLAMA_URL = "http://localhost:11434"
streamlit run rag_ollama_colab.py
```

---

## 6. Pontos fortes

- Arquitetura simples e desacoplada: trocar o PDF, o modelo ou o provedor é localizado.
- Nenhuma API paga: tudo roda com modelos open-source e GPU gratuita do Colab.
- `st.cache_resource` + `mtime` evita reconstruir o índice a cada interação.
- Prompt com "recusa" explícita reduz alucinações fora do documento.
- Histórico e botões de limpar/reinicializar melhoram a usabilidade.

---

## 7. Problemas conhecidos e melhorias sugeridas

| # | Item | Impacto | Sugestão |
|---|---|---|---|
| 1 | `OLLAMA_URL = ""` no código | Aplicação falha na inicialização se não for editado | Carregar de variável de ambiente (`os.getenv("OLLAMA_URL")`) com validation |
| 2 | URL do ngrok é efêmera | A cada reinício do Colab é preciso editar o `.py` | Exibir a URL num arquivo/JSON ou usar domínio fixo pago |
| 3 | Notebook não faz `ollama pull` | Modelos faltando → erro 404 no Colab | Adicionar célula de download |
| 4 | Token do ngrok nos **outputs salvos** do notebook | **Vazamento de credencial** | Revogar o token no dashboard do ngrok e limpar todos os outputs (`Edit → Clear all outputs`) antes de commitar |
| 5 | Ollama exposto publicamente sem autenticação | Qualquer pessoa pode usar sua GPU/quotas | Usar authtoken do ngrok (`--basic-auth`) ou firewall/IP allowlist |
| 6 | Índice FAISS não é persistido em disco | Recriado a cada novo processo/sessão | Salvar com `FAISS.save_local()` e carregar com `load_local()` |
| 7 | `requirements.txt` sem versões fixadas | Quebras com atualizações do ecossistema LangChain | Fixar versões (ex.: `langchain-ollama>=0.3,<0.4`) |
| 8 | `langchain-openai` e `python-dotenv` no requirements | Dependências não utilizadas | Remover ou passar a usar `.env` de verdade |
| 9 | Flag `--host-header` do ngrok está deprecated | Aviso no log; pode falhar em versões futuras | Migrar para traffic policy / `--request-header` |
| 10 | Bloqueio do navegador do ngrok free | Cliente HTTP pode receber página de aviso | Enviar header `Ngrok-Skip-Browser-Warning` ou usar plano pago |
| 11 | Sem streaming de resposta | Usuário espera o texto completo | Usar `stream=True` + `st.write_stream` |
| 12 | `caminho_pdf` ignorado em `criar_banco_vetores` | Parâmetro enganoso | Usar o parâmetro no `PyPDFLoader` em vez da global `PDF_PATH` |
| 13 | Sem `.gitignore` | `.venv/` pode ser versionado por engano | Adicionar `.venv/`, `__pycache__/`, `.env` |
| 14 | Sem testes | Mudanças no prompt/chain não são validadas | Testes unitários para `carregar_pdf`, `separar_em_blocos` e a chain |

---

## 8. Resumo rápido de execução

```bash
# Colab: células 1→3 (T4 GPU, authtoken ngrok, ollama pull dos 2 modelos)
# Copiar URL https://XXXX.ngrok-free.app do output

# Local:
source .venv/bin/activate
pip install -r requirements.txt
# editar OLLAMA_URL em rag_ollama_colab.py
streamlit run rag_ollama_colab.py   # → http://localhost:8501
```
