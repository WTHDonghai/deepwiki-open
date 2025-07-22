# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

### Development Setup
```bash
# Install dependencies
npm install
pip install -r api/requirements.txt

# Start backend (FastAPI)
python -m api.main
# or specify port: PORT=8001 python -m api.main

# Start frontend (Next.js)
npm run dev
# or: yarn dev

# Build for production
npm run build
```

### Docker Setup
```bash
# Quick start with Docker Compose
docker-compose up

# Build and run locally
docker build -t deepwiki-open .
docker run -p 8001:8001 -p 3000:3000 -v ~/.adalflow:/root/.adalflow -v $(pwd)/.env:/app/.env deepwiki-open
```

### Testing & Linting
```bash
# Frontend linting
npm run lint

# Python tests
python -m pytest test/
```

## Architecture Overview

DeepWiki-Open is a full-stack application that generates interactive wikis from Git repositories using AI:

### Core Components

**Backend (FastAPI)**
- **api/api.py**: Main FastAPI application with endpoints for wiki generation, chat, and repository analysis
- **api/rag.py**: Retrieval Augmented Generation system using FAISS for code embeddings
- **api/data_pipeline.py**: Repository cloning, parsing, and indexing pipeline
- **api/tools/embedder.py**: Embedding generation for code chunks
- **api/clients/**: AI provider integrations (Google Gemini, OpenAI, Azure, Ollama, Bedrock, Dashscope, OpenRouter)

**Frontend (Next.js 15)**
- **src/app/page.tsx**: Main application entry point
- **src/app/[owner]/[repo]/**: Dynamic routes for individual repository wikis
- **src/components/**: React components including Mermaid diagrams, configuration modals, and chat interface
- **src/contexts/LanguageContext.tsx**: Multi-language support (10 languages)

**AI Integration**
- **Provider system**: Supports 6 AI providers via JSON configuration (api/config/generator.json)
- **Embedding**: Uses OpenAI embeddings by default, configurable via api/config/embedder.json
- **RAG**: FAISS-based retrieval with configurable chunking strategies

### Key Data Flow

1. **Repository Processing**: User enters repo URL → Backend clones → Code analysis → Embedding generation → FAISS indexing
2. **Wiki Generation**: AI generates structured documentation → Mermaid diagrams → Cached in ~/.adalflow/
3. **Interactive Chat**: User questions → RAG retrieval → Streamed AI responses with code context

### Configuration Files

- **api/config/generator.json**: AI models and parameters per provider
- **api/config/embedder.json**: Embedding configuration
- **api/config/repo.json**: Repository filtering rules
- **.env**: API keys and environment variables

### Environment Variables

**Required**:
- `OPENAI_API_KEY`: For embeddings (even if using other providers)

**Optional** (per provider):
- `GOOGLE_API_KEY`: Google Gemini
- `OPENROUTER_API_KEY`: OpenRouter access
- `AZURE_OPENAI_*`: Azure OpenAI credentials
- `OLLAMA_HOST`: Local Ollama endpoint
- `BEDROK_*`: AWS Bedrock access

**Other**:
- `PORT`: API server port (default: 8001)
- `DEEPWIKI_AUTH_MODE`: Enable authorization mode
- `LOG_LEVEL`: Logging verbosity

### Data Storage

- **~/.adalflow/repos/**: Cloned repositories
- **~/.adalflow/databases/**: FAISS indexes and embeddings
- **~/.adalflow/wikicache/**: Generated wiki content
- **api/logs/**: Application logs (mounted in Docker)

### Key Technologies

- **Backend**: FastAPI, Pydantic, FAISS, AdaFlow, LangChain components
- **Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS, Mermaid
- **AI**: Google Gemini, OpenAI GPT-4o, Claude, Llama, AWS Bedrock, Azure OpenAI
- **Deployment**: Docker, Docker Compose, GitHub Container Registry