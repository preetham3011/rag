# Adaptive Context Compression RAG System - UML Diagrams

## 1. Use Case Diagram

```mermaid
graph TB
    subgraph "Adaptive Context Compression RAG System"
        UC1[Upload Document]
        UC2[Process Document]
        UC3[Query System]
        UC4[Detect Intent]
        UC5[Retrieve Chunks]
        UC6[Compress Context]
        UC7[Generate Answer]
        UC8[View Results]
        UC9[Export Results]
        UC10[Configure System]
        UC11[View Metrics]
    end
    
    Researcher((Researcher))
    Admin((Admin))
    LLM[Google Gemini API]
    VectorDB[FAISS Vector Store]
    
    Researcher -->|uploads| UC1
    Researcher -->|asks| UC3
    Researcher -->|views| UC8
    Researcher -->|exports| UC9
    
    Admin -->|configures| UC10
    Admin -->|monitors| UC11
    
    UC1 -->|triggers| UC2
    UC3 -->|triggers| UC4
    UC4 -->|triggers| UC5
    UC5 -->|triggers| UC6
    UC6 -->|triggers| UC7
    UC7 -->|displays| UC8
    
    UC2 -.->|uses| VectorDB
    UC5 -.->|queries| VectorDB
    UC7 -.->|calls| LLM
    
    style UC1 fill:#1565C0,stroke:#0D47A1,color:#fff
    style UC3 fill:#1565C0,stroke:#0D47A1,color:#fff
    style UC7 fill:#C62828,stroke:#B71C1C,color:#fff
    style Researcher fill:#2E7D32,stroke:#1B5E20,color:#fff
    style Admin fill:#F57F17,stroke:#E65100,color:#fff
```

## 2. Class Diagram

```mermaid
classDiagram
    class RAGController {
        +config: Config
        +pipeline: Pipeline
        +process_query(query: str) Answer
        +upload_document(file: File) bool
        +initialize_system() void
    }
    
    class DocumentProcessor {
        -pdf_parser: PyMuPDF
        +extract_text(pdf: File) str
        +detect_sections(text: str) List~Section~
        +clean_text(text: str) str
    }
    
    class SectionDetector {
        -section_patterns: List~Pattern~
        +detect_abstract(text: str) Section
        +detect_methodology(text: str) Section
        +detect_results(text: str) Section
        +detect_references(text: str) Section
    }
    
    class Chunker {
        -chunk_size: int
        -overlap: int
        +chunk_by_sentence(text: str) List~Chunk~
        +chunk_by_section(sections: List~Section~) List~Chunk~
        +calculate_overlap(chunks: List~Chunk~) void
    }
    
    class EmbeddingGenerator {
        -model: SentenceTransformer
        +generate_embeddings(text: str) ndarray
        +batch_generate(texts: List~str~) List~ndarray~
    }
    
    class VectorStore {
        -index: FAISS
        -metadata: Dict
        +add_embeddings(vectors: List~ndarray~, metadata: Dict) void
        +search(query_vector: ndarray, k: int) List~Chunk~
        +save_index(path: str) void
        +load_index(path: str) void
    }
    
    class IntentDetector {
        -intent_patterns: Dict
        +detect_intent(query: str) Intent
        +classify_query(query: str) str
    }
    
    class Intent {
        <<enumeration>>
        METHOD
        RESULT
        API_USAGE
        DEFINITION
        COMPARISON
    }
    
    class Retriever {
        -vector_store: VectorStore
        -embedding_gen: EmbeddingGenerator
        -top_k: int
        +retrieve(query: str) List~Chunk~
        +rerank(chunks: List~Chunk~, query: str) List~Chunk~
    }
    
    class EvidenceSelector {
        -sentence_splitter: SpaCy
        +select_sentences(chunks: List~Chunk~, intent: Intent) List~Sentence~
        +score_relevance(sentence: str, query: str) float
        +rank_by_position(sentences: List~Sentence~) List~Sentence~
    }
    
    class BudgetManager {
        -max_tokens: int
        -current_usage: int
        +calculate_tokens(text: str) int
        +enforce_budget(evidence: List~Sentence~) List~Sentence~
        +get_compression_ratio() float
    }
    
    class CompressionStrategy {
        <<interface>>
        +compress(chunks: List~Chunk~, budget: int) str
    }
    
    class MethodCompressionStrategy {
        +compress(chunks: List~Chunk~, budget: int) str
        -prioritize_methodology() void
    }
    
    class ResultCompressionStrategy {
        +compress(chunks: List~Chunk~, budget: int) str
        -prioritize_findings() void
    }
    
    class APICompressionStrategy {
        +compress(chunks: List~Chunk~, budget: int) str
        -prioritize_code_examples() void
    }
    
    class LLMWrapper {
        -api_key: str
        -model: str
        +generate_answer(prompt: str, context: str) str
        +validate_response(response: str) bool
    }
    
    class Answer {
        +text: str
        +citations: List~Citation~
        +tokens_used: int
        +compression_ratio: float
        +intent: Intent
    }
    
    class Citation {
        +page_number: int
        +chunk_id: str
        +text_snippet: str
    }
    
    class Chunk {
        +id: str
        +text: str
        +embedding: ndarray
        +metadata: Dict
        +section_type: str
        +page_number: int
    }
    
    class Config {
        +chunk_size: int
        +embedding_model: str
        +llm_model: str
        +max_tokens: int
        +top_k: int
    }
    
    RAGController --> DocumentProcessor
    RAGController --> Retriever
    RAGController --> IntentDetector
    RAGController --> EvidenceSelector
    RAGController --> BudgetManager
    RAGController --> LLMWrapper
    RAGController --> Config
    
    DocumentProcessor --> SectionDetector
    DocumentProcessor --> Chunker
    
    Chunker --> EmbeddingGenerator
    EmbeddingGenerator --> VectorStore
    
    Retriever --> VectorStore
    Retriever --> EmbeddingGenerator
    
    IntentDetector --> Intent
    
    EvidenceSelector --> CompressionStrategy
    CompressionStrategy <|.. MethodCompressionStrategy
    CompressionStrategy <|.. ResultCompressionStrategy
    CompressionStrategy <|.. APICompressionStrategy
    
    BudgetManager --> Answer
    LLMWrapper --> Answer
    Answer --> Citation
    VectorStore --> Chunk
```

## 3. Data Flow Diagram (Level 0 - Context Diagram)

```mermaid
graph LR
    User((User/Researcher))
    
    subgraph "RAG System"
        System[Adaptive Context<br/>Compression RAG]
    end
    
    PDF[(PDF Document)]
    LLM[Google Gemini<br/>API]
    VectorDB[(FAISS<br/>Vector Store)]
    
    User -->|1. Upload PDF| PDF
    PDF -->|2. Document| System
    User -->|3. Query| System
    System -->|4. Store Vectors| VectorDB
    VectorDB -->|5. Retrieve Chunks| System
    System -->|6. Compressed Context| LLM
    LLM -->|7. Generated Answer| System
    System -->|8. Results & Metrics| User
    
    style System fill:#2E7D32,stroke:#1B5E20,color:#fff
    style User fill:#1565C0,stroke:#0D47A1,color:#fff
    style LLM fill:#EF6C00,stroke:#E65100,color:#fff
    style VectorDB fill:#6A1B9A,stroke:#4A148C,color:#fff
```

## 3. Data Flow Diagram (Level 1 - System Processes)

```mermaid
graph TB
    subgraph "Ingestion Pipeline"
        P1[1.0<br/>Document<br/>Processing]
        P2[2.0<br/>Chunking &<br/>Embedding]
    end
    
    subgraph "Query Pipeline"
        P3[3.0<br/>Intent<br/>Detection]
        P4[4.0<br/>Vector<br/>Retrieval]
        P5[5.0<br/>Evidence<br/>Selection]
        P6[6.0<br/>Context<br/>Compression]
        P7[7.0<br/>Answer<br/>Generation]
    end
    
    User((User))
    
    D1[(Document<br/>Store)]
    D2[(Vector<br/>Store)]
    D3[(Metadata<br/>Store)]
    
    LLM[Gemini API]
    
    User -->|PDF File| P1
    P1 -->|Raw Text| P2
    P1 -->|Sections| D1
    P2 -->|Vectors| D2
    P2 -->|Chunk Metadata| D3
    
    User -->|Query| P3
    P3 -->|Intent Type| P4
    D2 -->|Top-K Chunks| P4
    P4 -->|Retrieved Chunks| P5
    D3 -->|Metadata| P5
    P5 -->|Ranked Sentences| P6
    P6 -->|Compressed Context| P7
    P7 -->|Prompt| LLM
    LLM -->|Response| P7
    P7 -->|Answer + Metrics| User
    
    style P1 fill:#0D47A1,stroke:#01579B,color:#fff
    style P2 fill:#0D47A1,stroke:#01579B,color:#fff
    style P3 fill:#E65100,stroke:#BF360C,color:#fff
    style P4 fill:#E65100,stroke:#BF360C,color:#fff
    style P5 fill:#E65100,stroke:#BF360C,color:#fff
    style P6 fill:#B71C1C,stroke:#880E4F,color:#fff
    style P7 fill:#B71C1C,stroke:#880E4F,color:#fff
```

## 4. Component Diagram

```mermaid
graph TB
    subgraph "Presentation Layer"
        UI[Streamlit Web UI]
    end
    
    subgraph "Application Layer"
        Controller[RAG Controller]
        SessionMgr[Session Manager]
        ErrorHandler[Error Handler]
    end
    
    subgraph "Business Logic Layer"
        subgraph "Ingestion Components"
            DocProc[Document Processor]
            SectDet[Section Detector]
            Chunker[Chunker]
        end
        
        subgraph "Retrieval Components"
            IntentDet[Intent Detector]
            Retriever[Retriever]
            Reranker[Reranker]
        end
        
        subgraph "Compression Components"
            EvidSel[Evidence Selector]
            BudgetMgr[Budget Manager]
            CompStrategy[Compression Strategy]
        end
        
        subgraph "Generation Components"
            LLMWrap[LLM Wrapper]
            AnswerGen[Answer Generator]
            CitationExt[Citation Extractor]
        end
    end
    
    subgraph "Data Access Layer"
        VectorStore[Vector Store Manager]
        MetadataStore[Metadata Store]
        ConfigStore[Config Store]
    end
    
    subgraph "Infrastructure Layer"
        subgraph "ML Models"
            EmbedModel[Sentence-BERT<br/>Embedding Model]
            SpacyModel[SpaCy NLP<br/>Model]
        end
        
        subgraph "Storage"
            FAISS[(FAISS Index)]
            JSONStore[(JSON Files)]
            YAML[(YAML Config)]
        end
        
        subgraph "External APIs"
            GeminiAPI[Google Gemini<br/>API]
        end
    end
    
    UI -->|HTTP| Controller
    Controller --> SessionMgr
    Controller --> ErrorHandler
    
    Controller --> DocProc
    Controller --> IntentDet
    Controller --> Retriever
    Controller --> EvidSel
    Controller --> LLMWrap
    
    DocProc --> SectDet
    DocProc --> Chunker
    Chunker --> EmbedModel
    
    IntentDet --> SpacyModel
    Retriever --> VectorStore
    Retriever --> EmbedModel
    Retriever --> Reranker
    
    EvidSel --> CompStrategy
    EvidSel --> BudgetMgr
    
    LLMWrap --> AnswerGen
    LLMWrap --> GeminiAPI
    AnswerGen --> CitationExt
    
    VectorStore --> FAISS
    MetadataStore --> JSONStore
    ConfigStore --> YAML
    
    style UI fill:#1565C0,stroke:#0D47A1,color:#fff
    style Controller fill:#2E7D32,stroke:#1B5E20,color:#fff
    style DocProc fill:#6A1B9A,stroke:#4A148C,color:#fff
    style IntentDet fill:#EF6C00,stroke:#E65100,color:#fff
    style EvidSel fill:#C62828,stroke:#B71C1C,color:#fff
    style LLMWrap fill:#00838F,stroke:#006064,color:#fff
```

## 5. Sequence Diagram - Query Processing Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant Controller as RAG Controller
    participant IntentDet as Intent Detector
    participant Retriever as Retriever
    participant VectorStore as FAISS Vector Store
    participant EvidSel as Evidence Selector
    participant BudgetMgr as Budget Manager
    participant LLM as LLM Wrapper
    participant GeminiAPI as Gemini API
    
    User->>UI: Enter Query
    UI->>Controller: process_query(query)
    
    Controller->>IntentDet: detect_intent(query)
    IntentDet->>IntentDet: analyze_keywords()
    IntentDet->>IntentDet: match_patterns()
    IntentDet-->>Controller: intent_type (e.g., METHOD)
    
    Controller->>Retriever: retrieve(query, intent)
    Retriever->>VectorStore: generate_embedding(query)
    VectorStore-->>Retriever: query_vector
    Retriever->>VectorStore: search(query_vector, top_k=20)
    VectorStore-->>Retriever: top_k_chunks
    
    Retriever->>Retriever: rerank(chunks, query)
    Retriever-->>Controller: ranked_chunks
    
    Controller->>EvidSel: select_evidence(chunks, intent)
    
    loop For each chunk
        EvidSel->>EvidSel: split_sentences()
        EvidSel->>EvidSel: score_relevance(sentence)
    end
    
    EvidSel-->>Controller: ranked_sentences
    
    Controller->>BudgetMgr: enforce_budget(sentences, max_tokens)
    BudgetMgr->>BudgetMgr: calculate_tokens(sentences)
    BudgetMgr->>BudgetMgr: truncate_to_budget()
    BudgetMgr-->>Controller: compressed_context
    
    Controller->>LLM: generate_answer(query, context)
    LLM->>LLM: build_prompt(query, context)
    LLM->>GeminiAPI: chat.send_message(prompt)
    
    GeminiAPI-->>LLM: response_text
    LLM->>LLM: extract_citations()
    LLM->>LLM: validate_response()
    LLM-->>Controller: Answer(text, citations, metrics)
    
    Controller-->>UI: display_results(answer)
    UI-->>User: Show Answer + Metrics
    
    Note over User,GeminiAPI: Total Processing Time: 2-4 seconds
```

## 6. Sequence Diagram - Document Ingestion Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant Controller as RAG Controller
    participant DocProc as Document Processor
    participant SectDet as Section Detector
    participant Chunker as Chunker
    participant EmbedGen as Embedding Generator
    participant VectorStore as FAISS Vector Store
    
    User->>UI: Upload PDF
    UI->>Controller: upload_document(file)
    
    Controller->>DocProc: process_document(pdf_file)
    DocProc->>DocProc: extract_text(pdf)
    DocProc->>DocProc: clean_text(raw_text)
    
    DocProc->>SectDet: detect_sections(text)
    SectDet->>SectDet: find_abstract()
    SectDet->>SectDet: find_methodology()
    SectDet->>SectDet: find_results()
    SectDet->>SectDet: find_references()
    SectDet-->>DocProc: sections_list
    
    DocProc->>Chunker: chunk_text(sections)
    
    loop For each section
        Chunker->>Chunker: split_by_sentences()
        Chunker->>Chunker: apply_overlap()
        Chunker->>Chunker: add_metadata(section, page)
    end
    
    Chunker-->>DocProc: chunks_with_metadata
    DocProc-->>Controller: processed_chunks
    
    Controller->>EmbedGen: generate_embeddings(chunks)
    
    loop For each chunk
        EmbedGen->>EmbedGen: encode(chunk.text)
    end
    
    EmbedGen-->>Controller: embeddings_array
    
    Controller->>VectorStore: add_embeddings(embeddings, metadata)
    VectorStore->>VectorStore: build_faiss_index()
    VectorStore->>VectorStore: store_metadata()
    VectorStore-->>Controller: index_created
    
    Controller-->>UI: ingestion_complete
    UI-->>User: Display Success Message
    
    Note over User,VectorStore: Processing time varies by document size
```

## 7. Deployment Diagram

```mermaid
graph TB
    subgraph UserDevice["User Device"]
        Browser["Web Browser
        Chrome/Firefox/Safari"]
    end
    
    subgraph AppServer["Application Server - Single Machine"]
        subgraph Container["Docker Container / Virtual Environment"]
            StreamlitApp["Streamlit Application
            Port 8501"]
            
            subgraph PyRuntime["Python Runtime"]
                RAGApp["RAG Application
                Python 3.9+"]
                
                subgraph MLModels["ML Models - In-Memory"]
                    SentBERT["Sentence-BERT
                    all-MiniLM-L6-v2"]
                    SpaCy["SpaCy
                    en_core_web_sm"]
                end
                
                subgraph VectorStoreMemory["Vector Store - In-Memory"]
                    FAISS["FAISS Index
                    L2 Distance"]
                end
            end
            
            subgraph FileSystem["File System"]
                Uploads["/uploads/
                PDF Files"]
                Indices["/indices/
                FAISS Binary"]
                Metadata["/metadata/
                JSON Files"]
                Config["/config/
                YAML Files"]
            end
        end
    end
    
    subgraph ExtServices["External Services - Cloud"]
        GeminiAPI["Google Gemini API
        gemini-1.5-pro
        HTTPS"]
        HuggingFace["HuggingFace Hub
        Model Downloads
        HTTPS"]
    end
    
    subgraph DevTools["Development Tools"]
        Git["Git Repository
        GitHub"]
        PyPI["PyPI
        Package Manager"]
    end
    
    Browser -->|HTTPS/HTTP| StreamlitApp
    StreamlitApp --> RAGApp
    RAGApp --> SentBERT
    RAGApp --> SpaCy
    RAGApp --> FAISS
    RAGApp --> Uploads
    RAGApp --> Indices
    RAGApp --> Metadata
    RAGApp --> Config
    
    RAGApp -->|REST API HTTPS| GeminiAPI
    SentBERT -.->|Model Download| HuggingFace
    SpaCy -.->|Model Download| HuggingFace
    
    RAGApp -.->|Source Code| Git
    RAGApp -.->|Dependencies| PyPI
    
    style Browser fill:#1565C0,stroke:#0D47A1,color:#fff
    style StreamlitApp fill:#C62828,stroke:#B71C1C,color:#fff
    style RAGApp fill:#2E7D32,stroke:#1B5E20,color:#fff
    style FAISS fill:#6A1B9A,stroke:#4A148C,color:#fff
    style GeminiAPI fill:#EF6C00,stroke:#E65100,color:#fff
    style SentBERT fill:#00838F,stroke:#006064,color:#fff
    style SpaCy fill:#00838F,stroke:#006064,color:#fff
```

## 8. Deployment Diagram - Production Architecture (Future State)

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[NGINX<br/>SSL Termination]
    end
    
    subgraph "Web Tier (Auto-scaled)"
        Web1[Streamlit Instance 1]
        Web2[Streamlit Instance 2]
        Web3[Streamlit Instance N]
    end
    
    subgraph "Application Tier (Kubernetes Cluster)"
        subgraph "Ingestion Service"
            Ingest1[Pod 1]
            Ingest2[Pod 2]
        end
        
        subgraph "Query Service"
            Query1[Pod 1]
            Query2[Pod 2]
            Query3[Pod 3]
        end
        
        subgraph "Compression Service"
            Comp1[Pod 1]
            Comp2[Pod 2]
        end
    end
    
    subgraph "Data Tier"
        VectorDB[(Vector DB<br/>Weaviate/Qdrant<br/>Replicated)]
        Cache[(Redis Cache<br/>Query Results)]
        ObjectStore[(S3/MinIO<br/>Document Storage)]
    end
    
    subgraph "Monitoring & Observability"
        Prometheus[Prometheus<br/>Metrics]
        Grafana[Grafana<br/>Dashboards]
        ELK[ELK Stack<br/>Logs]
    end
    
    subgraph "External Services"
        GeminiAPI[Google Gemini API]
        CloudStorage[Cloud Storage<br/>Backups]
    end
    
    Users((Users)) -->|HTTPS| LB
    LB --> Web1
    LB --> Web2
    LB --> Web3
    
    Web1 --> Ingest1
    Web2 --> Query1
    Web3 --> Query2
    
    Ingest1 --> VectorDB
    Ingest2 --> VectorDB
    
    Query1 --> Cache
    Query2 --> Cache
    Query3 --> Cache
    
    Cache -.->|Cache Miss| VectorDB
    
    Query1 --> Comp1
    Query2 --> Comp2
    Query3 --> Comp1
    
    Comp1 -->|API Call| GeminiAPI
    Comp2 -->|API Call| GeminiAPI
    
    Ingest1 --> ObjectStore
    
    VectorDB -.->|Backup| CloudStorage
    ObjectStore -.->|Backup| CloudStorage
    
    Web1 -.->|Metrics| Prometheus
    Query1 -.->|Metrics| Prometheus
    Comp1 -.->|Metrics| Prometheus
    
    Prometheus --> Grafana
    
    Web1 -.->|Logs| ELK
    Query1 -.->|Logs| ELK
    Comp1 -.->|Logs| ELK
    
    style LB fill:#C62828,stroke:#B71C1C,color:#fff
    style VectorDB fill:#6A1B9A,stroke:#4A148C,color:#fff
    style Cache fill:#EF6C00,stroke:#E65100,color:#fff
    style GeminiAPI fill:#2E7D32,stroke:#1B5E20,color:#fff
```

---

## Summary

This document contains 8 comprehensive diagrams:

1. **Use Case Diagram** - Shows user interactions and system boundaries
2. **Class Diagram** - Detailed object-oriented design with 20+ classes
3. **Data Flow Diagram (Level 0)** - High-level context diagram
4. **Data Flow Diagram (Level 1)** - Detailed process flows
5. **Component Diagram** - System components and their relationships
6. **Sequence Diagram (Query)** - Step-by-step query processing flow
7. **Sequence Diagram (Ingestion)** - Document processing workflow
8. **Deployment Diagram (Current)** - Single-server deployment architecture
9. **Deployment Diagram (Future)** - Microservices production architecture

These diagrams provide a complete architectural view of the Adaptive Context Compression RAG system from multiple perspectives.
