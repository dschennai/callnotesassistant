```mermaid
flowchart TD
    %% =========================
    %% HLD: High-Level Design Framework
    %% =========================

    subgraph U["👤 User (Creative Team Member)"]
        UI1[Project Dashboard]
        UI2[Creative Brief Review]
        UI3[Persona Selection]
        UI4[Optional Creative Hub]
        UI5[Parameter Review & Refinement]
        UI6[Final Asset Review]
    end

    subgraph F["🌐 Frontend (Next.js 15+, React 19)"]
        F1[Server Components & Actions]
        F2[Redux Toolkit + Tailwind UI]
        F3[MUI, React Select, Dropzone]
    end

    subgraph B["⚙️ Backend (FastAPI + Server Logic)"]
        B1[RESTful API Endpoints]
        B2[LLM Orchestration Engine]
        B3[Workflow State Machine]
        B4[Audit & Logging Layer]
    end

    subgraph D["🗄️ Data & Storage Layer"]
        D1[Firestore - Project Data Store]
        D2[Redis - Temp Chat Session Store]
        D3[GCS - Asset Storage]
    end

    subgraph AI["🧠 AI & Generative Services"]
        AI1[LLM (Gemini / OpenAI) - Text Tasks]
        AI2[Image Generator - Visual Assets]
    end

    %% User ↔ Frontend
    U -->|UI Interactions| F
    F -->|Server Actions / API Calls| B

    %% Backend ↔ Data
    B -->|Read/Write| D1
    B -->|Cache Sessions| D2
    B -->|Upload/Fetch Assets| D3

    %% Backend ↔ AI
    B -->|Prompt Requests| AI1
    AI1 -->|Generated Briefs, Personas, Prompts| B
    B -->|Final Image Prompt| AI2
    AI2 -->|Generated Image| B

    %% Data ↔ Frontend
    D1 -->|Project Data Sync| F
    D3 -->|Asset URLs| F

    %% Notes
    note right of AI1
        LLM Tasks Include:
        - Creative Brief Distillation
        - Persona Suggestion
        - Goal Setting
        - Moodboard Generation
        - Prompt Guide Compilation
        - Parameter Proposal & Chat
        - Final Image Prompt Synthesis
    end

    classDef layer fill:#eef6ff,stroke:#6fa8dc,stroke-width:1px,corner-radius:10px;
    classDef ai fill:#fff0f5,stroke:#d46a6a,stroke-width:1px,corner-radius:10px;
    classDef data fill:#f9f9f9,stroke:#999,stroke-width:1px,corner-radius:10px;

    class F,B,D layer;
    class AI ai;
    class D data;
