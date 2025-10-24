```mermaid
flowchart TD
    subgraph U["User - Creative Team Member"]
        UI1[Project Dashboard]
        UI2[Creative Brief Review]
        UI3[Persona Selection]
        UI4[Creative Hub - Optional]
        UI5[Parameter Review and Refinement]
        UI6[Final Asset Review]
    end

    subgraph F["Frontend - Next.js 15 and React 19"]
        F1[Server Components and Actions]
        F2[Redux Toolkit and Tailwind UI]
        F3[MUI, React Select, Dropzone]
    end

    subgraph B["Backend - FastAPI and Server Logic"]
        B1[RESTful API Endpoints]
        B2[LLM Orchestration Engine]
        B3[Workflow State Machine]
        B4[Audit and Logging Layer]
    end

    subgraph D["Data and Storage Layer"]
        D1[Firestore - Project Data Store]
        D2[Redis - Temporary Chat Session Store]
        D3[Google Cloud Storage - Asset Repository]
    end

    subgraph AI["AI and Generative Services"]
        AI1[LLM for Text Tasks such as Briefs, Personas, and Prompts]
        AI2[Image Generator for Visual Assets]
    end

    %% User ↔ Frontend
    U -->|User Interactions| F
    F -->|Server Actions or API Calls| B

    %% Backend ↔ Data
    B -->|Read or Write Data| D1
    B -->|Cache Sessions| D2
    B -->|Upload or Fetch Assets| D3

    %% Backend ↔ AI
    B -->|Prompt Requests| AI1
    AI1 -->|Generated Briefs and Prompts| B
    B -->|Image Prompt| AI2
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
        - Parameter Proposal and Chat
        - Final Image Prompt Synthesis
    end

    classDef layer fill:#eef6ff,stroke:#6fa8dc,stroke-width:1px,corner-radius:8px;
    classDef ai fill:#fff0f5,stroke:#d46a6a,stroke-width:1px,corner-radius:8px;
    classDef data fill:#f9f9f9,stroke:#999,stroke-width:1px,corner-radius:8px;

    class F,B,D layer;
    class AI ai;
    class D data;
