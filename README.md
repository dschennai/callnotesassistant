```mermaid
graph TD
    A[User Logs In] --> B{Project Dashboard};
    B --> C[Start New Project];
    B --> D[Resume Existing Project];

    C --> E[Phase 1: Project Setup];
    E -- Submit Form --> F((System: Create Project & Synthesize Constraints));
    F --> G[Phase 2: Creative Brief Review];
    G -- Save Brief & Proceed --> H((System: Save Brief & Analyze Personas));
    H --> I[Phase 2.5: Persona Selection];
    I -- Accept Persona & Proceed --> J((System: Save Persona));
    J --> B;

    D --> K{Route to Correct Phase based on Project Status};
    K --> G;
    K --> I;
    K --> L[Optional Creative Hub];
    K --> M[Final Constraint Review];
    K --> N[Phase 5A: Parameter Proposal Review];
    K --> O[Phase 6: Final Asset Review];


    B -- Project Status: awaiting_optional_input --> L;
    L --> L1{Goal Setting};
    L --> L2{Messaging & Composition};
    L --> L3{Visual Style Development};
    L --> L4[Resume Main Workflow];

    L1 -- Submit --> B;
    L2 -- Submit --> B;
    L3 -- Finalize Mood Board --> B;

    L4 --> M;
    M -- Lock Constraints & Compile --> P((System: Compile Prompt Guide));
    P --> N;

    N --> Q["Approve All & Proceed"];
    N --> R["Initiate Refinement Chat"];

    Q --> S((System: Finalize Parameters & Generate Asset));

    R --> T[Phase 5B: Interactive Refinement];
    T -- Finalize Parameters --> S;

    S --> O;
    O --> U["Complete Image Workflow"];
    O --> V["Request Rebrief"];

    U --> W((System: Upload to GCS & Set Status to Complete));
    W --> B;

    V -- Submit Feedback --> P;
