```mermaid
flowchart TD
    %% Title
    A[Start: Client Request Received] --> B[Phase 1: Project Kick-Off & Information Gathering]

    %% Phase 1
    subgraph P1[Phase 1: Project Kick-Off & Information Gathering]
        B1[1. Receive and Formalize Request in Basecamp]
        B2[2. Locate Core Brand Assets from shared drives]
        B3[Exception: Missing or outdated assets - Notify Senior or Lead]
        B4[3. Review Client Insights document]
        B5[4. Confirm Technical Specs from Master Spec Sheet]
        B6[Challenges: Scattered information, time-consuming research]
    end
    B --> B1 --> B2 --> B3 --> B4 --> B5 --> C[Proceed to Phase 2]

    %% Phase 2
    subgraph P2[Phase 2: Strategic Deconstruction and Creative Briefing]
        C1[1. Analyze and interpret client brief]
        C2[2. Synthesize objective into one-sentence goal]
        C3[3. Understand target audience and negative personas]
        C4[4. Define call to action and desired emotion]
        C5[5. Develop and submit formal Creative Brief for approval]
        C6[Outcome: Approved Creative Brief becomes guiding document]
    end
    C --> C1 --> C2 --> C3 --> C4 --> C5 --> D[Proceed to Phase 3]

    %% Phase 3
    subgraph P3[Phase 3: Concepting and Visual Direction]
        D1[1. Create mood board using inspiration sources]
        D2[2. Create compositional sketches - thumbnails]
        D3[3. Present mood board and sketches for directional approval]
        D4[Outcome: Approved mood board defines visual direction]
    end
    D --> D1 --> D2 --> D3 --> E[Proceed to Phase 4]

    %% Phase 4
    subgraph P4[Phase 4: Execution Planning and Parameter Definition]
        E1[1. Identify specific visual parameters: shot, lighting, color, composition]
        E2[2. Conversational review with stakeholder - Document feedback in Basecamp]
        E3[3. Finalize execution plan and lock approved parameters]
        E4[Outcome: Finalized creative parameters become production prompt]
    end
    E --> E1 --> E2 --> E3 --> F[Proceed to Phase 5]

    %% Phase 5
    subgraph P5[Phase 5: Production and Iteration]
        F1[1. Asset sourcing or creation per finalized plan]
        F2[2. Review and minor iteration based on feedback]
        F3[3. Export final deliverables using standard naming convention]
        F4[Outcome: Approved visual asset ready for delivery]
    end
    F --> F1 --> F2 --> F3 --> G[End: Final Asset Delivered]

    %% Style and grouping
    classDef phase fill:#eaf5ff,stroke:#6fa8dc,stroke-width:1px;
    class P1,P2,P3,P4,P5 phase;
