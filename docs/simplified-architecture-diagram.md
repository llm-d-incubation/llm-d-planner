# Simplified Architecture Diagram for Slides

## Option 1: High-Level Flow (Recommended for slides)

```mermaid
---
config:
  layout: fixed
  look: neo
---
flowchart LR
    A["💬 Chat UI"] --> B["🧠 Extract<br>Intent"]
    B --> C["📋 Generate<br>Specification"]
    B -.-> LLM["LLM Backend"]
    C --> D["✏️ Review &amp;<br>Edit Specs"]
    D --> E["🎯 Generate<br>Recommendations"]
    E --> F["🚀 Deploy"]
    E -.-> KB[("Knowledge Base<br>Benchmarks, SLOs,<br>47 Models")]
    F -.-> K8S["☸️ Kubernetes Cluster"]
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#fff4e1
    style LLM fill:#E1BEE7
    style D fill:#e1f5ff
    style E fill:#fff4e1
    style F fill:#e1ffe1
    style K8S fill:#BBDEFB
    style KB fill:#f0f0f0
```

## Option 2: Component View

```mermaid
graph TB
    subgraph "User Interface"
        UI[Chat Interface]
        SPEC[Specification Editor]
        SELECT[Recommendation Selector]
    end

    subgraph "Backend Services"
        INTENT[Intent Extraction]
        SPECGEN[Specification Service]
        REC[Recommendation Engine]
        YAML[YAML Generator]
    end

    subgraph "Data"
        KB[(Database<br/>Benchmarks<br/>Use Case SLOs<br/>47 Models)]
    end

    subgraph "Deployment"
        K8S[Kubernetes<br/>KServe + vLLM]
    end

    UI --> INTENT
    INTENT --> SPECGEN
    SPECGEN --> SPEC
    SPEC --> REC
    REC --> KB
    REC --> SELECT
    SELECT --> YAML
    YAML --> K8S

    style UI fill:#e1f5ff
    style SPEC fill:#e1f5ff
    style SELECT fill:#e1f5ff
    style INTENT fill:#fff4e1
    style SPECGEN fill:#fff4e1
    style REC fill:#fff4e1
    style KB fill:#f0f0f0
    style YAML fill:#e1ffe1
    style K8S fill:#e1ffe1
```

## Option 3: End-to-End Flow with Labels

```mermaid
flowchart LR
    A["👤 User<br/><i>Describe needs</i>"]
    B["🧠 Extract<br/><i>Intent</i>"]
    C["📋 Spec<br/><i>Generate</i>"]
    D["✏️ Review<br/><i>Edit specs</i>"]
    E["🎯 Recommend<br/><i>Model + GPU</i>"]
    F["📄 YAML<br/><i>Generate configs</i>"]
    G["☸️ Deploy<br/><i>Kubernetes</i>"]

    A --> B --> C --> D --> E --> F --> G

    KB[("📚 Database<br/>Benchmarks<br/>9 Use Cases<br/>47 Models")]
    E <-.-> KB

    style A fill:#fff
    style B fill:#fff4e1
    style C fill:#fff4e1
    style D fill:#e1f5ff
    style E fill:#fff4e1
    style F fill:#e1ffe1
    style G fill:#e1ffe1
    style KB fill:#f0f0f0
```

## Option 4: Vertical Stack (Best for portrait slides)

```mermaid
graph TB
    User["👤 User Input<br/>Natural language requirements"]

    subgraph "Planner"
        Chat["💬 Conversational Interface"]
        Intent["🧠 Intent Extraction"]
        Spec["📋 Specification Service<br/><small>Traffic Profile | SLO Targets | Quality Weights</small>"]
        Review["✏️ Review &amp; Edit Specification"]
        Rec["🎯 Recommendation Engine<br/><small>Model Selection | Capacity Planning | Scoring</small>"]
        KB[("📚 Database<br/><small>Benchmarks p95/ITL | 9 Use Case SLOs | 47 Models</small>")]
        Deploy["🚀 Deployment Automation<br/><small>YAML Generation | K8s Deployment</small>"]
    end

    K8S["☸️ Kubernetes Cluster"]

    User --> Chat
    Chat --> Intent
    Intent --> Spec
    Spec --> Review
    Review --> Rec
    Rec <--> KB
    Rec --> Deploy
    Deploy --> K8S

    style User fill:#fff
    style Chat fill:#e1f5ff
    style Intent fill:#fff4e1
    style Spec fill:#fff4e1
    style Review fill:#e1f5ff
    style Rec fill:#fff4e1
    style KB fill:#f0f0f0
    style Deploy fill:#e1ffe1
    style K8S fill:#e1ffe1
```

## Usage Instructions

1. **Copy the diagram you prefer** from above
2. **Go to https://mermaid.live**
3. **Paste the Mermaid code**
4. **Click "Actions" → "PNG" or "SVG"** to download
5. **Insert into Google Slides**

**Recommendations:**
- **Option 1** (High-Level Flow) - Best for executive summary
- **Option 3** (End-to-End Flow) - Best for showing complete user journey
- **Option 4** (Vertical Stack) - Best if you need portrait orientation

All diagrams are simplified to fit on a single slide while preserving the core concepts.

The pipeline matches the API:
`extract-intent → generate-specification → [review/edit] → generate-recommendations → generate-deployment → deploy-bundle-to-cluster`
