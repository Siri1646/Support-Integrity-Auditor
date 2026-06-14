# Architecture Diagram

```mermaid
flowchart TD
    A[Raw CRM Ticket Dataset] --> B[Pseudo-Label Engine]
    B --> C[Independent Severity Signals]

    C --> C1[Rule-Based NLP Signal]
    C --> C2[Resolution-Time Signal]
    C --> C3[Issue Category Signal]

    C1 --> D[Signal Fusion Layer]
    C2 --> D
    C3 --> D

    D --> E[Inferred Severity]
    E --> F[Compare with Assigned Priority]
    F --> G[Binary Mismatch Label]

    G --> H[Fine-Tuned DistilBERT Classifier]
    H --> I[Prediction: Consistent or Mismatch]

    I --> J[Evidence Dossier Generator]
    J --> K[Streamlit Dashboard]

    K --> L[Single Ticket Audit]
    K --> M[Batch CSV Audit]
    K --> N[Analytics and Heatmap]