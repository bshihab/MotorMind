# MotorMind Project Structure

This document outlines the structure of the MotorMind project, highlighting the organization of its components.

## Overview

MotorMind is organized into a modular structure that separates concerns and promotes clean architecture:

```
motormind/
├── ml/               # Machine learning components
├── backend/          # Server-side and database components
├── examples/         # Example scripts and demos
└── docs/             # Documentation files
```

## Machine Learning (`ml/`)

The `ml/` directory contains all components related to processing EEG data and applying machine learning models:

```
ml/
├── data/             # Data handling
│   └── preprocessing/  # EEG data preparation
│       └── features.py # Feature extraction for EEG data
├── models/           # ML model implementations
│   ├── llm_wrapper.py  # Base LLM integration
│   └── langchain_wrapper.py  # LangChain integration
├── training/         # Training scripts and utilities
└── inference/        # Inference/prediction code
```

Key files:
- `ml/data/preprocessing/features.py`: Extracts features from raw EEG data
- `ml/models/llm_wrapper.py`: Core integration with LLMs
- `ml/models/langchain_wrapper.py`: Enhanced LLM integration using LangChain

## Backend (`backend/`)

The `backend/` directory houses server-side components and database integrations:

```
backend/
├── api/              # API endpoints
├── database/         # Database connections and schemas
│   └── supabase.py   # Supabase client and utilities
└── services/         # Business logic services
```

Key files:
- `backend/database/supabase.py`: Client for interacting with Supabase

## Examples (`examples/`)

The `examples/` directory contains demonstration scripts:

```
examples/
├── eeg_llm_demo.py         # Demo for base LLM integration
└── eeg_llm_langchain_demo.py  # Demo for LangChain integration
```

Key files:
- `examples/eeg_llm_demo.py`: Demonstrates basic EEG-LLM integration
- `examples/eeg_llm_langchain_demo.py`: Showcases LangChain capabilities

## Documentation (`docs/`)

The `docs/` directory contains project documentation:

```
docs/
├── project_structure.md       # This file
├── eeg_llm_architecture.md    # Architecture overview
└── langchain_integration.md   # Details on LangChain integration
```

## Dependency Flow

The project's components interact in the following way:

1. Raw EEG data is processed in `ml/data/preprocessing/`
2. Features are extracted and formatted for LLM analysis
3. LLM analysis is performed through either:
   - Direct API calls via `ml/models/llm_wrapper.py`
   - Enhanced processing via `ml/models/langchain_wrapper.py`
4. Results are stored and managed in Supabase via `backend/database/supabase.py`
5. Example scripts in `examples/` demonstrate the full pipeline

## LangChain Integration

The LangChain integration builds on the existing architecture:

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│   EEG Features    │────▶│  LangChainWrapper │────▶│ Structured Output │
│   (features.py)   │     │(langchain_wrapper)│     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
                               │        ▲
                               │        │
                               ▼        │
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│   Supabase DB     │◀───▶│ LangChainSupabase │     │ LangChain Models  │
│  (supabase.py)    │     │   Integration     │────▶│  and Utilities    │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

The LangChain integration maintains compatibility with the existing codebase while adding enhanced capabilities for LLM interaction.