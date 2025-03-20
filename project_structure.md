# MotorMind Project Structure

## Directory Structure

```
motormind/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI application
│   │   ├── endpoints/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py         # Authentication endpoints
│   │   │   ├── eeg_data.py     # EEG data upload/management endpoints
│   │   │   └── predictions.py  # Model prediction endpoints
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py      # Pydantic models for API
│   │   └── dependencies.py     # API dependencies
│   ├── database/
│   │   ├── __init__.py
│   │   ├── supabase.py         # Supabase client configuration
│   │   └── migrations/         # SQL migrations for Supabase setup
│   └── services/
│       ├── __init__.py
│       ├── auth_service.py     # Authentication logic
│       ├── eeg_service.py      # EEG data processing logic
│       └── llm_service.py      # LLM integration service
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Auth/           # Authentication components
│   │   │   ├── Dashboard/      # User dashboard components
│   │   │   ├── EEGViewer/      # EEG data visualization
│   │   │   └── Predictions/    # Model predictions display
│   │   ├── hooks/
│   │   │   ├── useAuth.js      # Authentication hook
│   │   │   └── useSupabase.js  # Supabase client hook
│   │   ├── pages/
│   │   ├── services/
│   │   │   └── api.js          # API service for backend
│   │   ├── App.js
│   │   └── index.js
│   ├── package.json
│   └── README.md
├── ml/
│   ├── data/
│   │   ├── preprocessing/
│   │   │   ├── __init__.py
│   │   │   ├── filters.py      # EEG filtering functions
│   │   │   └── features.py     # Feature extraction for LLM input
│   │   └── loaders/
│   │       ├── __init__.py
│   │       └── eeg_loader.py   # Data loading utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llm_wrapper.py      # LLM integration code
│   │   ├── feature_processor.py # Processing features for LLM
│   │   └── evaluation.py       # Model evaluation utilities
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py            # Training script
│   │   └── fine_tune.py        # LLM fine-tuning script
│   └── inference/
│       ├── __init__.py
│       └── predict.py          # Inference script
├── notebooks/
│   ├── data_exploration.ipynb  # Data exploration
│   ├── feature_extraction.ipynb # Feature extraction experiments
│   └── model_evaluation.ipynb  # Model evaluation
├── tests/
│   ├── backend/
│   │   └── test_api.py
│   ├── ml/
│   │   └── test_models.py
│   └── conftest.py
├── .env.example                # Example environment variables
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker setup
├── README.md                   # Project README
└── setup.py                    # Package setup
```

## Key Components

### 1. Supabase Integration

The project will use Supabase for:

- **Authentication**: User management and authentication with JWT tokens
- **Database**: PostgreSQL database for:
  - EEG raw data storage with TimescaleDB extension for time-series optimization
  - User profiles and metadata
  - Model results and performance metrics
  - Research collaboration data
- **Storage**: For larger EEG datasets, experiment results, and model weights
- **Edge Functions**: For lightweight processing and API integrations
- **Realtime**: For collaborative research and monitoring

### 2. EEG Data Processing Pipeline

1. **Data Collection**: Interface with EEG hardware (support for multiple devices)
2. **Preprocessing**: Filtering, artifact removal, epoching
3. **Feature Extraction**: Extracting relevant features for LLM analysis 
4. **Data Storage**: Storing processed data in Supabase

### 3. LLM Integration

1. **Feature-to-Text Conversion**: Converting EEG features to a format digestible by LLMs
2. **Fine-tuning**: Adapting a pre-trained LLM to understand EEG data patterns
3. **Inference Pipeline**: Using the LLM to classify and interpret EEG signals
4. **Explanability**: Extracting reasoning steps from LLM outputs

### 4. Web Application

1. **User Dashboard**: Visualization of EEG data and model predictions
2. **Data Management**: Upload, annotation, and management of EEG datasets
3. **Model Interaction**: Interfacing with the LLM for real-time predictions
4. **Research Collaboration**: Tools for sharing datasets and findings

## Database Schema (Supabase)

### Tables

1. **users**
   - Managed by Supabase Auth

2. **eeg_datasets**
   ```sql
   CREATE TABLE eeg_datasets (
     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
     user_id UUID REFERENCES auth.users(id),
     name TEXT NOT NULL,
     description TEXT,
     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
     updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
     metadata JSONB,
     is_public BOOLEAN DEFAULT FALSE
   );
   ```

3. **eeg_recordings**
   ```sql
   CREATE TABLE eeg_recordings (
     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
     dataset_id UUID REFERENCES eeg_datasets(id),
     user_id UUID REFERENCES auth.users(id),
     recording_date TIMESTAMP WITH TIME ZONE,
     duration INTEGER,  -- in seconds
     channels INTEGER,
     sampling_rate INTEGER,
     metadata JSONB,
     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
   );
   ```

4. **eeg_data** (Using TimescaleDB for time-series optimization)
   ```sql
   CREATE TABLE eeg_data (
     time TIMESTAMP WITH TIME ZONE NOT NULL,
     recording_id UUID REFERENCES eeg_recordings(id),
     channel INTEGER,
     value FLOAT,
     PRIMARY KEY (recording_id, time, channel)
   );
   -- Convert to hypertable
   SELECT create_hypertable('eeg_data', 'time');
   ```

5. **model_predictions**
   ```sql
   CREATE TABLE model_predictions (
     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
     recording_id UUID REFERENCES eeg_recordings(id),
     user_id UUID REFERENCES auth.users(id),
     model_version TEXT NOT NULL,
     prediction JSONB NOT NULL,
     explanation TEXT,
     confidence FLOAT,
     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
   );
   ```

6. **research_projects**
   ```sql
   CREATE TABLE research_projects (
     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
     name TEXT NOT NULL,
     description TEXT,
     owner_id UUID REFERENCES auth.users(id),
     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
     updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
     is_public BOOLEAN DEFAULT FALSE
   );
   ```

7. **project_members**
   ```sql
   CREATE TABLE project_members (
     project_id UUID REFERENCES research_projects(id),
     user_id UUID REFERENCES auth.users(id),
     role TEXT NOT NULL,
     joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
     PRIMARY KEY (project_id, user_id)
   );
   ```

## API Endpoints

### Authentication
- POST /api/auth/register
- POST /api/auth/login
- POST /api/auth/refresh
- POST /api/auth/logout

### EEG Data
- GET /api/datasets
- POST /api/datasets
- GET /api/datasets/{id}
- PUT /api/datasets/{id}
- DELETE /api/datasets/{id}
- POST /api/datasets/{id}/recordings
- GET /api/recordings/{id}
- POST /api/recordings/{id}/analyze

### Model Predictions
- POST /api/predict
- GET /api/predictions/{id}
- GET /api/users/{id}/predictions

### Research Collaboration
- GET /api/projects
- POST /api/projects
- PUT /api/projects/{id}
- POST /api/projects/{id}/members
- GET /api/projects/{id}/datasets 