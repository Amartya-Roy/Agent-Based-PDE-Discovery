# Agents4ED: Agentic PDE Discovery System

A **fully autonomous** 4-agent system for discovering partial differential equations (PDEs) from spatiotemporal data. The system uses Large Language Models (LLMs) and Vision-Language Models (VLMs) to identify governing equations without any hardcoded operator libraries.

## Architecture

The system uses 4 specialized AI agents in a collaborative pipeline:

```
┌─────────────────────────┐
│  1. Differential        │  Converts raw data → derivatives + contour plots
│     Observer            │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  2. Phenomenology       │  VLM analyzes contour → identifies physics patterns
│     Extractor (VLM)     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  3. Governing Law       │  LLM generates 10 candidate PDEs based on patterns
│     Synthesizer (LLM)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  4. Equation Arbiter    │  LLM writes Python code to fit & evaluate candidates
│     (Engineer LLM)      │  Selects best PDE using R² + parsimony
└───────────┬─────────────┘
            │
            ▼
     [If R² < 80%: Loop back to Agent 3]
```
---

## Requirements

### Python Version
- Python 3.9+ recommended

### Dependencies
```bash
pip install numpy scipy matplotlib pyyaml pyautogen openai
```

### API Key (Required)
The system uses NVIDIA's API for LLM and VLM inference. You need an NVIDIA API key.

**Option 1: Environment Variable (Recommended)**
```bash
# macOS/Linux
export NVIDIA_API_KEY="nvapi-your-api-key-here"

# Windows (Command Prompt)
set NVIDIA_API_KEY=nvapi-your-api-key-here

# Windows (PowerShell)
$env:NVIDIA_API_KEY="nvapi-your-api-key-here"
```

**Option 2: In config file**
Edit `datasets_config.yaml` and set:
```yaml
llm_config:
  api_key: "nvapi-your-api-key-here"
```

---

## Quick Start

### macOS / Linux

```bash
# 1. Navigate to the project folder
cd "/Users/...."

# 2. Create virtual environment (first time only)
python3 -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Install dependencies
pip install numpy scipy matplotlib pyyaml pyautogen openai

# 5. Set API key
export NVIDIA_API_KEY="nvapi-your-api-key-here"

# 6. Run PDE discovery
python -c "from pde_discovery_system import predict_pde; predict_pde('kdv_data')"
```

### Windows (Command Prompt)

```cmd
:: 1. Navigate to the project folder
cd "C:\path\to\New Code"

:: 2. Create virtual environment (first time only)
python -m venv .venv

:: 3. Activate virtual environment
.venv\Scripts\activate.bat

:: 4. Install dependencies
pip install numpy scipy matplotlib pyyaml pyautogen openai

:: 5. Set API key
set NVIDIA_API_KEY=nvapi-your-api-key-here

:: 6. Run PDE discovery
python -c "from pde_discovery_system import predict_pde; predict_pde('kdv_data')"
```

### Windows (PowerShell)

```powershell
# 1. Navigate to the project folder
cd "C:\path\to\New Code"

# 2. Create virtual environment (first time only)
python -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install numpy scipy matplotlib pyyaml pyautogen openai

# 5. Set API key
$env:NVIDIA_API_KEY="nvapi-your-api-key-here"

# 6. Run PDE discovery
python -c "from pde_discovery_system import predict_pde; predict_pde('kdv_data')"
```

---

## Available Datasets

The following datasets are pre-configured:

| Dataset Key | PDE Type | Description |
|-------------|----------|-------------|
| `kdv_data` | Korteweg-de Vries | Soliton waves |
| `burgers_data` | Burgers | Shock waves |
| `chafee_infante_data` | Chafee-Infante | Reaction-diffusion (cubic) |
| `KS_data` | Kuramoto-Sivashinsky | Chaotic dynamics |
| `fisher_data` | Fisher-KPP | Population dynamics |

### Run on existing datasets

```python
from pde_discovery_system import predict_pde

# Discover PDE for any dataset
result = predict_pde('kdv_data')           # KdV equation
result = predict_pde('burgers_data')       # Burgers equation
result = predict_pde('chafee_infante_data') # Chafee-Infante
result = predict_pde('KS_data')            # Kuramoto-Sivashinsky
result = predict_pde('fisher_data')        # Fisher-KPP
```

Or run all datasets:
```python
from pde_discovery_system import AgenticPDEDiscovery

discovery = AgenticPDEDiscovery('datasets_config.yaml')
results = discovery.run_discovery()  # Runs on all datasets
```

---

## Adding New Data

### Supported Formats
- `.mat` (MATLAB files) - requires scipy
- `.npy` (NumPy arrays)

### Data Requirements
Your data should contain:
1. **Solution field `u(x,t)`**: 2D array with shape `(nx, nt)` where:
   - Rows = spatial points
   - Columns = time points
2. **Spatial grid `x`**: 1D array of spatial coordinates
3. **Temporal grid `t`**: 1D array of time coordinates

### Option A: Add to config file

Edit `datasets_config.yaml`:

```yaml
datasets:
  # ... existing datasets ...
  
  my_new_data:
    name: "My Custom Dataset"
    description: "Description of my PDE data"
    data_files:
      solution: "Data/my_data.mat"
      contour_plot: "Contours/my_data_contour.png"
    mat_keys:
      u: "usol"    # Variable name containing solution in .mat file
      x: "x"       # Variable name for spatial grid
      t: "t"       # Variable name for time grid
    physics_domain: "general"
```

For `.npy` files:
```yaml
  my_npy_data:
    name: "My NPY Dataset"
    description: "Data stored as numpy arrays"
    data_files:
      solution: "Data/my_solution.npy"
      x_grid: "Data/my_x.npy"
      t_grid: "Data/my_t.npy"
      contour_plot: "Contours/my_npy_contour.png"
    file_type: "npy"
    physics_domain: "general"
```

Then run:
```python
from pde_discovery_system import predict_pde
result = predict_pde('my_new_data')
```

### Option B: Programmatic loading

```python
from pde_discovery_system import AgenticPDEDiscovery
import numpy as np

# Create discovery instance
discovery = AgenticPDEDiscovery('datasets_config.yaml')

# Manually add dataset
discovery.datasets['my_custom_data'] = {
    'u': my_solution_array,      # Shape: (nx, nt)
    'x': my_spatial_grid,        # Shape: (nx,)
    't': my_time_grid,           # Shape: (nt,)
    'config': {'name': 'My Custom'},
    'precomputed_derivs': {}     # Optional: pre-computed derivatives
}

# Run discovery
result = discovery.discover_pde('my_custom_data')
```

### Pre-computed Derivatives (Optional)

If your `.mat` file contains pre-computed derivatives, the system will automatically detect and use them:

```matlab
% In MATLAB, save derivatives along with solution:
save('my_data.mat', 'U', 'x', 't', 'U_t', 'U_x', 'U_xx');
```

The system looks for variables named `U_t`, `U_x`, `U_xx` in `.mat` files.

---

## Project Structure

```
New Code/
├── pde_discovery_system.py    # Main system (4-agent architecture)
├── datasets_config.yaml       # Configuration file
├── README.md                  # This file
├── Data/                      # Dataset files
│   ├── Kdv.mat
│   ├── burgers.mat
│   ├── kuramoto_sivishinky.mat
│   ├── fisher_nonlin_groundtruth.mat
│   ├── chafee_infante_CI.npy
│   ├── chafee_infante_x.npy
│   └── chafee_infante_t.npy
├── Contours/                  # Generated contour plots
│   └── [dataset]_contour.png
└── .venv/                     # Python virtual environment (optional)
```

---

## Configuration Options

Edit `datasets_config.yaml` to customize:

```yaml
# Discovery parameters
discovery_settings:
  acceptance_threshold: 0.80    # Minimum R² to accept equation (0-1)
  max_candidate_rounds: 5       # Max retry rounds if below threshold
  subsample_factor: 4           # Data subsampling (not used in current version)
  max_terms: 3                  # Preferred max terms in equation

# LLM models
llm_config:
  primary_model: "qwen/qwen3-coder-480b-a35b-instruct"  # Text LLM
  vision_model: "microsoft/phi-4-multimodal-instruct"   # Vision LLM
  api_base: "https://integrate.api.nvidia.com/v1"
  temperature: 0.7
  max_tokens: 2000
```

---


