"""
LLM4ED: Agentic PDE Discovery System (4-Agent Architecture)
============================================================

4 Agents:
1. Differential Observer - Converts raw data into physics-aware representations (derivatives + contour plots)
2. Phenomenology Extractor - Identifies dominant physical regime from data (VLM)
3. Governing Law Synthesizer - Generates diverse candidate governing equations
4. Equation Arbiter - Selects the most plausible equation with confidence score

Workflow: Agent 1 → Agent 2 → Agent 3 → Agent 4 → Decision Gate (if confidence < 80%, loop back)

NO manual operator/operand library - LLMs generate equations AND fitting code.
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import json
try:
    import scipy.io as scipy_io
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
import os
from typing import Dict, List, Any
import base64
from pathlib import Path
import re
import tempfile
import subprocess
import sys

# AutoGen imports
try:
    import autogen
    from autogen import GroupChat, GroupChatManager, AssistantAgent, UserProxyAgent, Agent
    import logging
    logging.getLogger('autogen.oai.client').setLevel(logging.ERROR)
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    print("AutoGen not available. Install with: pip install pyautogen")


class AgenticPDEDiscovery:
    """4-Agent PDE Discovery System - Fully Agentic (no hardcoded fitting)"""
    
    def __init__(self, config_path):
        self.config_dir = os.path.dirname(os.path.abspath(config_path))
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.datasets = {}
        self.discovered_equations = {}
        self._setup_agents()
    
    def _resolve_path(self, rel_path):
        """Resolve relative path from config directory"""
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.normpath(os.path.join(self.config_dir, rel_path))

    def _default_contour_path(self, dataset_key: str) -> str:
        """Default contour output path inside the workspace."""
        return self._resolve_path(f"Contours/{dataset_key}_contour.png")
        
    def _setup_agents(self):
        """Setup 4-agent system: VLM, LLM_Generator, Engineer, Admin"""
        if not AUTOGEN_AVAILABLE:
            return
            
        try:
            llm_config = self.config.get('llm_config', {})
            
            # LLM config for text agents
            text_llm_config = {
                "config_list": [{
                    "model": llm_config.get('primary_model', 'qwen/qwen3-coder-480b-a35b-instruct'),
                    "base_url": llm_config.get('api_base', 'https://integrate.api.nvidia.com/v1'),
                    "api_key": llm_config.get('api_key', os.getenv('NVIDIA_API_KEY'))
                }]
            }
            
            # VLM config for vision agent
            vlm_config = {
                "config_list": [{
                    "model": llm_config.get('vision_model', 'microsoft/phi-4-multimodal-instruct'),
                    "base_url": llm_config.get('api_base', 'https://integrate.api.nvidia.com/v1'),
                    "api_key": llm_config.get('api_key', os.getenv('NVIDIA_API_KEY')),
                    "price": [0, 0]
                }]
            }
            
            def termination_check(msg):
                return "<<FINAL_RESULT>>" in str(msg.get("content", ""))
            
            # Agent 1: Differential Observer (orchestrator + data processing)
            self.admin = autogen.UserProxyAgent(
                name="Differential_Observer",
                system_message="You are the Differential Observer. You convert raw spatio-temporal data into physics-aware representations (derivatives and contour plots). Pass processed data to other agents and collect results.",
                is_termination_msg=termination_check,
                human_input_mode="NEVER",
                code_execution_config=False,
            )
            
            # Agent 2: Phenomenology Extractor - Identifies dominant physical regime
            self.vlm_agent = autogen.AssistantAgent(
                name="Phenomenology_Extractor",
                llm_config=vlm_config,
                system_message="""You are the Phenomenology Extractor - a Vision-Language Model expert in identifying dominant physical regimes from PDE data.

Your job: Analyze the contour plot to identify SPECIFIC physical patterns and output a SYMBOLIC PATTERN DESCRIPTOR.

LOOK FOR THESE PATTERNS:

1. TRAVELING FRONTS:
   - S-shaped or sigmoid wave moving in one direction
   - Sharp transition between two states (e.g., 0 to 1)
   - Constant wave speed, maintains shape
   - Amplitude bounded between 0 and 1 (or similar)
   - KEY: Look for QUADRATIC saturation (u, u**2 terms)

2. SHOCK/STEEP GRADIENTS :
   - Steep gradient that forms from smooth initial condition
   - NO oscillations or ripples near the shock
   - Smooth on one side, sharp drop on other (N-wave or sawtooth)
   - Shock gets sharper then diffuses/smooths over time

3. SOLITONS/DISPERSION:
   - Localized POSITIVE humps (peaks) moving at constant speed
   - OSCILLATORY tails or small ripples trailing the main peak
   - Multiple peaks may exist but maintain separation
   - Peaks are smooth, not sharp discontinuities

4. CHAOTIC PATTERNS:
   - Irregular, turbulent-looking patterns
   - No clear repeating structure
   - Complex spatiotemporal behavior

5. BISTABLE/REACTION-DIFFUSION :
   - Solution settles to distinct stable states (around +1, -1, or 0)
   - Transition layers (sharp fronts) between stable regions
   - Amplitude BOUNDED - doesn't grow indefinitely
   - May show spatial patterns with distinct "islands" or regions
   - KEY: Look for cubic saturation (u - u³ terms)
   - NO traveling waves or shocks - more like equilibrium patterns

6. PURE DIFFUSION:
   - Smooth Gaussian-like spreading
   - Amplitude decreasing over time

CRITICAL DISTINCTION:
- If you see TRAVELING FRONTS with 0→1 transitions and S-shape → Fisher-KPP (reaction-diffusion with u, u**2)
- If you see BOUNDED amplitude with TRANSITION LAYERS between stable states (±1) → Chafee-Infante (reaction-diffusion with u, u**3)
- If you see TRAVELING structures or shocks → Burgers/KdV (convection with u*u_x)
- If you see CHAOTIC spatiotemporal patterns → KS (u_xx, u_xxxx, u*u_x)

OUTPUT FORMAT:
PATTERN TYPE: [which pattern you see]
KEY OBSERVATIONS:
- [specific detail 1]
- [specific detail 2]
- [specific detail 3]
SUGGESTED TERMS: [which derivative terms match: u_xx, u*u_x, u_xxx, u, u**2, u**3, etc.]"""
            )
            
            # Agent 3: Governing Law Synthesizer - Generates diverse candidate equations
            self.llm_agent = autogen.AssistantAgent(
                name="Governing_Law_Synthesizer",
                llm_config=text_llm_config,
                system_message="""You are the Governing Law Synthesizer - an expert in partial differential equations.

Your job: Generate 10 DIVERSE candidate PDEs that could govern the observed data based on the pattern descriptor from the Phenomenology Extractor.

You WILL be shown (1) a contour plot and (2) numeric data samples (u and derivatives at points).
Use both to propose candidates that actually match the data scale and behavior.


IMPORTANT CONSTRAINTS:
- Only use these symbols on the RHS: u, u_x, u_xx, u_xxx, u_xxxx
- Allowed operators: +, -, *, /, **, parentheses
- Write coefficients explicitly (the Engineer will re-fit them).
- For reaction terms, use u**3 NOT u^3

FORMAT:
CANDIDATE 1: u_t = c1*u_xx + c2*u + c3*u**3
  Reason: Chafee-Infante form - mandatory candidate for reaction-diffusion

... continue to CANDIDATE 10

MOST LIKELY: [list 1-3 candidates that best match VLM observations]"""
            )
            
            # Agent 4: Equation Arbiter - Selects the most plausible equation
            self.engineer_agent = autogen.AssistantAgent(
                name="Equation_Arbiter",
                llm_config=text_llm_config,
                system_message="""You are the Equation Arbiter - a Python engineer who evaluates and selects the most plausible PDE.

Your job: Given candidate PDEs and derivative data, write Python code that fits and evaluates each candidate.

OUTPUT YOUR CODE in a ```python block.

YOUR CODE MUST:
1. Load the .npy files using the EXACT paths provided in the prompt (u.npy, u_t.npy, u_x.npy, u_xx.npy, u_xxx.npy, u_xxxx.npy)
2. Flatten all arrays to 1D using .flatten()
3. For EACH of the 10 candidates from Governing_Law_Synthesizer:
   - Build a design matrix X from the relevant terms (e.g., for "u_xx + u*u_x", use [u_xx, u*u_x])
   - Fit coefficients using np.linalg.lstsq(X, u_t, rcond=None)
   - Compute prediction: pred = X @ coeffs
   - Calculate R² = 1 - sum((u_t - pred)²) / sum((u_t - mean(u_t))²)
4. PARSIMONY: Among candidates with R² >= 0.95, prefer the one with FEWEST terms.
   - Count terms in each candidate (e.g., "u_xx + u - u**3" has 3 terms)
   - If multiple candidates have R² > 0.95, select the one with fewer terms
   - This avoids overfitting with unnecessary terms
5. Print results in this EXACT format:
   <<FINAL_RESULT>>
   BEST_PDE: u_t = [fitted equation with coefficients]
   CONFIDENCE: [R² value as decimal, e.g., 0.995000]
   ALL_RESULTS:
   1. [candidate] -> R²=[value]
   ... (all 10)
   <<END_RESULT>>

TERM CONSTRUCTION EXAMPLES:
- For term "u*u_x": use the array u*u_x
- For term "u**3": use the array u**3  
- For term "u_xx": use the array u_xx directly

Use ONLY numpy (import numpy as np). No other libraries."""
            )
            
            # Speaker selection: Differential_Observer -> Phenomenology_Extractor -> Governing_Law_Synthesizer -> Equation_Arbiter -> Differential_Observer
            def speaker_flow(last_speaker: Agent, groupchat):
                if len(groupchat.messages) == 0:
                    return self.vlm_agent
                    
                last_name = groupchat.messages[-1]["name"]
                
                if last_name == "Differential_Observer":
                    return self.vlm_agent
                elif last_name == "Phenomenology_Extractor":
                    return self.llm_agent
                elif last_name == "Governing_Law_Synthesizer":
                    return self.engineer_agent
                elif last_name == "Equation_Arbiter":
                    return self.admin
                else:
                    return self.admin
            
            # Create group chat with all 4 agents
            self.groupchat = GroupChat(
                agents=[self.admin, self.vlm_agent, self.llm_agent, self.engineer_agent],
                messages=[],
                max_round=4,  # Admin->VLM->LLM->Engineer->stop
                speaker_selection_method=speaker_flow,
            )
            
            self.manager = GroupChatManager(
                groupchat=self.groupchat, 
                llm_config=text_llm_config
            )
            
        except Exception as e:
            print(f"Agent setup failed: {e}")
            self.groupchat = None

    def load_dataset(self, dataset_key):
        """Load dataset from config"""
        if dataset_key not in self.config['datasets']:
            return None
        
        config = self.config['datasets'][dataset_key]
        
        if 'data_files' in config and 'solution' in config['data_files']:
            file_path = self._resolve_path(config['data_files']['solution'])
            if (not os.path.exists(file_path)) and ('backup_solution' in config['data_files']):
                backup_path = self._resolve_path(config['data_files']['backup_solution'])
                if os.path.exists(backup_path):
                    file_path = backup_path
        else:
            file_path = config.get('file_path', '')
            if file_path:
                file_path = self._resolve_path(file_path)
            
        if not file_path:
            return None
        
        try:
            if file_path.endswith('.mat'):
                if not SCIPY_AVAILABLE:
                    print("SciPy is required to load .mat files. Install with: pip install scipy")
                    return None

                data = scipy_io.loadmat(file_path)
                mat_keys = config.get('mat_keys', {})
                
                if mat_keys:
                    u_key = mat_keys.get('u', 'usol')
                    x_key = mat_keys.get('x', 'x')
                    t_key = mat_keys.get('t', 't')
                    
                    if isinstance(u_key, list):
                        for key in u_key:
                            if key in data:
                                u = np.real(data[key])
                                break
                    else:
                        u = np.real(data[u_key])
                    
                    x = np.real(data[x_key]).flatten()
                    t = np.real(data[t_key]).flatten()
                else:
                    largest_array = None
                    max_size = 0
                    for key, value in data.items():
                        if isinstance(value, np.ndarray) and value.ndim == 2:
                            if value.size > max_size:
                                largest_array = value
                                max_size = value.size
                    u = np.real(largest_array)
                    x = np.linspace(0, 1, u.shape[0])
                    t = np.linspace(0, 1, u.shape[1])
                
                # Check if pre-computed derivatives exist in the .mat file
                precomputed_derivs = {}
                for deriv_key, deriv_name in [('U_t', 'u_t'), ('U_x', 'u_x'), ('U_xx', 'u_xx')]:
                    if deriv_key in data:
                        precomputed_derivs[deriv_name] = np.real(data[deriv_key])
                if precomputed_derivs:
                    print(f"  [INFO] Found pre-computed derivatives in .mat file: {list(precomputed_derivs.keys())}")
                    
            elif file_path.endswith('.npy'):
                u = np.real(np.load(file_path))
                precomputed_derivs = {}  # No pre-computed derivatives for .npy files
                if 'data_files' in config:
                    if 'x_grid' in config['data_files']:
                        x = np.real(np.load(self._resolve_path(config['data_files']['x_grid'])))
                        t = np.real(np.load(self._resolve_path(config['data_files']['t_grid'])))
                    else:
                        x = np.linspace(0, 1, u.shape[0])
                        t = np.linspace(0, 1, u.shape[1])
                else:
                    x = np.linspace(0, 1, u.shape[0])
                    t = np.linspace(0, 1, u.shape[1])
            
            self.datasets[dataset_key] = {'u': u, 'x': x, 't': t, 'config': config, 'precomputed_derivs': precomputed_derivs}
            return True
            
        except Exception as e:
            print(f"Failed to load {dataset_key}: {e}")
            return None

    def _compute_derivative_fields(
        self,
        u: np.ndarray,
        x: np.ndarray,
        t: np.ndarray,
        spatial_method: str = 'spectral',
    ) -> Dict[str, np.ndarray]:
        """Compute u_t and spatial derivatives up to 4th order."""
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        dt = t[1] - t[0] if len(t) > 1 else 1.0

        # Time derivative
        u_t = np.gradient(u, dt, axis=1, edge_order=2)

        spatial_method = (spatial_method or 'spectral').strip().lower()

        if spatial_method == 'spectral':
            # Spectral differentiation along x (assumes uniform grid)
            nx = u.shape[0]
            k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
            u_hat = np.fft.fft(u, axis=0)

            def d_dx(order: int) -> np.ndarray:
                factor = (1j * k) ** order
                return np.real(np.fft.ifft(factor[:, None] * u_hat, axis=0))

            u_x = d_dx(1)
            u_xx = d_dx(2)
            u_xxx = d_dx(3)
            u_xxxx = d_dx(4)
        else:
            u_x = np.gradient(u, dx, axis=0, edge_order=2)
            u_xx = np.gradient(u_x, dx, axis=0, edge_order=2)
            u_xxx = np.gradient(u_xx, dx, axis=0, edge_order=2)
            u_xxxx = np.gradient(u_xxx, dx, axis=0, edge_order=2)

        return {
            'u': u,
            'u_t': u_t,
            'u_x': u_x,
            'u_xx': u_xx,
            'u_xxx': u_xxx,
            'u_xxxx': u_xxxx,
        }

    def _make_data_samples_for_llm(
        self,
        fields: Dict[str, np.ndarray],
        x: np.ndarray,
        t: np.ndarray,
        n_points: int = 12,
        seed: int = 0,
    ) -> List[Dict[str, float]]:
        """Provide a tiny set of representative numeric samples for the LLM."""
        rng = np.random.default_rng(seed)
        nx, nt = fields['u'].shape

        # Avoid boundaries where derivatives are noisiest
        ix = rng.integers(2, max(3, nx - 2), size=n_points)
        it = rng.integers(2, max(3, nt - 2), size=n_points)

        samples: List[Dict[str, float]] = []
        for i, j in zip(ix, it):
            samples.append({
                'x': float(x[i]) if i < len(x) else float(i),
                't': float(t[j]) if j < len(t) else float(j),
                'u': float(np.real(fields['u'][i, j])),
                'u_t': float(np.real(fields['u_t'][i, j])),
                'u_x': float(np.real(fields['u_x'][i, j])),
                'u_xx': float(np.real(fields['u_xx'][i, j])),
                'u_xxx': float(np.real(fields['u_xxx'][i, j])),
                'u_xxxx': float(np.real(fields['u_xxxx'][i, j])),
            })
        return samples

    def _extract_candidate_pdes(self) -> List[str]:
        """Extract RHS expressions from the Governing Law Synthesizer agent output."""
        if not hasattr(self, 'groupchat') or self.groupchat is None:
            return []

        rhs_list: List[str] = []
        for msg in self.groupchat.messages:
            if msg.get('name') != 'Governing_Law_Synthesizer':
                continue
            content = msg.get('content', '')
            if not isinstance(content, str):
                continue

            # Match: CANDIDATE k: u_t = ...
            for m in re.finditer(r"CANDIDATE\s*\d+\s*:\s*u_t\s*=\s*([^\n\r]+)", content, flags=re.IGNORECASE):
                rhs = m.group(1).strip()
                rhs_list.append(rhs)

            # Fallback: any line with u_t = ...
            if not rhs_list:
                for m in re.finditer(r"u_t\s*=\s*([^\n\r]+)", content, flags=re.IGNORECASE):
                    rhs_list.append(m.group(1).strip())

        # Deduplicate while preserving order
        seen = set()
        out: List[str] = []
        for rhs in rhs_list:
            rhs_norm = rhs.replace('−', '-').strip()
            if rhs_norm not in seen:
                out.append(rhs_norm)
                seen.add(rhs_norm)
        return out[:10]

    def _extract_engineer_code(self) -> str:
        """Extract Python code from Equation Arbiter agent output."""
        if not hasattr(self, 'groupchat') or self.groupchat is None:
            return ""

        for msg in self.groupchat.messages:
            if msg.get('name') != 'Equation_Arbiter':
                continue
            content = msg.get('content', '')
            if not isinstance(content, str):
                continue

            # Extract code block
            match = re.search(r"```python\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            
            # Try without language specifier
            match = re.search(r"```\s*(.*?)```", content, re.DOTALL)
            if match:
                return match.group(1).strip()

        return ""

    def _extract_final_result(self, output: str) -> Dict[str, Any]:
        """Parse the <<FINAL_RESULT>> block from Equation Arbiter's code output."""
        result = {
            'equation': '',
            'confidence': 0.0,
            'all_results': []
        }
        
        # Find the FINAL_RESULT block
        match = re.search(r"<<FINAL_RESULT>>(.*?)<<END_RESULT>>", output, re.DOTALL)
        if not match:
            # Try to find partial results
            match = re.search(r"<<FINAL_RESULT>>(.*)", output, re.DOTALL)
        
        if not match:
            return result
            
        block = match.group(1)
        
        # Extract BEST_PDE
        pde_match = re.search(r"BEST_PDE:\s*(.+?)(?:\n|$)", block)
        if pde_match:
            raw_equation = pde_match.group(1).strip()
            # Clean up the equation by pruning small coefficients
            result['equation'] = self._prune_small_coefficients(raw_equation)
        
        # Extract CONFIDENCE
        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", block)
        if conf_match:
            result['confidence'] = float(conf_match.group(1))
        
        # Extract ALL_RESULTS
        all_match = re.search(r"ALL_RESULTS:(.*?)(?:<<|$)", block, re.DOTALL)
        if all_match:
            for line in all_match.group(1).strip().split('\n'):
                line = line.strip()
                if line and '->' in line:
                    result['all_results'].append(line)
        
        return result

    def _prune_small_coefficients(self, equation: str, relative_threshold: float = 0.001) -> str:
        """Remove terms with coefficients smaller than threshold relative to the largest coefficient.
        
        Uses relative threshold: a term is kept if |coeff| >= relative_threshold * |max_coeff|
        This ensures important terms like u_xxx in KdV are not removed even if numerically small.
        Also removes terms with 0 coefficient and simplifies coefficient display.
        """
        if not equation:
            return equation
        
        # Remove "u_t = " prefix if present
        eq_body = equation
        prefix = ""
        if "u_t" in equation and "=" in equation:
            parts = equation.split("=", 1)
            prefix = parts[0].strip() + " = "
            eq_body = parts[1].strip()
        
        # Use regex to find all terms: coeff*variable or just variable
        # Pattern matches: optional sign, optional coefficient with *, then variable expression
        term_pattern = r'([+-]?)\s*(\d+\.?\d*(?:e[+-]?\d+)?)\s*\*\s*([a-zA-Z_][a-zA-Z0-9_*\s]*)|([+-]?)\s*([a-zA-Z_][a-zA-Z0-9_*\s]*)'
        
        # First, let's parse the equation more carefully
        # Replace minus signs that are between terms with a delimiter we can split on
        # But keep minus signs that are part of numbers (like -0.5)
        eq_body = re.sub(r'\s+-\s+', ' MINUS ', eq_body)
        eq_body = re.sub(r'\s+\+\s+', ' PLUS ', eq_body)
        
        # Split by our delimiters
        parts = re.split(r'\s+(PLUS|MINUS)\s+', eq_body)
        
        # Parse terms with their signs
        parsed_terms = []
        next_sign = 1  # positive
        
        for part in parts:
            part = part.strip()
            if part == 'PLUS':
                next_sign = 1
                continue
            elif part == 'MINUS':
                next_sign = -1
                continue
            elif not part:
                continue
                
            # Try to extract coefficient and variable
            # Pattern: coefficient * variable_expression
            match = re.match(r'^([+-]?\d+\.?\d*(?:e[+-]?\d+)?)\s*\*\s*(.+)$', part, re.IGNORECASE)
            if match:
                coeff_str = match.group(1)
                var_expr = match.group(2).strip()
                try:
                    coeff = float(coeff_str) * next_sign
                    # Handle nested coefficients like "0.4*u*u_x" in var_expr
                    nested_match = re.match(r'^(\d+\.?\d*(?:e[+-]?\d+)?)\s*\*\s*(.+)$', var_expr)
                    while nested_match:
                        coeff *= float(nested_match.group(1))
                        var_expr = nested_match.group(2).strip()
                        nested_match = re.match(r'^(\d+\.?\d*(?:e[+-]?\d+)?)\s*\*\s*(.+)$', var_expr)
                    parsed_terms.append((coeff, var_expr))
                except ValueError:
                    parsed_terms.append((next_sign, part))
            else:
                # No coefficient, just variable
                if part.startswith('-'):
                    parsed_terms.append((-1.0 * next_sign, part[1:].strip()))
                elif part.startswith('+'):
                    parsed_terms.append((1.0 * next_sign, part[1:].strip()))
                else:
                    parsed_terms.append((1.0 * next_sign, part))
            
            next_sign = 1  # Reset sign after processing term
        
        if not parsed_terms:
            return equation
        
        # Find maximum coefficient magnitude
        max_coeff = max(abs(c) for c, _ in parsed_terms) if parsed_terms else 1.0
        threshold = relative_threshold * max_coeff
        
        # Filter terms and format
        kept_terms = []
        for coeff, var_expr in parsed_terms:
            # Skip terms with coefficient effectively 0
            if abs(coeff) < 1e-9:
                continue
            # Skip terms below threshold
            if abs(coeff) < threshold:
                continue
                
            # Format the coefficient
            rounded = round(coeff)
            if abs(rounded) > 0 and abs(coeff - rounded) < 0.05:
                # Close to an integer
                coeff_int = int(rounded)
                if coeff_int == 1:
                    kept_terms.append((1.0, var_expr))
                elif coeff_int == -1:
                    kept_terms.append((-1.0, var_expr))
                else:
                    kept_terms.append((float(coeff_int), var_expr))
            else:
                kept_terms.append((coeff, var_expr))
        
        if not kept_terms:
            return equation  # Return original if nothing left
        
        # Reconstruct equation
        result_parts = []
        for i, (coeff, var_expr) in enumerate(kept_terms):
            if abs(coeff - 1.0) < 0.001:
                if i == 0:
                    result_parts.append(var_expr)
                else:
                    result_parts.append(f"+ {var_expr}")
            elif abs(coeff + 1.0) < 0.001:
                result_parts.append(f"-{var_expr}")
            elif coeff < 0:
                result_parts.append(f"{coeff:.4g}*{var_expr}")
            else:
                if i == 0:
                    result_parts.append(f"{coeff:.4g}*{var_expr}")
                else:
                    result_parts.append(f"+ {coeff:.4g}*{var_expr}")
        
        result = ' '.join(result_parts)
        # Clean up double spaces and "- -" patterns
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\+ -', '- ', result)
        result = re.sub(r'- -', '+ ', result)
        
        return prefix + result

    def _run_engineer_code(self, code: str, data_dir: str) -> str:
        """Execute the Engineer's generated code and capture output."""
        # Write code to temp file
        code_file = os.path.join(data_dir, "engineer_fit.py")
        with open(code_file, 'w') as f:
            f.write(code)
        
        # Run the code
        try:
            result = subprocess.run(
                [sys.executable, code_file],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=data_dir
            )
            output = result.stdout + "\n" + result.stderr
            return output
        except subprocess.TimeoutExpired:
            return "ERROR: Code execution timed out"
        except Exception as e:
            return f"ERROR: {str(e)}"

    def discover_pde(self, dataset_key):
        """Main discovery using 4-agent system (fully agentic)"""
        
        # Validate agents are ready
        llm_cfg = self.config.get('llm_config', {}) if isinstance(self.config, dict) else {}
        api_key = (llm_cfg.get('api_key') or os.getenv('NVIDIA_API_KEY') or '').strip()
        agents_ready = (
            AUTOGEN_AVAILABLE
            and bool(api_key)
            and hasattr(self, 'groupchat')
            and (self.groupchat is not None)
        )

        if not AUTOGEN_AVAILABLE:
            raise RuntimeError(
                "AutoGen is not available but agentic mode is required. "
                "Install with: pip install pyautogen"
            )
        if not api_key:
            raise RuntimeError(
                "NVIDIA API key missing. Set NVIDIA_API_KEY in your environment "
                "or provide llm_config.api_key in datasets_config.yaml."
            )
        if not agents_ready:
            raise RuntimeError(
                "Agent system not initialized correctly (groupchat missing). "
                "Check AutoGen installation and llm_config settings."
            )
        
        if dataset_key not in self.datasets:
            if not self.load_dataset(dataset_key):
                print(f"Failed to load {dataset_key}")
                return None
        
        data = self.datasets[dataset_key]
        u, x, t = data['u'], data['x'], data['t']
        config = data['config']
        precomputed_derivs = data.get('precomputed_derivs', {})
        
        # ===== AGENT 1: Differential Observer - Data Processing + Contour Generation =====
        # Always save contour to permanent Contours/ folder for inspection
        contour_dir = self._resolve_path("Contours")
        os.makedirs(contour_dir, exist_ok=True)
        contour_path = os.path.join(contour_dir, f"{dataset_key}_contour.png")
        
        # Always regenerate contour plot to ensure it's up-to-date
        try:
            plt.figure(figsize=(8, 6))
            extent = [float(np.min(x)), float(np.max(x)), float(np.min(t)), float(np.max(t))]
            plt.imshow(np.real(u).T, aspect='auto', origin='lower', extent=extent, cmap='viridis')
            plt.colorbar(label='u(x,t)')
            plt.xlabel('x (space)')
            plt.ylabel('t (time)')
            plt.title(f"Contour Plot: {dataset_key}\nAmplitude range: [{float(np.min(u)):.3f}, {float(np.max(u)):.3f}]")
            plt.tight_layout()
            plt.savefig(contour_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[Agent 1 - Differential Observer] Generated contour plot: {contour_path}")
        except Exception as e:
            print(f"Contour plot generation failed for {dataset_key}: {e}")
            return None
        
        # Compute derivative fields - use precomputed if available
        if precomputed_derivs and 'u_t' in precomputed_derivs and 'u_xx' in precomputed_derivs:
            print(f"[Agent 1 - Differential Observer] Using pre-computed derivatives from data file...")
            # Pre-computed derivatives are usually on interior grid (boundaries removed)
            # Adjust u to match the interior grid
            u_interior = u[1:-1, 1:-1] if precomputed_derivs['u_t'].shape != u.shape else u
            x_interior = x[1:-1] if precomputed_derivs['u_t'].shape[0] == len(x) - 2 else x
            t_interior = t[1:-1] if precomputed_derivs['u_t'].shape[1] == len(t) - 2 else t
            
            # Build fields dict with precomputed values
            fields = {
                'u': u_interior,
                'u_t': precomputed_derivs['u_t'],
                'u_x': precomputed_derivs.get('u_x', np.gradient(u_interior, x_interior[1]-x_interior[0] if len(x_interior) > 1 else 1.0, axis=0, edge_order=2)),
                'u_xx': precomputed_derivs['u_xx'],
                'u_xxx': np.gradient(precomputed_derivs['u_xx'], x_interior[1]-x_interior[0] if len(x_interior) > 1 else 1.0, axis=0, edge_order=2),
                'u_xxxx': np.zeros_like(u_interior),  # Not needed for most PDEs
            }
            # Use interior grid for samples
            u, x, t = u_interior, x_interior, t_interior
        else:
            print(f"[Agent 1 - Differential Observer] Computing derivatives using spectral method...")
            spatial_method = 'spectral'  # Use spectral for all (better accuracy)
            fields = self._compute_derivative_fields(u, x, t, spatial_method=spatial_method)
        
        samples = self._make_data_samples_for_llm(fields, x, t, n_points=12, seed=0)
        print(f"[Agent 1 - Differential Observer] Derivatives computed. Passing to agents...")
        
        # Save derivative data to temp directory for Engineer agent
        data_dir = tempfile.mkdtemp(prefix="pde_discovery_")
        np.save(os.path.join(data_dir, "u.npy"), fields['u'])
        np.save(os.path.join(data_dir, "u_t.npy"), fields['u_t'])
        np.save(os.path.join(data_dir, "u_x.npy"), fields['u_x'])
        np.save(os.path.join(data_dir, "u_xx.npy"), fields['u_xx'])
        np.save(os.path.join(data_dir, "u_xxx.npy"), fields['u_xxx'])
        np.save(os.path.join(data_dir, "u_xxxx.npy"), fields['u_xxxx'])
        
        # Create multimodal message with image
        b64_img = base64.b64encode(Path(contour_path).read_bytes()).decode("utf-8")
        
        # Build prompt for agents
        prompt = f"""Analyze this PDE data to discover the governing equation.

DATA CHARACTERISTICS:
- Grid size: {u.shape[0]} x {u.shape[1]} (space x time)
- Amplitude range: [{float(np.min(u)):.4f}, {float(np.max(u)):.4f}]
- Amplitude std: {float(np.std(u)):.4f}

=== NUMERIC DATA SAMPLES (random interior points) ===
Each row is one (x,t) point with u and derivatives.
{json.dumps(samples, indent=2)}

The contour plot shows u(x,t) where x is space (horizontal) and t is time (vertical).

COMMON PDE FORMS TO CONSIDER:
- KdV: u_t = c1*u*u_x + c2*u_xxx (dispersive waves with solitons)
- Burgers: u_t = c1*u*u_x + c2*u_xx (shock waves)
- Heat/Diffusion: u_t = c1*u_xx (diffusion)
- Kuramoto-Sivashinsky: u_t = c1*u_xx + c2*u_xxxx + c3*u*u_x
- Chafee-Infante: u_t = c1*u_xx + c2*u + c3*u**3

WORKFLOW:
1. VLM_Analyzer: Analyze the contour plot and identify physical patterns
2. LLM_PDE_Generator: Generate 10 candidate PDEs based on patterns - INCLUDE the common forms above with coefficients to fit
3. Engineer: Write Python code using least squares regression to fit each candidate and find the best one

DATA FILES FOR ENGINEER (already saved):
- {data_dir}/u.npy
- {data_dir}/u_t.npy
- {data_dir}/u_x.npy
- {data_dir}/u_xx.npy
- {data_dir}/u_xxx.npy
- {data_dir}/u_xxxx.npy

Engineer: Load these files, flatten them to 1D, and evaluate ALL candidates using R² metric."""

        # Run the agent conversation
        acceptance_threshold = float(self.config.get('discovery_settings', {}).get('acceptance_threshold', 0.80))
        max_rounds = int(self.config.get('discovery_settings', {}).get('max_candidate_rounds', 5))
        
        best_result = None
        
        for round_idx in range(max_rounds):
            print(f"\n{'='*60}")
            print(f"DISCOVERING PDE FOR: {dataset_key} (round {round_idx+1}/{max_rounds})")
            print(f"{'='*60}")
            
            # Build round-specific prompt
            round_prompt = prompt
            if round_idx > 0 and best_result:
                round_prompt += f"\n\nPrevious best had confidence {best_result['confidence']:.0%}, which is below {acceptance_threshold:.0%}. Generate 10 NEW different candidates."
            
            seed_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": round_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}", "detail": "high"}},
                ],
            }
            
            self.groupchat.reset()
            self.admin.initiate_chat(self.manager, message=seed_msg)
            print(f"{'='*60}\n")
            
            # Extract and run Engineer's code
            engineer_code = self._extract_engineer_code()
            result = None
            
            if engineer_code:
                print("Running Engineer's fitting code...")
                code_output = self._run_engineer_code(engineer_code, data_dir)
                print(code_output)
                
                # Parse results
                result = self._extract_final_result(code_output)
            
            if result and result.get('equation'):
                confidence = result['confidence']
                print(f"\nROUND {round_idx+1}: Best PDE = {result['equation']} | Confidence = {confidence:.0%}")
                
                if best_result is None or confidence > best_result['confidence']:
                    best_result = result
                
                if confidence >= acceptance_threshold:
                    print(f"✓ Accepted (confidence >= {acceptance_threshold:.0%})")
                    break
                else:
                    print(f"✗ Below threshold ({acceptance_threshold:.0%}), trying again...")
            else:
                print("WARNING: Could not parse results from Engineer's code output")
        
        # Cleanup temp directory
        import shutil
        try:
            shutil.rmtree(data_dir)
        except:
            pass
        
        if best_result and best_result['equation']:
            return {
                'equation': best_result['equation'],
                'confidence': best_result['confidence'],
                'accepted': best_result['confidence'] >= acceptance_threshold,
                'acceptance_threshold': acceptance_threshold,
                'method': 'Fully agentic: VLM + LLM + Engineer (code generation)',
                'all_results': best_result.get('all_results', [])
            }
        else:
            return {
                'equation': 'Discovery failed',
                'confidence': 0.0,
                'accepted': False,
                'method': 'Fully agentic (failed)',
            }
    
    def run_discovery(self, dataset_keys=None):
        """Run discovery on multiple datasets"""
        if dataset_keys is None:
            dataset_keys = list(self.config['datasets'].keys())
        
        results = {}
        for key in dataset_keys:
            result = self.discover_pde(key)
            if result:
                results[key] = result
                self.discovered_equations[key] = result
        
        # Summary
        print("\n" + "="*60)
        print("DISCOVERY RESULTS")
        print("="*60)
        for key, result in results.items():
            print(f"\n{key}:")
            print(f"  Equation: {result['equation']}")
            print(f"  Confidence: {result['confidence']:.0%}")
        
        return results


def predict_pde(dataset_name):
    """Simple interface for PDE prediction"""
    discovery = AgenticPDEDiscovery('datasets_config.yaml')
    
    if discovery.load_dataset(dataset_name):
        result = discovery.discover_pde(dataset_name)
        
        if result:
            print(f"\n{'='*50}")
            print(f"FINAL PREDICTION: {result['equation']}")
            print(f"Confidence: {result['confidence']:.0%}")
            print(f"{'='*50}")
            return result
    
    print(f"Failed to predict PDE for {dataset_name}")
    return None


if __name__ == "__main__":
    result = predict_pde('kdv_data')
