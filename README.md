# Socratic-Zero: Bootstrapping Reasoning via Data-Free Agent Co-evolution

**Paper**: [http://arxiv.org/abs/2509.24726](http://arxiv.org/abs/2509.24726)

## Abstract

Socratic-Zero is a fully autonomous framework that generates high-quality training data from minimal seed examples through the co-evolution of three agents: the **Teacher**, the **Solver**, and the **Generator**. Starting from only 100 seed questions, our Socratic-Solver-8B achieves an average gain of +20.2 percentage points over prior data synthesis methods across seven mathematical reasoning benchmarks, while synthetic data from Socratic-Generator-32B enables student LLMs to achieve superior performance compared to state-of-the-art commercial LLMs.

## 1. Introduction

Recent breakthroughs in large language models (LLMs) on reasoning tasks rely heavily on massive, high-quality datasets—typically human-annotated and thus difficult to scale. Socratic-Zero addresses this limitation by implementing a paradigm-shifting framework that eliminates dependency on large-scale external datasets while enabling truly autonomous reasoning improvement.

Inspired by the Socratic method of learning through questioning, our approach implements co-evolution between three agents:
- **Solver**: Learns to reason and solve mathematical problems
- **Teacher**: Acts as an oracle for evaluation and strategic problem generation  
- **Generator**: Learns to distill and scale the Teacher's problem generation strategy

## 2. System Architecture

### 2.1 The Socratic-Zero Framework

The system operates as a self-improving loop among three agents, formalizing reasoning improvement as an adaptive curriculum learning problem:

```
Questions → Solver → Teacher → DPO Triplets → Next Round
    ↓         ↓         ↓           ↓            ↓
Collection  Grading  Cartesian   Parquet     Weight Update
```

### 2.2 Agent Definitions

1. **Solver (𝒮)**: An agent with policy π_θ_𝒮 that maps problems to solution trajectories, improving through preference feedback on successful and failed attempts.

2. **Teacher (𝒯)**: A fixed, high-capacity LLM providing:
   - Verification function V(q, y) → {0, 1} for solution correctness
   - Problem refinement function G(q, y_fail) → (q', y'_ref) for curriculum generation

3. **Generator (𝒢)**: An agent with policy π_θ_𝒢 that learns to mimic the Teacher's refinement strategy, generating optimally challenging problems for the current Solver.

### 2.3 Project Structure

```
ProSetting/
├── scripts/
│   ├── run_training.py           # Unified training orchestrator
│   ├── auto_trainer.py           # Fully automated training pipeline
│   └── semi_auto_trainer.py      # Interactive training controller
├── collectors/                   # Data collection subsystem
│   ├── trajectory_collector.py   # Multi-GPU trajectory generation
│   └── data_normalizer.py        # Data standardization utilities
├── processors/                   # Data processing subsystem
│   ├── reward_calculator.py      # Teacher-based reward computation
│   ├── question_enhancer.py      # Progressive question generation
│   └── solver_data_processor.py  # Training data preparation
├── datasets/                     # Dataset management subsystem
│   ├── dpo_data_converter.py     # DPO format conversion
│   └── data_saver.py             # Persistent storage management
├── trainers/                     # Training execution subsystem
│   ├── trl_trainer.py            # TRL-based training implementation
│   └── gpu_manager.py            # Resource management utilities
├── managers/                     # System management subsystem
│   ├── round_controller.py       # Multi-round training coordination
│   └── question_manager.py       # Question pool management
└── core/                         # Core system utilities
    └── state_manager.py          # Training state persistence
```

## 3. Methodology

### 3.1 Solver Training via Online Preference Optimization

The Solver improves through Direct Preference Optimization (DPO), leveraging the Teacher's verification function to create preference pairs from correct and incorrect solutions:

```python
L_DPO(θ_S; θ_ref) = -E[log σ(β log π_θ_S(y_w|q)/π_θ_ref(y_w|q) - β log π_θ_S(y_l|q)/π_θ_ref(y_l|q))]
```

### 3.2 Generator Training via Value-Weighted Distillation

The Generator learns to produce optimally challenging problems using a utility function that scores problems based on the Solver's success rate:

```python
U(q'|π_θ_S) = exp(-(s_q' - μ)²/2σ²)
```

Where μ=0.5 targets problems at the frontier of the Solver's capabilities.

### 3.3 Progressive Training Strategy

- **Rounds 1-2**: Data accumulation phase without model updates
- **Round 3+**: Active training with progressive weight transfer
- **Question Pool Evolution**: Systematic expansion through teacher-guided enhancement
- **Failure Recovery**: Automatic replay of failed questions with enhanced variants

## 4. Experimental Results

### 4.1 Solver Performance

Our Socratic-Solver-8B achieves remarkable improvements across seven mathematical reasoning benchmarks:

| Benchmark | Baseline | Socratic-Zero | Improvement |
|-----------|----------|---------------|-------------|
| AMC-23    | 45.8%    | 63.7%        | +17.9      |
| Minerva   | 41.9%    | 52.4%        | +10.5      |
| MATH-500  | 62.7%    | 81.2%        | +18.5      |
| GSM8K     | 74.6%    | 87.3%        | +12.7      |
| Olympiad  | 35.9%    | 55.1%        | +19.2      |
| AIME-25   | 11.4%    | 24.6%        | +13.2      |
| AIME-24   | 12.3%    | 28.4%        | +16.1      |
| **Average** | **40.7%** | **56.1%** | **+15.4** |

### 4.2 Generator Effectiveness

Our Socratic-Generator-32B demonstrates superior data generation capabilities:

- **Validity Rate**: 95.6% (vs 89.1% baseline)
- **Downstream Utility**: 37.72% average accuracy
- **Competitive Performance**: Outperforms much larger commercial models including GPT-5, Gemini-2.5-Pro, and Claude-4.1-Opus

### 4.3 Cross-Architecture Generalization

The framework shows consistent effectiveness across different model architectures:
- **GLM4-9B**: +17.1 points improvement
- **Qwen3-14B**: +17.3 points improvement
- **Transfer to General Reasoning**: +6.02 points on BBEH, MMLU-Pro, SuperGPQA

## 5. Quick Start

### 5.1 Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with key parameters:
# SOLVER_MODEL_PATH=/path/to/solver/model
# QUESTIONS_FILE=/path/to/questions.json
# WORKSPACE_DIR=/path/to/workspace
# TEACHER_BASE_URL=http://your-teacher-api

# Verify environment
python utils/status_checker.py --quick
```

### 5.2 Training Execution

#### Fully Automated Training (Recommended)
```bash
python scripts/auto_trainer.py
```

#### Interactive Training
```bash
python scripts/semi_auto_trainer.py
```

### 5.3 System Configuration

```python
DEFAULT_CONFIG = {
    "max_rounds": 5,                    # Training iterations
    "save_rounds": [3, 4, 5],          # Checkpoint persistence
    "attempts_per_question": 8,         # Solution attempts per question
    "training_framework": "TRL_DPO",   # Optimization method
    "trl_num_processes": 8,            # Parallel training processes
    "trl_mixed_precision": "bf16"      # Numerical precision
}
```

## 6. Key Features

### 6.1 Multi-Agent Co-Evolution
- **Dynamic Curriculum**: Problems adapt to Solver's evolving capabilities
- **Strategic Generation**: Teacher creates targeted challenges based on failure analysis
- **Scalable Distillation**: Generator learns to produce high-quality problems autonomously

### 6.2 Automated Weight Transfer
- **Inter-round Continuity**: Automatic model weight propagation between training rounds
- **FSDP Integration**: Distributed weight merging for large-scale models
- **Checkpoint Recovery**: Robust state persistence and recovery mechanisms

### 6.3 Quality Control
- **Dual Verification**: MathRule + LLM judge for reliable evaluation
- **Teacher Self-Verification**: Automatic quality checks for generated problems
- **Feedback-Driven Monitoring**: Continuous curriculum quality assessment

## 7. Training Pipeline

### 7.1 Five-Stage Training Process

1. **Data Collection**: Multi-GPU parallel solver trajectory collection
2. **Data Grading**: Teacher batch grading with 32 concurrent processing
3. **DPO Conversion**: Convert grading results to preference triplets
4. **TRL Training**: Distributed training using accelerate + TRL
5. **Next Round Preparation**: Build enhanced question pool for next iteration

### 7.2 Curriculum Evolution

The system implements zone-adaptive problem generation:
- **Mastered Zone**: Problems consistently solved (success rate = 1.0)
- **Learning Zone**: Problems intermittently solved (0 < success rate < 1.0)  
- **Too Difficult Zone**: Problems consistently failed (success rate = 0)

New problems are strategically generated from Mastered and Learning zones to maintain optimal challenge levels.

## 8. Performance Monitoring

### 8.1 Training Metrics
- **Convergence Analysis**: Oscillatory patterns with bounded performance fluctuations
- **Reward Progression**: Stable ~50% high-reward problem generation
- **Cross-Architecture Consistency**: Robust improvements across model families

### 8.2 System Diagnostics
```bash
# Comprehensive system status
python utils/status_checker.py

# Quick health check  
python utils/status_checker.py --quick

# Training pipeline validation
python utils/test_runner.py
```

## 9. Troubleshooting

### 9.1 Common Issues

**Model Path Configuration**:
```bash
export SOLVER_MODEL_PATH="/path/to/model"
```

**GPU Memory Management**:
- Monitor usage: `nvidia-smi`
- Adjust batch sizes and parallelism parameters

**Training Recovery**:
```bash
# Automated recovery
python scripts/auto_trainer.py

# Manual state inspection
python utils/status_checker.py
```

### 9.2 Log Analysis
- **Training Logs**: `/tmp/trl_trainer.log`, `/tmp/auto_trainer.log`
- **State Files**: `{WORKSPACE_DIR}/training_state.json`
- **Progress Tracking**: `{WORKSPACE_DIR}/round_XX_progress.json`

## 10. Citation

If you use Socratic-Zero in your research, please cite:

```bibtex
@article{socratic2024,
  title={Socratic-Zero: Bootstrapping Reasoning via Data-Free Agent Co-evolution},
  author={Wang, Shaobo and Jiao, Zhengbo and Zhang, Zifan and Peng, Yilang and Ze, Xu and Yang, Boyu and Wang, Wei and Wei, Hu and Zhang, Linfeng},
  journal={arXiv preprint arXiv:2509.24726},
  year={2024},
  url={http://arxiv.org/abs/2509.24726}
}
```

## 11. License and Support

This project is released under an internal research license. For questions, technical support, or collaboration inquiries, please contact the development team.

---

**Technical Note**: The ProSetting implementation provides the computational infrastructure for the Socratic-Zero framework, with modular components (`collectors/`, `processors/`, `datasets/`, `trainers/`, `managers/`, `core/`) enabling systematic co-evolutionary training for mathematical reasoning advancement.