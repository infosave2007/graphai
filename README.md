# GraphAI: Hybrid Graph-Masked Attention Layer (DTG-MA)

**[Russian Version (Русская версия)](README_ru.md)**

> [!IMPORTANT]
> **Key Idea**: This method provides a definitive solution to **Catastrophic Forgetting**.
> The architecture guarantees that training on new tasks **mathematically cannot** degrade neural pathways responsible for old knowledge. This makes it an ideal foundation for *Continual Learning*.

## Overview

**GraphAI** is a Go-based implementation of the **Dynamic Topology Graph - Masked Attention (DTG-MA)** layer, integrated with real-world Large Language Models (LLMs) via the `Cybertron` library. This project demonstrates a **Continual Learning** architecture designed to adapt to new tasks without catastrophic forgetting.

Key architectural features:
*   **Dynamic Topology**: The graph structure expands dynamically as new tasks are introduced (`AddEdge`).
*   **Topology-Aware Attention**: Implements the masked attention formula $Softmax(\frac{QK^T}{\sqrt{d}} + M_{task})V$, where $M_{task}$ applies strict $-\infty$ masking to enforce task-specific topology.
*   **Zero-Forgetting**: Old task parameters are explicitly frozen (`Frozen` flag in `DTGEdge`) during new task training.
*   **Real LLM Integration**: Uses state-of-the-art pre-trained embeddings (e.g., BERT, MiniLM) as the input foundation.

## Features

- **True DTG-MA Logic**:
  - **Edge Metadata**: Every weight matrix is wrapped in a `DTGEdge` struct tracking its Task ID and frozen state.
  - **Strict Masking**: Uses `-Inf` masking to rigorously block attention pathways, preventing interference between tasks.
  - **Task Management**: Explicit `TaskID` based routing in the `Forward` pass.

- **Pure Go Ecosystem**:
  - Built on `Gorgonia` for computation graphs.
  - Integration with generic `Go` tensors.
  - No Python dependencies for the core logic.

## Comparison with State-of-the-Art

Three main classes of solutions exist in Continual Learning, but each has significant drawbacks:

1.  **Elastic Weight Consolidation (EWC)**
    *   *Method*: Uses Fisher Information Matrix to identify and penalize changes to "important" weights.
    *   *Drawback*: Computationally expensive (calculating Fisher Matrix) and only provides a "soft" constraint (forgetting can still happen).

2.  **Learning without Forgetting (LwF)**
    *   *Method*: Uses Knowledge Distillation where the old model teaches the new one.
    *   *Drawback*: Requires maintaining the old model and running inference on it during training, doubling the compute load.

3.  **Parameter Isolation**
    *   *Method*: Assigns separate sub-networks or adapters for each task.
    *   *Drawback*: Often leads to linear parameter growth without knowledge reuse.

### Why GraphAI (DTG-MA) is Better
**GraphAI** solves these issues by combining **Dynamic Topology** with **Masked Attention**:
*   **Efficient (vs EWC)**: No expensive Fisher Matrix calculations. Knowledge protection is architectural (`-Inf` Mask + Freezing), which has near-zero overhead.
*   **Fast (vs LwF)**: No need for Knowledge Distillation or keeping old models in memory.
*   **Guaranteed Isolation**: Unlike soft constraints, the Masked Attention mechanism mathematically guarantees **Zero-Forgetting**.
*   **Flexible**: The graph structure allows for potential knowledge reuse (unlike strict isolation) while maintaining separation where needed.

## Installation

### Prerequisites
- Go 1.25+ (or configured environment variable `ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25` for Gorgonia compatibility).

### Setup
1.  Initialize the module:
    ```bash
    go mod init graphai
    go mod tidy
    ```
2.  Ensure dependencies are downloaded:
    ```bash
    go get gorgonia.org/gorgonia
    go get github.com/nlpodyssey/cybertron
    ```

## Usage

### 1. Minimal Example (MiniLM)
Runs the training loop using `sentence-transformers/all-MiniLM-L6-v2` (fast, small model).
```bash
export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25
go run main.go layer.go real_llm.go head.go
```

### 2. Large Scale Test (BERT Base)
Runs the training loop using `bert-base-uncased` (standard 768-dim model) with a larger dataset.
```bash
export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25
go run run_large.go layer.go real_llm.go head.go
```
*Note: The first run will download the model weights (~440MB).*

## Architecture Details

### `layer.go`
Contains the core `HybridGraphLayer` implementation.
- **`DTGEdge`**: Struct representing a learnable connection `[Weight, TaskID, Frozen]`.
- **`Forward(input, taskID)`**: Computes masked attention. It selects the mask corresponding to `taskID` and applies it additively to the scaled dot-product scores before Softmax.
- **`FreezeOldTasks(currentTaskID)`**: Iterates through edges and sets `Frozen=true` for any edge belonging to previous tasks.

### `real_llm.go`
A wrapper around the `Cybertron` library.
- **`NewRealLLM(modelName)`**: Loads a specific HuggingFace model.
- **`GetEmbeddings(texts)`**: Converts a slice of strings into a `trainRaw` tensor suitable for the graph.

## Configuration

To fine-tune the training process, modify the `solver` configuration in `main.go` or `run_large.go`:
```go
// Adjust Learning Rate for convergence
solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.001))
```

To switch models, pass the model name to `NewRealLLM`:
```go
llm, err := NewRealLLM("bert-large-uncased")
```

## License
MIT License.
