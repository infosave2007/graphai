package main

import (
	"fmt"
	"math"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// DTGEdge represents a learnable connection in the graph with task metadata.
type DTGEdge struct {
	Weight *gorgonia.Node
	TaskID int
	Frozen bool
	Name   string
}

// HybridGraphLayer implements the Graph-Masked Attention adaptation layer (DTG-MA).
// It manages a dynamic set of edges (Weights) that are assigned to specific tasks.
// Old tasks can be frozen to prevent catastrophic forgetting.
type HybridGraphLayer struct {
	Graph     *gorgonia.ExprGraph
	InputDim  int
	OutputDim int

	// Core Matrices (Global Projections) - effectively shared edges, or we can split them?
	// For DTG-MA, usually specific paths are masked.
	// Let's implement Task-Specific Projections or Shared with Masked Attention.
	// If the user wants "Graph with task_id for each edge", it implies distinct parameters per task OR shared parameters with task-specific masks.
	// The prompt says "W_edge matrix weights... Freeze old...".
	// We will treat W_Q, W_K, W_V as collections of edges or just standard attention with Masking.
	// "Mask with -Inf applied to attention".
	// Let's stick to Standard Attention structure but manage the Weights as "Edges" that can be partially frozen if we expanded them.
	// BUT, simplest strict interpretation:
	// We have a base set of weights. New tasks might add NEW nodes/edges?
	// The prompt says "Dynamic expansion".
	// Let's implement: W is a list of blocks?
	// actually, standard implementation:
	// W_Q, W_K, W_V are initialized for Task 0.
	// For Task 1, we might ADD columns or use the same and rely on Mask?
	// The User emphasis: "Graph-Masked Attention... Mask with -Inf... TaskID tracking".

	// Implementation:
	// We will keep standard Q, K, V mechanism, but wrapping the WEIGHT nodes in `DTGEdge`.
	// AND we apply a strict Mask to the *scores*.

	Edges     []*DTGEdge
	TaskMasks map[int]*gorgonia.Node // Mask matrix for each TaskID
}

func NewHybridLayer(g *gorgonia.ExprGraph, inputDim, outputDim int) *HybridGraphLayer {
	l := &HybridGraphLayer{
		Graph:     g,
		InputDim:  inputDim,
		OutputDim: outputDim,
		Edges:     make([]*DTGEdge, 0),
		TaskMasks: make(map[int]*gorgonia.Node),
	}

	// Initialize Base Edges for Task 0
	l.AddEdge("W_Q", inputDim, outputDim, 0)
	l.AddEdge("W_K", inputDim, outputDim, 0)
	l.AddEdge("W_V", inputDim, outputDim, 0)

	return l
}

// AddEdge creates a new weight matrix for a specific task.
func (l *HybridGraphLayer) AddEdge(name string, row, col, taskID int) {
	w := gorgonia.NewMatrix(l.Graph, tensor.Float32,
		gorgonia.WithShape(row, col),
		gorgonia.WithName(fmt.Sprintf("%s_Task%d", name, taskID)),
		gorgonia.WithInit(gorgonia.GlorotU(1.0))) // Xavier

	l.Edges = append(l.Edges, &DTGEdge{
		Weight: w,
		TaskID: taskID,
		Frozen: false,
		Name:   name,
	})
}

// FreezeOldTasks locks parameters from previous tasks.
func (l *HybridGraphLayer) FreezeOldTasks(currentTaskID int) {
	for _, edge := range l.Edges {
		if edge.TaskID < currentTaskID {
			edge.Frozen = true
			fmt.Printf("Freezing edge %s (Task %d)\n", edge.Weight.Name(), edge.TaskID)
		}
	}
}

// AddTaskMask registers a binary/float mask for a task.
// 0.0 = Allow, -Inf = Block.
func (l *HybridGraphLayer) AddTaskMask(taskID int, maskData tensor.Tensor) {
	// Create constant node for mask
	// Make sure maskData has -1e9 or similar for blocking.
	// Creating a generic node from tensor.
	maskNode := gorgonia.NodeFromAny(l.Graph, maskData, gorgonia.WithName(fmt.Sprintf("Mask_Task%d", taskID)))
	l.TaskMasks[taskID] = maskNode
}

// Forward computes the output. In DTG-MA, we usually combine edges from active tasks?
// Or we select specific edges?
// Simpler interpretation: Standard Attention but with a Mask that depends on Task.
// And we only train edges associated with current task (or all if not frozen).
// For now, let's assume standard QKV attention using the *latest* W_Q, W_K, W_V or a sum?
// If we simply have ONE set of W_Q, W_K, W_V, but we "expand" them?
// Let's assume for this code: We use the edges matching the names "W_Q", "W_K", "W_V".
// If multiple exist (e.g. W_Q_Task0, W_Q_Task1), we might sum their outputs?
// Or usually in Adapter logic: Output = Base(x) + Adapter_Task1(x) ...
// The "Graph" part implies we might traverse valid edges.
// Let's implement: Sum of projections of all VALID edges for the given task configuration.
// BUT usually for Task N, we use Mask N.
// Let's assume we sum all 'W_Q' edges, all 'W_K' edges...
func (l *HybridGraphLayer) Forward(input *gorgonia.Node, taskID int) (*gorgonia.Node, error) {
	// 1. Project Input -> Q, K, V
	// We aggregate contributions from all edges that are valid/active.
	// For simplicity, let's sum the results of all W_Q edges, etc.

	var Q, K, V *gorgonia.Node
	var err error

	Q, err = l.computeProjection(input, "W_Q")
	if err != nil {
		return nil, err
	}
	K, err = l.computeProjection(input, "W_K")
	if err != nil {
		return nil, err
	}
	V, err = l.computeProjection(input, "W_V")
	if err != nil {
		return nil, err
	}

	// 2. Scaled Dot-Product Attention
	// Reshape logic (B, S, E) -> (B*S, E) is handled in projection usually,
	// but to do MatMul(Q, K^T) we need shape (Batch, Seq, Emb).
	// Gorgonia BatchedMatMul requires 3D tensors.
	// Input is assumed (Batch, Seq, Emb).

	// Q * K^T
	// (B, S, E) * (B, E, S) -> (B, S, S)
	// gorgonia.BatchedMatMul(a, b, transA, transB) usually?
	// Actually the API usually is BatchedMatMul(a, b). Transpose must be manual or via flags using different Op.
	// Standard Gorgonia: `gorgonia.BatchedMatMul` takes (a, b). To transpose, use `gorgonia.Transpose`.

	// Let's transpose K manualy: (B, S, E) -> (B, E, S).
	// Dimensions: 0=Batch, 1=Seq, 2=Emb. Transpose(1, 2)
	K_T, err := gorgonia.Transpose(K, 0, 2, 1)
	if err != nil {
		return nil, err
	}

	scores, err := gorgonia.BatchedMatMul(Q, K_T)
	if err != nil {
		return nil, err
	}

	// Scale
	dk := float64(l.OutputDim)
	scale := gorgonia.NodeFromAny(l.Graph, float32(1.0/math.Sqrt(dk)), gorgonia.WithName("Scale"))
	scores, err = gorgonia.HadamardProd(scores, scale) // Scalar mul
	if err != nil {
		return nil, err
	}

	// 3. APPLY MASK (CRITICAL!)
	// Mask must have -Inf where invalid.
	mask, ok := l.TaskMasks[taskID]
	if !ok {
		return nil, fmt.Errorf("mask for task %d not found", taskID)
	}
	// Mask shape should be (Seq, Seq) or broadcastable. (1, 1) works for 1-token.
	// scores + mask. 0 + (-Inf) = -Inf.
	scores, err = gorgonia.BroadcastAdd(scores, mask, nil, []byte{0}) // Broadcast batch?
	// If Mask is (1, 1), and scores (B, 1, 1), we broadcast over B.
	if err != nil {
		return nil, err
	}

	// 4. Softmax
	// Batched Softmax is tricky. Flatten -> Softmax -> Reshape.
	b, s, _ := scores.Shape()[0], scores.Shape()[1], scores.Shape()[2] // should be S
	flatScores, err := gorgonia.Reshape(scores, tensor.Shape{b * s, s})
	if err != nil {
		return nil, err
	}

	probsFlat, err := gorgonia.SoftMax(flatScores)
	if err != nil {
		return nil, err
	}

	probs, err := gorgonia.Reshape(probsFlat, tensor.Shape{b, s, s})
	if err != nil {
		return nil, err
	}

	// 5. Output = Probs * V
	output, err := gorgonia.BatchedMatMul(probs, V)
	if err != nil {
		return nil, err
	}

	return output, nil
}

// computeProjection sums up x*W for all edges with given name prefix (e.g. W_Q).
func (l *HybridGraphLayer) computeProjection(input *gorgonia.Node, prefix string) (*gorgonia.Node, error) {
	var sum *gorgonia.Node

	// Input (B, S, E). Reshape to (B*S, E) for MatMul
	b, s, e := input.Shape()[0], input.Shape()[1], input.Shape()[2]
	flatInput, err := gorgonia.Reshape(input, tensor.Shape{b * s, e})
	if err != nil {
		return nil, err
	}

	found := false
	for _, edge := range l.Edges {
		if edge.Name == prefix {
			// Calculate x * W
			proj, err := gorgonia.Mul(flatInput, edge.Weight)
			if err != nil {
				return nil, err
			}

			if sum == nil {
				sum = proj
			} else {
				sum, err = gorgonia.Add(sum, proj)
				if err != nil {
					return nil, err
				}
			}
			found = true
		}
	}
	if !found {
		return nil, fmt.Errorf("no edges found for %s", prefix)
	}

	// Reshape back to (B, S, E_out)
	// Assuming OutputDim matches projected dim
	reshaped, err := gorgonia.Reshape(sum, tensor.Shape{b, s, l.OutputDim})
	if err != nil {
		return nil, err
	}

	return reshaped, nil
}

// GetTrainableNodes returns only the nodes that are NOT frozen.
func (l *HybridGraphLayer) GetTrainableNodes() []*gorgonia.Node {
	nodes := make([]*gorgonia.Node, 0)
	for _, edge := range l.Edges {
		if !edge.Frozen {
			nodes = append(nodes, edge.Weight)
		}
	}
	return nodes
}
