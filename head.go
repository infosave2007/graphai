package main

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// ClassificationHead represents the final prediction layer.
type ClassificationHead struct {
	Graph  *gorgonia.ExprGraph
	Linear *gorgonia.Node // Weights (EmbDim -> NumClasses)
	Bias   *gorgonia.Node // (NumClasses)
}

func NewClassificationHead(g *gorgonia.ExprGraph, inputDim, numClasses int) *ClassificationHead {
	// Initialize W and b
	w := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(inputDim, numClasses), gorgonia.WithName("Head_W"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	b := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(1, numClasses), gorgonia.WithName("Head_b"), gorgonia.WithInit(gorgonia.Zeroes()))

	return &ClassificationHead{
		Graph:  g,
		Linear: w,
		Bias:   b,
	}
}

func (h *ClassificationHead) Forward(input *gorgonia.Node) (*gorgonia.Node, error) {
	// Input (Batch, Seq, Emb).
	// We do "Mean Pooling" logic or explicit Sum.
	// Since Seq=1 here usually, it's just Reshape -> Linear.

	b, s, e := input.Shape()[0], input.Shape()[1], input.Shape()[2]
	c := h.Linear.Shape()[1]

	flatInput, err := gorgonia.Reshape(input, tensor.Shape{b * s, e})
	if err != nil {
		return nil, err
	}

	logitsFlat, err := gorgonia.Mul(flatInput, h.Linear)
	if err != nil {
		return nil, err
	}

	// Broadcast Add Bias
	// Bias (1, C). Logits (B*S, C).
	logitsFlatBias, err := gorgonia.BroadcastAdd(logitsFlat, h.Bias, nil, []byte{0})
	if err != nil {
		return nil, err
	}

	logits, err := gorgonia.Reshape(logitsFlatBias, tensor.Shape{b, s, c})
	if err != nil {
		return nil, err
	}

	// Sum across seq (dim 1) to get (Batch, Classes)
	pooled, err := gorgonia.Sum(logits, 1)
	if err != nil {
		return nil, err
	}

	return pooled, nil
}
