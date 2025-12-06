package main

import (
	"context"
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textencoding"
	"gorgonia.org/tensor"
)

// RealLLM wraps a Cybertron BERT model for feature extraction.
type RealLLM struct {
	Interface textencoding.Interface
}

func NewRealLLM(optionalModelName string) (*RealLLM, error) {
	// Load a small BERT model (TinyBERT is good for fast tests, but let's use a standard one or distilled)
	// "sentence-transformers/all-MiniLM-L6-v2" is very popular and small.
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if optionalModelName != "" {
		modelName = optionalModelName
	}

	fmt.Println("Generating embeddings via Cybertron... (this may take time on first run)", modelName)

	m, err := tasks.Load[textencoding.Interface](&tasks.Config{
		ModelsDir: "./models",
		ModelName: modelName,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %v", err)
	}

	return &RealLLM{Interface: m}, nil
}

// GetEmbeddings returns the CLS token embedding or Mean Pooling.
// BERT Tiny Dim = 128.
func (m *RealLLM) GetEmbeddings(texts []string) (tensor.Tensor, error) {
	// Cybertron processes one by one or batched? Interface is usually single.
	// We will loop.

	var allEmbs []float32
	var dim int

	for _, text := range texts {
		result, err := m.Interface.Encode(context.Background(), text, 0) // 0 = pooling strategy? check docs.
		// Cybertron Encode returns *textencoding.Response which has Vector.
		if err != nil {
			return nil, err
		}

		vec := result.Vector
		v32 := make([]float32, vec.Size())
		// Spago Matrix to slice?
		// vec is mat.Matrix. Data() returns []float64 usually.
		data := vec.Data().F64()
		for i, v := range data {
			v32[i] = float32(v)
		}

		if dim == 0 {
			dim = len(v32)
		}
		allEmbs = append(allEmbs, v32...)
	}

	batchSize := len(texts)
	// Return (Batch, 1, Dim) - we treat it as Sequence Length 1 (Pooled)
	// To use attention masking layer, we might need Seq > 1?
	// If we use pooled (sentence) embeddings, our "Sequence" is the batch of sentences? No.
	// "Graph-Masked Attention" usually works on graphs of tokens or graphs of concepts.
	// If user wants to "finetune" this layer, we can treat the input as (Batch, 1, Emb).

	t := tensor.New(tensor.WithShape(batchSize, 1, dim), tensor.WithBacking(allEmbs))
	return t, nil
}
