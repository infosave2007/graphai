package main

import (
	"crypto/md5"
	"encoding/binary"
	"math/rand"

	"gorgonia.org/tensor"
)

// MockLLM simulates an external Pre-Trained Model (like Llama-2).
// In a real scenario, this would wrap go-llama.cpp or an ONNX session.
type MockLLM struct {
	EmbDim int
	Vocab  map[string][]float32
}

func NewMockLLM(embDim int) *MockLLM {
	return &MockLLM{
		EmbDim: embDim,
		Vocab:  make(map[string][]float32),
	}
}

// GetEmbeddings converts a batch of text (tokens) into embeddings.
// Returns a tensor of shape (Batch, SeqLen, EmbDim).
// We simulate this by hashing the token string to generate a deterministic vector.
func (m *MockLLM) GetEmbeddings(batch []string, seqLen int) tensor.Tensor {
	batchSize := len(batch)

	// Create backing slice
	data := make([]float32, batchSize*seqLen*m.EmbDim)

	for b, sentence := range batch {
		// Simple tokenization by space (in real life, use tokenizer)
		// For this mock, we just repeat the string hash or pretend words exist.
		// Actually, let's treat the whole string as one "sequence" of random tokens for simplicity,
		// OR split by space. Let's split by space.

		// Note: This is a placeholder. Real Llama would return valid semantic vectors.
		// Here, "good" -> Vector A, "bad" -> Vector B.
		// We use MD5 hash to deterministic seed random generator for each word.

		// For demo, we just fill with random deterministic noise based on the input string to simulate differentiation.
		seed := int64(0)
		hash := md5.Sum([]byte(sentence))
		seed = int64(binary.BigEndian.Uint64(hash[:8]))
		r := rand.New(rand.NewSource(seed))

		for s := 0; s < seqLen; s++ {
			for d := 0; d < m.EmbDim; d++ {
				// offset
				idx := (b * seqLen * m.EmbDim) + (s * m.EmbDim) + d
				// Generate value between -1 and 1
				val := r.Float32()*2 - 1
				data[idx] = val
			}
		}
	}

	t := tensor.New(tensor.WithShape(batchSize, seqLen, m.EmbDim), tensor.WithBacking(data))
	return t
}
