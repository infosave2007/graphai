package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// ClassificationHead is now in head.go

func main() {
	rand.Seed(time.Now().UnixNano())

	// 1. Setup Real LLM (Empty string = default MiniLM)
	llm, err := NewRealLLM("")
	if err != nil {
		log.Fatal(err)
	}

	// We need to know embedding dimension dynamically or hardcode for bert-tiny (128)
	// Let's do a warm-up call
	warmup, err := llm.GetEmbeddings([]string{"test"})
	if err != nil {
		log.Fatal(err)
	}
	embDim := warmup.Shape()[2] // (Batch, 1, Dim)
	fmt.Printf("Model Embedding Dimension: %d\n", embDim)

	g := gorgonia.NewGraph()
	seqLen := 1 // Pooled output
	numClasses := 2

	// The Adapter
	adapter := NewHybridLayer(g, embDim, embDim)

	// The Head
	head := NewClassificationHead(g, embDim, numClasses)

	// 2. Data (Real Strings)
	trainRaw := []struct {
		Text  string
		Label int
	}{
		// Positive (0)
		{"I love this product, it is amazing", 0},
		{"The best experience of my life", 0},
		{"Fantastic support and great quality", 0},
		{"I am very happy with the results", 0},
		{"What a wonderful day", 0},
		{"High quality and super fast delivery", 0},
		{"I enjoy using this software every day", 0},
		{"Absolutely brilliant work", 0},
		{"Good job team, well done", 0},
		{"Everything works perfectly fine", 0},

		// Negative (1)
		{"Absolutely terrible service and rude staff", 1},
		{"I hate waiting for so long, waste of time", 1},
		{"This is the worst item I ever bought", 1},
		{"Completely broken and useless", 1},
		{"I am very disappointed with this", 1},
		{"Errors everywhere, cannot use it", 1},
		{"Sad and boring experience", 1},
		{"Not recommended, stay away", 1},
		{"It failed to load multiple times", 1},
		{"Garbage quality, do not buy", 1},
	}

	texts := make([]string, len(trainRaw))
	for i, item := range trainRaw {
		texts[i] = item.Text
	}

	// GET REAL EMBEDDINGS
	fmt.Println("Генерация эмбеддингов через TinyBERT...")
	inputsVal, err := llm.GetEmbeddings(texts)
	if err != nil {
		log.Fatal(err)
	}

	batchSize := len(trainRaw)
	targetsVal := make([]float32, batchSize*numClasses)
	for i, item := range trainRaw {
		if item.Label == 0 {
			targetsVal[i*numClasses+0] = 1.0
		} else {
			targetsVal[i*numClasses+1] = 1.0
		}
	}
	yVal := tensor.New(tensor.WithShape(batchSize, numClasses), tensor.WithBacking(targetsVal))

	// 4. Create Graph and Layer
	g = gorgonia.NewGraph()
	seqLen = 1
	numClasses = 2 // Pos, Neg

	// Hybrid Layer (Task 0 = Train)
	adapter = NewHybridLayer(g, embDim, embDim)
	// Classification Head
	head = NewClassificationHead(g, embDim, numClasses)

	x := gorgonia.NodeFromAny(g, inputsVal, gorgonia.WithName("Inputs"))
	y := gorgonia.NodeFromAny(g, yVal, gorgonia.WithName("Targets"))

	// Mask for Task 0. Use -Inf properly.
	maskData := tensor.New(tensor.WithShape(seqLen, seqLen), tensor.Of(tensor.Float32))
	// 0.0 means unmasked. For SeqLen=1, it is trivial.
	maskData.SetAt(float32(0), 0)
	adapter.AddTaskMask(0, maskData)

	// Forward pass (Task 0)
	features, err := adapter.Forward(x, 0)
	if err != nil {
		log.Fatal(err)
	}

	logits, err := head.Forward(features)
	if err != nil {
		log.Fatal(err)
	}

	// Softmax + Loss
	probs, err := gorgonia.SoftMax(logits)
	if err != nil {
		log.Fatal(err)
	}
	// ... (Loss calculation same) ...
	diff, _ := gorgonia.Sub(probs, y)
	sqDiff, _ := gorgonia.Square(diff)
	loss, _ := gorgonia.Mean(sqDiff)

	var lossVal gorgonia.Value
	gorgonia.Read(loss, &lossVal)

	learnables := []*gorgonia.Node{}
	// Add Adapter Nodes (Task 0 = Active)
	learnables = append(learnables, adapter.GetTrainableNodes()...)
	learnables = append(learnables, head.Linear, head.Bias)

	if _, err := gorgonia.Grad(loss, learnables...); err != nil {
		log.Fatal(err)
	}

	// Read Predictions for debug
	var predsVal gorgonia.Value
	gorgonia.Read(probs, &predsVal)

	// 6. Solver
	solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.005))
	machine := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(learnables...))
	defer machine.Close()

	// TRAIN
	fmt.Println("=== Starting Training on Real Embeddings ===")
	if err := machine.RunAll(); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Initial Loss: %v\n", lossVal)
	initLoss := lossVal.Data().(float32)

	machine.Reset()

	epochs := 500
	for i := 0; i < epochs; i++ {
		if err := machine.RunAll(); err != nil {
			log.Fatal(err)
		}
		if err := solver.Step(gorgonia.NodesToValueGrads(learnables)); err != nil {
			log.Fatal(err)
		}
		if i%100 == 0 {
			fmt.Printf("Epoch %d: Loss %f\n", i, lossVal.Data())
		}
		machine.Reset()
	}

	if err := machine.RunAll(); err != nil {
		log.Fatal(err)
	}
	finalLoss := lossVal.Data().(float32)
	fmt.Printf("Final Loss: %v\n", finalLoss)

	if predsVal != nil {
		preds := predsVal.Data().([]float32)
		fmt.Println("Training set sample predictions:")
		for k := 0; k < 4; k++ {
			fmt.Printf("Train[%d] (Pos): %.3f, %.3f\n", k, preds[k*2], preds[k*2+1])
		}
		for k := 10; k < 14; k++ {
			fmt.Printf("Train[%d] (Neg): %.3f, %.3f\n", k, preds[k*2], preds[k*2+1])
		}
	}

	improvement := ((initLoss - finalLoss) / initLoss) * 100
	fmt.Printf("Efficiency (Loss Reduction): %.2f%%\n", improvement)

	// EVALUATION ON TEST SET (Unseen Data)
	// Must recreate graph and COPY weights manually since Graph is static.
	fmt.Println("\n=== Evaluating on Test Set (New Data) ===")
	testRaw := []struct {
		Text  string
		Label int
	}{
		{"Joyful and happy day", 0},       // Pos
		{"Start up failed completely", 1}, // Neg
	}
	testTexts := []string{testRaw[0].Text, testRaw[1].Text}

	fmt.Println("Generating embeddings for test set...")
	testInputsVal, err := llm.GetEmbeddings(testTexts)
	if err != nil {
		log.Fatal(err)
	}

	gTest := gorgonia.NewGraph()
	// New Layer for Test (Task 1)
	layerTest := NewHybridLayer(gTest, embDim, embDim)
	// IMPORTANT: Manually copy trained weights to Test layer
	// Assumes Edge order matches (W_Q, W_K, W_V).
	if len(layerTest.Edges) != len(adapter.Edges) {
		log.Fatal("Mismatch in edges count for test")
	}
	for i, edge := range adapter.Edges {
		// Get trained value
		val := edge.Weight.Value()
		if val == nil {
			continue
		}
		clone := val.(tensor.Tensor).Clone().(tensor.Tensor)

		// Set to test layer edge
		testEdge := layerTest.Edges[i]
		gorgonia.Let(testEdge.Weight, clone)
	}

	headTest := NewClassificationHead(gTest, embDim, numClasses)
	gorgonia.Let(headTest.Linear, head.Linear.Value())
	gorgonia.Let(headTest.Bias, head.Bias.Value())

	xTest := gorgonia.NodeFromAny(gTest, testInputsVal, gorgonia.WithName("TestInputs"))

	// Mask for Test Task (Task 1).
	maskTest := tensor.New(tensor.WithShape(1, 1), tensor.Of(tensor.Float32))
	maskTest.SetAt(float32(0), 0)
	layerTest.AddTaskMask(1, maskTest)

	featTest, _ := layerTest.Forward(xTest, 1)
	logitsTest, _ := headTest.Forward(featTest)
	probsTest, _ := gorgonia.SoftMax(logitsTest)

	tmTest := gorgonia.NewTapeMachine(gTest)
	if err := tmTest.RunAll(); err != nil {
		log.Fatal(err)
	}

	outProbs := probsTest.Value().Data().([]float32)
	fmt.Printf("Text: '%s' -> Pos: %.4f, Neg: %.4f (Expected: Pos)\n", testRaw[0].Text, outProbs[0], outProbs[1])
	fmt.Printf("Text: '%s' -> Pos: %.4f, Neg: %.4f (Expected: Neg)\n", testRaw[1].Text, outProbs[2], outProbs[3])
}
