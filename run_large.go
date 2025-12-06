package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// run_large.go demonstrates training on a larger model (BERT Base) with more data.

func main() {
	rand.Seed(time.Now().UnixNano())

	// 1. Load Large Model
	// "bert-base-uncased" is ~440MB.
	// Make sure you have disk space and internet connection.
	largeModelName := "bert-base-uncased"
	fmt.Printf("=== Initializing Large Scale Test (%s) ===\n", largeModelName)

	llm, err := NewRealLLM(largeModelName)
	if err != nil {
		log.Fatal(err)
	}

	// Warmup to get dimensions
	warmup, err := llm.GetEmbeddings([]string{"check"})
	if err != nil {
		log.Fatal(err)
	}
	embDim := warmup.Shape()[2]
	fmt.Printf("Model Loaded. Embedding Dimension: %d (Expected 768 for Base)\n", embDim)

	g := gorgonia.NewGraph()
	seqLen := 1
	numClasses := 2

	// Hybrid Layer
	adapter := NewHybridLayer(g, embDim, embDim)
	// Classification Head
	head := NewClassificationHead(g, embDim, numClasses)

	// 2. Generate/Define Large Dataset
	// We will manually define ~40 examples to show "Larger Options".
	// In real life, load from CSV.
	trainRaw := []struct {
		Text  string
		Label int
	}{
		// Positive
		{"I absolutely love this features, works perfectly", 0},
		{"Great job on the update, successful deploy", 0},
		{"Wonderful interface and smooth experience", 0},
		{"The best tool I have ever used", 0},
		{"Incredible performance boost", 0},
		{"Highly recommended to everyone", 0},
		{"Customer support is fantastic and helpful", 0},
		{"Solved all my problems instantly", 0},
		{"A masterpiece of engineering", 0},
		{"Very intuitive and easy to use", 0},
		{"Lightning fast processing speed", 0},
		{"Reliable and robust system", 0},
		{"Exceeded my expectations", 0},
		{"Happy to use this service", 0},
		{"Five stars rating from me", 0},
		{"Secure and safe implementation", 0},
		{"Beautiful design and colors", 0},
		{"Correct output every time", 0},
		{"Seamless integration with my app", 0},
		{"Just perfect in every way", 0},

		// Negative
		{"This is total garbage, crashed immediately", 1},
		{"Wasted my time and money", 1},
		{"Terrible user experience, very laggy", 1},
		{"Support never replies, rude staff", 1},
		{"Errors and bugs everywhere", 1},
		{"Does not work as advertised", 1},
		{"Completely broken update", 1},
		{"I hate this interface, so confusing", 1},
		{"Slow, buggy and unreliable", 1},
		{"Failed to save my data, data loss", 1},
		{"Security vulnerability found", 1},
		{"Ugly and outdated design", 1},
		{"Incorrect results and calculation errors", 1},
		{"Cannot connect to server, timeout", 1},
		{"Uninstalled it right away", 1},
		{"Not worth the price, too expensive", 1},
		{"Frustrating and annoying bugs", 1},
		{"Worst experience ever", 1},
		{"Do not download this app", 1},
		{"System failure and critical error", 1},
	}

	fmt.Printf("Dataset Size: %d samples\n", len(trainRaw))

	texts := make([]string, len(trainRaw))
	labels := make([]float32, len(trainRaw)*numClasses)
	for i, item := range trainRaw {
		texts[i] = item.Text
		if item.Label == 0 {
			labels[i*2] = 1.0
		} else {
			labels[i*2+1] = 1.0
		}
	}

	// 3. Get Embeddings (Batching might be needed for huge datasets, but 40 is fine)
	fmt.Println("Generating embeddings for all training data...")
	start := time.Now()
	inputsVal, err := llm.GetEmbeddings(texts)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Embeddings generated in %v\n", time.Since(start))

	yVal := tensor.New(tensor.WithShape(len(trainRaw), numClasses), tensor.WithBacking(labels))

	// 4. Build Graph
	x := gorgonia.NodeFromAny(g, inputsVal, gorgonia.WithName("Inputs"))
	y := gorgonia.NodeFromAny(g, yVal, gorgonia.WithName("Targets"))

	maskData := tensor.New(tensor.WithShape(seqLen, seqLen), tensor.Of(tensor.Float32))
	maskData.SetAt(float32(0), 0)
	adapter.AddTaskMask(0, maskData)

	features, _ := adapter.Forward(x, 0)
	logits, _ := head.Forward(features)
	probs, _ := gorgonia.SoftMax(logits)

	diff, _ := gorgonia.Sub(probs, y)
	sqDiff, _ := gorgonia.Square(diff)
	loss, _ := gorgonia.Mean(sqDiff)

	var lossVal gorgonia.Value
	gorgonia.Read(loss, &lossVal)

	learnables := []*gorgonia.Node{}
	learnables = append(learnables, adapter.GetTrainableNodes()...)
	learnables = append(learnables, head.Linear, head.Bias)

	if _, err := gorgonia.Grad(loss, learnables...); err != nil {
		log.Fatal(err)
	}

	// Use Adam
	// standard for BERT fine-tuning is low (e.g. 2e-5), but we train adapter from scratch.
	// 0.005 failed. Triyng 0.001.
	solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.001))
	machine := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(learnables...))
	defer machine.Close()

	// Train Loop
	fmt.Println("=== Starting Training Loop ===")
	if err := machine.RunAll(); err != nil {
		log.Fatal(err)
	}
	initLoss := lossVal.Data().(float32)
	fmt.Printf("Start Loss: %f\n", initLoss)
	machine.Reset()

	epochs := 1000 // Should converge faster with better embeddings
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
	fmt.Printf("Final Loss: %f\n", finalLoss)
	fmt.Printf("Efficiency: %.2f%%\n", ((initLoss-finalLoss)/initLoss)*100)

	// Test
	fmt.Println("\n=== TEST PHASE ===")
	testRaw := []struct {
		Text     string
		Expected string
	}{
		{"I am extremely satisfied with the robust functionality", "Pos"},
		{"It keeps crashing and checking for updates forever", "Neg"},
	}
	testTexts := []string{testRaw[0].Text, testRaw[1].Text}
	testEmbs, _ := llm.GetEmbeddings(testTexts)

	gTest := gorgonia.NewGraph()
	lTest := NewHybridLayer(gTest, embDim, embDim)
	// Copy Weights manually
	if len(lTest.Edges) != len(adapter.Edges) {
		log.Fatal("Edge count mismatch")
	}
	for i, edge := range adapter.Edges {
		val := edge.Weight.Value()
		if val == nil {
			continue
		}
		lTest.Edges[i].Weight = gorgonia.NodeFromAny(gTest, val.(tensor.Tensor).Clone().(tensor.Tensor), gorgonia.WithName(edge.Name))
	}

	hTest := NewClassificationHead(gTest, embDim, numClasses)
	gorgonia.Let(hTest.Linear, head.Linear.Value())
	gorgonia.Let(hTest.Bias, head.Bias.Value())

	xTest := gorgonia.NodeFromAny(gTest, testEmbs, gorgonia.WithName("TestInput"))
	maskTest := tensor.New(tensor.WithShape(1, 1), tensor.Of(tensor.Float32))
	maskTest.SetAt(float32(0), 0)
	lTest.AddTaskMask(1, maskTest)

	fTest, _ := lTest.Forward(xTest, 1)
	logTest, _ := hTest.Forward(fTest)
	probTest, _ := gorgonia.SoftMax(logTest)

	tmTest := gorgonia.NewTapeMachine(gTest)
	if err := tmTest.RunAll(); err != nil {
		log.Fatal(err)
	}

	res := probTest.Value().Data().([]float32)
	fmt.Printf("'%s' -> Pos: %.3f, Neg: %.3f\n", testRaw[0].Text, res[0], res[1])
	fmt.Printf("'%s' -> Pos: %.3f, Neg: %.3f\n", testRaw[1].Text, res[2], res[3])
}
