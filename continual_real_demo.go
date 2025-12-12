//go:build continual_real_demo
// +build continual_real_demo

package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// continual_real_demo.go
// Domain-continual learning demo on real text embeddings (movies -> sports).
//
// Run:
//   export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25
//   go run -tags continual_real_demo continual_real_demo.go layer.go head.go real_llm.go
//
// Notes:
// - Uses sentence embeddings (SeqLen=1) from Cybertron models.
// - Demonstrates *no forgetting* for Task0 by:
//   (a) freezing Task0 edges, (b) using task-scoped edges in forward, (c) using per-task heads.

type headWeights struct {
	W tensor.Tensor
	B tensor.Tensor
}

type weightsBundle struct {
	Adapter map[string]tensor.Tensor
	Heads   map[int]headWeights
}

type forwardMode int

const (
	forwardTaskScoped forwardMode = iota
	forwardNaiveMixed
)

type trainOptions struct {
	ForwardMode forwardMode
	FreezeOld   bool
	SharedHead  bool
	Epochs0     int
	Epochs1     int
	LearnRate   float64
}

type evalOptions struct {
	ForwardMode forwardMode
	SharedHead  bool
}

func main() {
	start := time.Now()

	modelName := os.Getenv("DTG_MODEL")
	llm, err := NewRealLLM(modelName)
	if err != nil {
		log.Fatal(err)
	}

	// Task0: movie reviews sentiment
	movieTrain := []labeledText{
		{"This movie was amazing, great acting and a beautiful story", 0},
		{"A fantastic film, I loved every minute", 0},
		{"Brilliant direction and wonderful soundtrack", 0},
		{"Great cinematography and impressive performances", 0},
		{"I enjoyed the film, it was fun and touching", 0},
		{"Wonderful movie with strong emotions", 0},
		{"The plot was engaging and the ending was satisfying", 0},
		{"A masterpiece, truly inspiring", 0},
		{"Solid script and excellent cast", 0},
		{"Beautiful visuals and great pacing", 0},
		{"This film made me smile, highly recommended", 0},
		{"Surprisingly good, better than expected", 0},

		{"Terrible movie, boring plot and bad acting", 1},
		{"Worst film of the year, a complete waste of time", 1},
		{"Disappointing and poorly written, I hated it", 1},
		{"Painfully slow and predictable, not recommended", 1},
		{"The story made no sense and the pacing was horrible", 1},
		{"Awful film, I regret watching it", 1},
		{"Bad acting and a weak script", 1},
		{"Confusing plot, messy editing", 1},
		{"Overhyped and underdelivered", 1},
		{"Boring and forgettable", 1},
		{"A frustrating experience, not worth it", 1},
		{"Cheap jokes and poor direction", 1},
	}
	movieTest := []labeledText{
		{"Excellent movie, great cast and direction", 0},
		{"I loved this film, very enjoyable", 0},
		{"Good story and nice soundtrack", 0},
		{"Terrible film, boring and slow", 1},
		{"Bad movie, weak acting", 1},
		{"I hated it, complete waste of time", 1},
	}

	// Task1: sports comments sentiment (same labels: pos/neg)
	sportTrain := []labeledText{
		{"What a great match, incredible teamwork and energy", 0},
		{"Amazing performance, the team played beautifully", 0},
		{"Fantastic win, brilliant strategy by the coach", 0},
		{"Great comeback, the players showed real character", 0},
		{"Excellent defense and smart passing", 0},
		{"Outstanding victory, well deserved", 0},
		{"They dominated the game from start to finish", 0},
		{"Strong performance, very disciplined", 0},
		{"Great skills and solid defense", 0},
		{"Wonderful victory, the squad looked confident", 0},
		{"The fans were happy, fantastic atmosphere", 0},
		{"Impressive stamina and effort", 0},

		{"Terrible game, sloppy defense and poor decisions", 1},
		{"Awful performance, they kept making stupid mistakes", 1},
		{"Disappointing loss, no coordination at all", 1},
		{"Painful to watch, completely outplayed", 1},
		{"Bad coaching and terrible execution", 1},
		{"Horrible match, the team was confused", 1},
		{"They played badly and deserved to lose", 1},
		{"Weak defense, too many errors", 1},
		{"A frustrating match, awful finishing", 1},
		{"Terrible decisions and no game plan", 1},
		{"Embarrassing performance", 1},
		{"They collapsed in the second half", 1},
	}
	sportTest := []labeledText{
		{"Great win, excellent teamwork", 0},
		{"Amazing match, very strong performance", 0},
		{"Solid defense and good strategy", 0},
		{"Terrible match, many mistakes", 1},
		{"Awful performance, bad coaching", 1},
		{"They were confused and played badly", 1},
	}

	// Build embeddings
	movieTrainX, movieTrainY := embedDataset(llm, movieTrain)
	movieTestX, movieTestY := embedDataset(llm, movieTest)
	sportTrainX, sportTrainY := embedDataset(llm, sportTrain)
	sportTestX, sportTestY := embedDataset(llm, sportTest)

	embDim := movieTrainX.Shape()[2]
	fmt.Printf("Embedding dim: %d\n", embDim)
	if modelName == "" {
		fmt.Println("Model: (default) sentence-transformers/all-MiniLM-L6-v2")
	} else {
		fmt.Printf("Model: %s\n", modelName)
	}

	epochs0 := getenvInt("DTG_EPOCHS0", 500)
	epochs1 := getenvInt("DTG_EPOCHS1", 450)
	lr := getenvFloat("DTG_LR", 0.0015)
	runs := getenvInt("DTG_RUNS", 3)
	seedBase := getenvInt("DTG_SEED", 42)
	fmt.Printf("Train params: epochs0=%d epochs1=%d lr=%.6f\n", epochs0, epochs1, lr)
	fmt.Printf("Repro: runs=%d seedBase=%d\n", runs, seedBase)

	type runMetrics struct {
		baselineAcc0Before float64
		baselineAcc0After  float64
		baselineAcc1       float64
		baselineMaxDelta   float32

		dtgAcc0Before float64
		dtgAcc0After  float64
		dtgAcc1       float64
		dtgMaxDelta   float32
	}

	all := make([]runMetrics, 0, runs)

	for r := 0; r < runs; r++ {
		seed := int64(seedBase + r)
		fmt.Printf("\n--- Run %d/%d (seed=%d) ---\n", r+1, runs, seed)

		// Baseline (naive)
		rand.Seed(seed)
		baseline0, err := trainTask(0, nil, embDim, movieTrainX, movieTrainY, trainOptions{
			ForwardMode: forwardNaiveMixed,
			FreezeOld:   false,
			SharedHead:  true,
			Epochs0:     epochs0,
			Epochs1:     epochs1,
			LearnRate:   lr,
		})
		if err != nil {
			log.Fatal(err)
		}
		acc0BeforeBaseline, probs0BeforeBaseline, err := evalTask(0, baseline0, embDim, movieTestX, movieTestY, evalOptions{ForwardMode: forwardNaiveMixed, SharedHead: true})
		if err != nil {
			log.Fatal(err)
		}
		baseline01, err := trainTask(1, baseline0, embDim, sportTrainX, sportTrainY, trainOptions{
			ForwardMode: forwardNaiveMixed,
			FreezeOld:   false,
			SharedHead:  true,
			Epochs0:     epochs0,
			Epochs1:     epochs1,
			LearnRate:   lr,
		})
		if err != nil {
			log.Fatal(err)
		}
		acc0AfterBaseline, probs0AfterBaseline, err := evalTask(0, baseline01, embDim, movieTestX, movieTestY, evalOptions{ForwardMode: forwardNaiveMixed, SharedHead: true})
		if err != nil {
			log.Fatal(err)
		}
		acc1Baseline, _, err := evalTask(1, baseline01, embDim, sportTestX, sportTestY, evalOptions{ForwardMode: forwardNaiveMixed, SharedHead: true})
		if err != nil {
			log.Fatal(err)
		}
		maxAbsBaseline := maxAbsDiff(probs0BeforeBaseline, probs0AfterBaseline)

		// DTG-MA
		rand.Seed(seed)
		w0, err := trainTask(0, nil, embDim, movieTrainX, movieTrainY, trainOptions{
			ForwardMode: forwardTaskScoped,
			FreezeOld:   false,
			SharedHead:  false,
			Epochs0:     epochs0,
			Epochs1:     epochs1,
			LearnRate:   lr,
		})
		if err != nil {
			log.Fatal(err)
		}
		acc0Before, probs0Before, err := evalTask(0, w0, embDim, movieTestX, movieTestY, evalOptions{ForwardMode: forwardTaskScoped})
		if err != nil {
			log.Fatal(err)
		}
		w01, err := trainTask(1, w0, embDim, sportTrainX, sportTrainY, trainOptions{
			ForwardMode: forwardTaskScoped,
			FreezeOld:   true,
			SharedHead:  false,
			Epochs0:     epochs0,
			Epochs1:     epochs1,
			LearnRate:   lr,
		})
		if err != nil {
			log.Fatal(err)
		}
		acc0After, probs0After, err := evalTask(0, w01, embDim, movieTestX, movieTestY, evalOptions{ForwardMode: forwardTaskScoped})
		if err != nil {
			log.Fatal(err)
		}
		acc1, _, err := evalTask(1, w01, embDim, sportTestX, sportTestY, evalOptions{ForwardMode: forwardTaskScoped})
		if err != nil {
			log.Fatal(err)
		}
		maxAbs := maxAbsDiff(probs0Before, probs0After)

		fmt.Printf("Baseline  Task0 acc: %.3f -> %.3f | Task1 acc: %.3f | Max|Δprob|: %.10f\n", acc0BeforeBaseline, acc0AfterBaseline, acc1Baseline, maxAbsBaseline)
		fmt.Printf("DTG-MA    Task0 acc: %.3f -> %.3f | Task1 acc: %.3f | Max|Δprob|: %.10f\n", acc0Before, acc0After, acc1, maxAbs)

		all = append(all, runMetrics{
			baselineAcc0Before: acc0BeforeBaseline,
			baselineAcc0After:  acc0AfterBaseline,
			baselineAcc1:       acc1Baseline,
			baselineMaxDelta:   maxAbsBaseline,
			dtgAcc0Before:      acc0Before,
			dtgAcc0After:       acc0After,
			dtgAcc1:            acc1,
			dtgMaxDelta:        maxAbs,
		})
	}

	// Aggregate summary
	var b0b, b0a, b1 float64
	var d0b, d0a, d1 float64
	var bMax, dMax float64
	for _, m := range all {
		b0b += m.baselineAcc0Before
		b0a += m.baselineAcc0After
		b1 += m.baselineAcc1
		bMax += float64(m.baselineMaxDelta)
		d0b += m.dtgAcc0Before
		d0a += m.dtgAcc0After
		d1 += m.dtgAcc1
		dMax += float64(m.dtgMaxDelta)
	}
	den := float64(len(all))
	if den > 0 {
		b0b /= den
		b0a /= den
		b1 /= den
		bMax /= den
		d0b /= den
		d0a /= den
		d1 /= den
		dMax /= den
	}

	fmt.Println("\n=== Summary (mean over runs) ===")
	fmt.Printf("Baseline  Task0 acc: %.3f -> %.3f | Task1 acc: %.3f | Mean Max|Δprob|: %.10f\n", b0b, b0a, b1, bMax)
	fmt.Printf("DTG-MA    Task0 acc: %.3f -> %.3f | Task1 acc: %.3f | Mean Max|Δprob|: %.10f\n", d0b, d0a, d1, dMax)

	fmt.Printf("Done in %v\n", time.Since(start))
}

type labeledText struct {
	Text  string
	Label int // 0=pos, 1=neg
}

func embedDataset(llm *RealLLM, items []labeledText) (tensor.Tensor, tensor.Tensor) {
	texts := make([]string, len(items))
	labels := make([]float32, len(items)*2)
	for i, it := range items {
		texts[i] = it.Text
		labels[i*2+it.Label] = 1
	}
	x, err := llm.GetEmbeddings(texts)
	if err != nil {
		log.Fatal(err)
	}
	y := tensor.New(tensor.WithShape(len(items), 2), tensor.WithBacking(labels))
	return x, y
}

func trainTask(taskID int, base *weightsBundle, embDim int, trainX, trainY tensor.Tensor, opts trainOptions) (*weightsBundle, error) {
	g := gorgonia.NewGraph()

	adapter := NewHybridLayer(g, embDim, embDim)
	for t := 1; t <= taskID; t++ {
		adapter.AddTask(t)
	}

	// SeqLen is 1 for sentence embeddings; mask is trivial but required by layer.
	mask := tensor.New(tensor.WithShape(1, 1), tensor.Of(tensor.Float32))
	_ = mask.SetAt(float32(0), 0, 0)
	adapter.AddTaskMask(0, mask)
	adapter.AddTaskMask(1, mask)

	if opts.FreezeOld {
		adapter.FreezeOldTasks(taskID)
	}

	head := NewClassificationHead(g, embDim, 2)
	headSlot := taskID
	if opts.SharedHead {
		headSlot = 0
	}

	if base != nil {
		for _, edge := range adapter.Edges {
			if val, ok := base.Adapter[edge.Weight.Name()]; ok {
				gorgonia.Let(edge.Weight, val)
			}
		}
		if hw, ok := base.Heads[headSlot]; ok {
			gorgonia.Let(head.Linear, hw.W)
			gorgonia.Let(head.Bias, hw.B)
		}
	}

	x := gorgonia.NodeFromAny(g, trainX, gorgonia.WithName("X"))
	y := gorgonia.NodeFromAny(g, trainY, gorgonia.WithName("Y"))

	var features *gorgonia.Node
	var err error
	switch opts.ForwardMode {
	case forwardTaskScoped:
		features, err = adapter.ForwardTaskScoped(x, taskID)
	case forwardNaiveMixed:
		features, err = adapter.Forward(x, taskID)
	default:
		return nil, fmt.Errorf("unknown forward mode: %d", opts.ForwardMode)
	}
	if err != nil {
		return nil, err
	}
	logits, err := head.Forward(features)
	if err != nil {
		return nil, err
	}
	probs, err := gorgonia.SoftMax(logits)
	if err != nil {
		return nil, err
	}

	loss, err := crossEntropyFromProbs(probs, y)
	if err != nil {
		return nil, err
	}

	learnables := []*gorgonia.Node{}
	learnables = append(learnables, adapter.GetTrainableNodes()...)
	learnables = append(learnables, head.Linear, head.Bias)
	if _, err := gorgonia.Grad(loss, learnables...); err != nil {
		return nil, err
	}

	if opts.LearnRate <= 0 {
		opts.LearnRate = 0.0015
	}
	solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(opts.LearnRate))
	machine := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(learnables...))
	defer machine.Close()

	epochs := opts.Epochs1
	if taskID == 0 {
		epochs = opts.Epochs0
	}
	if epochs <= 0 {
		epochs = 200
	}

	for i := 0; i < epochs; i++ {
		if err := machine.RunAll(); err != nil {
			return nil, err
		}
		if err := solver.Step(gorgonia.NodesToValueGrads(learnables)); err != nil {
			return nil, err
		}
		machine.Reset()
	}

	out := &weightsBundle{Adapter: map[string]tensor.Tensor{}, Heads: map[int]headWeights{}}
	if base != nil {
		for k, v := range base.Adapter {
			out.Adapter[k] = v.Clone().(tensor.Tensor)
		}
		for tid, hw := range base.Heads {
			out.Heads[tid] = headWeights{W: hw.W.Clone().(tensor.Tensor), B: hw.B.Clone().(tensor.Tensor)}
		}
	}
	for _, edge := range adapter.Edges {
		if edge.Weight.Value() == nil {
			continue
		}
		out.Adapter[edge.Weight.Name()] = edge.Weight.Value().(tensor.Tensor).Clone().(tensor.Tensor)
	}
	if head.Linear.Value() != nil && head.Bias.Value() != nil {
		out.Heads[headSlot] = headWeights{
			W: head.Linear.Value().(tensor.Tensor).Clone().(tensor.Tensor),
			B: head.Bias.Value().(tensor.Tensor).Clone().(tensor.Tensor),
		}
	}

	return out, nil
}

func evalTask(taskID int, w *weightsBundle, embDim int, xVal, yVal tensor.Tensor, opts evalOptions) (float64, []float32, error) {
	g := gorgonia.NewGraph()

	adapter := NewHybridLayer(g, embDim, embDim)
	// In this demo, we only use tasks 0 and 1
	adapter.AddTask(1)

	mask := tensor.New(tensor.WithShape(1, 1), tensor.Of(tensor.Float32))
	_ = mask.SetAt(float32(0), 0, 0)
	adapter.AddTaskMask(0, mask)
	adapter.AddTaskMask(1, mask)

	for _, edge := range adapter.Edges {
		if val, ok := w.Adapter[edge.Weight.Name()]; ok {
			gorgonia.Let(edge.Weight, val)
		}
	}

	head := NewClassificationHead(g, embDim, 2)
	headSlot := taskID
	if opts.SharedHead {
		headSlot = 0
	}
	hw, ok := w.Heads[headSlot]
	if !ok {
		return 0, nil, fmt.Errorf("missing head weights for slot %d (task %d)", headSlot, taskID)
	}
	gorgonia.Let(head.Linear, hw.W)
	gorgonia.Let(head.Bias, hw.B)

	x := gorgonia.NodeFromAny(g, xVal, gorgonia.WithName("X_eval"))
	y := gorgonia.NodeFromAny(g, yVal, gorgonia.WithName("Y_eval"))

	var features *gorgonia.Node
	var err error
	switch opts.ForwardMode {
	case forwardTaskScoped:
		features, err = adapter.ForwardTaskScoped(x, taskID)
	case forwardNaiveMixed:
		features, err = adapter.Forward(x, taskID)
	default:
		return 0, nil, fmt.Errorf("unknown forward mode: %d", opts.ForwardMode)
	}
	if err != nil {
		return 0, nil, err
	}
	logits, err := head.Forward(features)
	if err != nil {
		return 0, nil, err
	}
	probs, err := gorgonia.SoftMax(logits)
	if err != nil {
		return 0, nil, err
	}

	// run
	m := gorgonia.NewTapeMachine(g)
	defer m.Close()
	if err := m.RunAll(); err != nil {
		return 0, nil, err
	}

	p := probs.Value().Data().([]float32)
	acc := accuracyFromProbs(p, y.Value().Data().([]float32))
	cp := make([]float32, len(p))
	copy(cp, p)
	return acc, cp, nil
}

func getenvInt(key string, def int) int {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	i, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return i
}

func getenvFloat(key string, def float64) float64 {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return def
	}
	return f
}

func crossEntropyFromProbs(probs, yOneHot *gorgonia.Node) (*gorgonia.Node, error) {
	eps := float32(1e-7)
	epsN := gorgonia.NodeFromAny(probs.Graph(), eps)
	pSafe, err := gorgonia.Add(probs, epsN)
	if err != nil {
		return nil, err
	}
	logP, err := gorgonia.Log(pSafe)
	if err != nil {
		return nil, err
	}
	mul, err := gorgonia.HadamardProd(yOneHot, logP)
	if err != nil {
		return nil, err
	}
	sum, err := gorgonia.Sum(mul, 1)
	if err != nil {
		return nil, err
	}
	mean, err := gorgonia.Mean(sum)
	if err != nil {
		return nil, err
	}
	neg, err := gorgonia.Neg(mean)
	if err != nil {
		return nil, err
	}
	return neg, nil
}

func accuracyFromProbs(probs []float32, yOneHot []float32) float64 {
	// probs shape: (B,2)
	b := len(yOneHot) / 2
	correct := 0
	for i := 0; i < b; i++ {
		p0 := probs[i*2]
		p1 := probs[i*2+1]
		pred := 0
		if p1 > p0 {
			pred = 1
		}
		trueLabel := 0
		if yOneHot[i*2+1] > yOneHot[i*2] {
			trueLabel = 1
		}
		if pred == trueLabel {
			correct++
		}
	}
	return float64(correct) / float64(b)
}

func maxAbsDiff(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.NaN())
	}
	max := float32(0)
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > max {
			max = d
		}
	}
	return max
}
