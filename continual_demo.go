//go:build continual_demo
// +build continual_demo

package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// continual_demo.go
// Run:
//   export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25
//   go run -tags continual_demo continual_demo.go layer.go head.go
//
// What it demonstrates:
//   1) Train Task 0
//   2) Freeze Task 0 params
//   3) Add Task 1 params and train only them
//   4) Show that Task 0 predictions stay the same (no forgetting) because
//      Task 0 parameters are not updated and Task 0 forward uses task-scoped edges.

const (
	seqLen    = 4
	embDim    = 16
	numClass  = 2
	epochsT0  = 300
	epochsT1  = 300
	learnRate = 0.01
)

type headWeights struct {
	W tensor.Tensor
	B tensor.Tensor
}

type taskWeights struct {
	Adapter map[string]tensor.Tensor // edge node name -> value
	Heads   map[int]headWeights      // taskID -> head weights
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Synthetic tasks:
	// Task0 label is encoded only in tokens [0..1]
	// Task1 label is encoded only in tokens [2..3]
	t0X, t0Y := makeTaskBatch(32, 0)
	t1X, t1Y := makeTaskBatch(32, 1)

	// Train Task 0
	w0, err := trainTask(0, nil, t0X, t0Y)
	if err != nil {
		log.Fatal(err)
	}
	// Evaluate Task 0 (baseline)
	t0PredsBefore, err := evalTask(0, w0, t0X)
	if err != nil {
		log.Fatal(err)
	}

	// Train Task 1, carrying Task 0 weights (but freezing them)
	// Train Task 1, carrying Task 0 weights (but freezing Task 0 edges)
	w01, err := trainTask(1, w0, t1X, t1Y)
	if err != nil {
		log.Fatal(err)
	}
	// Evaluate Task 0 again after Task 1 training
	t0PredsAfter, err := evalTask(0, w01, t0X)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("\n=== Continual Learning Check ===")
	fmt.Println("We evaluate Task0 before and after training Task1.")
	fmt.Println("If DTG-MA isolation works, Task0 predictions should match closely.")

	maxAbsDiff := float32(0)
	for i := range t0PredsBefore {
		d := t0PredsBefore[i] - t0PredsAfter[i]
		if d < 0 {
			d = -d
		}
		if d > maxAbsDiff {
			maxAbsDiff = d
		}
	}
	fmt.Printf("Max |Δprob| on Task0 after Task1 training: %.8f\n", maxAbsDiff)
	fmt.Println("(Should be ~0; tiny float noise may appear depending on backend.)")
}

func trainTask(taskID int, base *taskWeights, trainX, trainY tensor.Tensor) (*taskWeights, error) {
	g := gorgonia.NewGraph()

	adapter := NewHybridLayer(g, embDim, embDim)
	for t := 1; t <= taskID; t++ {
		adapter.AddTask(t)
	}

	// Add masks for all tasks we might run in this graph.
	// We deliberately separate token blocks to make the masking idea tangible.
	// Task0 attends only within tokens 0..1. Task1 attends only within tokens 2..3.
	adapter.AddTaskMask(0, makeBlockMask(0, 2))
	adapter.AddTaskMask(1, makeBlockMask(2, 4))

	// Freeze older tasks when training a new one.
	adapter.FreezeOldTasks(taskID)

	// Per-task head: we only train the head for the current task.
	head := NewClassificationHead(g, embDim, numClass)

	// Load existing weights (edges for all tasks up to taskID, plus head for this task if present)
	if base != nil {
		for _, edge := range adapter.Edges {
			if val, ok := base.Adapter[edge.Weight.Name()]; ok {
				gorgonia.Let(edge.Weight, val)
			}
		}
		if hw, ok := base.Heads[taskID]; ok {
			if hw.W != nil {
				gorgonia.Let(head.Linear, hw.W)
			}
			if hw.B != nil {
				gorgonia.Let(head.Bias, hw.B)
			}
		}
	}

	x := gorgonia.NodeFromAny(g, trainX, gorgonia.WithName("X"))
	y := gorgonia.NodeFromAny(g, trainY, gorgonia.WithName("Y"))

	features, err := adapter.ForwardTaskScoped(x, taskID)
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

	// MSE loss vs one-hot (kept simple for demo)
	diff, _ := gorgonia.Sub(probs, y)
	sqDiff, _ := gorgonia.Square(diff)
	loss, _ := gorgonia.Mean(sqDiff)

	learnables := []*gorgonia.Node{}
	// Train only non-frozen adapter edges + head
	learnables = append(learnables, adapter.GetTrainableNodes()...)
	learnables = append(learnables, head.Linear, head.Bias)

	if _, err := gorgonia.Grad(loss, learnables...); err != nil {
		return nil, err
	}

	solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(learnRate))
	machine := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(learnables...))
	defer machine.Close()

	epochs := epochsT0
	if taskID == 1 {
		epochs = epochsT1
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

	// Export weights
	out := &taskWeights{Adapter: map[string]tensor.Tensor{}, Heads: map[int]headWeights{}}
	if base != nil {
		for k, v := range base.Adapter {
			out.Adapter[k] = v.Clone().(tensor.Tensor)
		}
		for tid, hw := range base.Heads {
			out.Heads[tid] = headWeights{W: hw.W.Clone().(tensor.Tensor), B: hw.B.Clone().(tensor.Tensor)}
		}
	}
	for _, edge := range adapter.Edges {
		val := edge.Weight.Value()
		if val == nil {
			continue
		}
		out.Adapter[edge.Weight.Name()] = val.(tensor.Tensor).Clone().(tensor.Tensor)
	}
	if head.Linear.Value() != nil && head.Bias.Value() != nil {
		out.Heads[taskID] = headWeights{
			W: head.Linear.Value().(tensor.Tensor).Clone().(tensor.Tensor),
			B: head.Bias.Value().(tensor.Tensor).Clone().(tensor.Tensor),
		}
	}

	return out, nil
}

func evalTask(taskID int, w *taskWeights, xVal tensor.Tensor) ([]float32, error) {
	g := gorgonia.NewGraph()
	adapter := NewHybridLayer(g, embDim, embDim)
	// Add all tasks for which we have edges
	maxTask := 0
	for name := range w.Adapter {
		// quick parse: we only care about Task suffix, but keep it simple
		_ = name
	}
	// In this demo we only need tasks 0 and 1.
	maxTask = 1
	for t := 1; t <= maxTask; t++ {
		adapter.AddTask(t)
	}
	adapter.AddTaskMask(0, makeBlockMask(0, 2))
	adapter.AddTaskMask(1, makeBlockMask(2, 4))

	for _, edge := range adapter.Edges {
		if val, ok := w.Adapter[edge.Weight.Name()]; ok {
			gorgonia.Let(edge.Weight, val)
		}
	}

	head := NewClassificationHead(g, embDim, numClass)
	hw, ok := w.Heads[taskID]
	if !ok {
		return nil, fmt.Errorf("missing head weights for task %d", taskID)
	}
	gorgonia.Let(head.Linear, hw.W)
	gorgonia.Let(head.Bias, hw.B)

	x := gorgonia.NodeFromAny(g, xVal, gorgonia.WithName(fmt.Sprintf("X_eval_task_%d", taskID)))
	feat, err := adapter.ForwardTaskScoped(x, taskID)
	if err != nil {
		return nil, err
	}
	logits, err := head.Forward(feat)
	if err != nil {
		return nil, err
	}
	probs, err := gorgonia.SoftMax(logits)
	if err != nil {
		return nil, err
	}

	m := gorgonia.NewTapeMachine(g)
	defer m.Close()
	if err := m.RunAll(); err != nil {
		return nil, err
	}

	out := probs.Value().Data().([]float32)
	cp := make([]float32, len(out))
	copy(cp, out)
	return cp, nil
}

func makeTaskBatch(batch int, taskID int) (tensor.Tensor, tensor.Tensor) {
	// X: (B, S, D)
	data := make([]float32, batch*seqLen*embDim)
	labels := make([]float32, batch*numClass)

	for b := 0; b < batch; b++ {
		label := b % 2
		labels[b*numClass+label] = 1

		for s := 0; s < seqLen; s++ {
			for d := 0; d < embDim; d++ {
				idx := (b*seqLen*embDim + s*embDim + d)
				v := rand.Float32()*0.1 - 0.05 // small noise

				// Encode label in different token blocks per task.
				if taskID == 0 {
					if s < 2 {
						if label == 0 {
							v += 1.0
						} else {
							v -= 1.0
						}
					}
				} else {
					if s >= 2 {
						if label == 0 {
							v += 1.0
						} else {
							v -= 1.0
						}
					}
				}
				data[idx] = v
			}
		}
	}

	x := tensor.New(tensor.WithShape(batch, seqLen, embDim), tensor.WithBacking(data))
	y := tensor.New(tensor.WithShape(batch, numClass), tensor.WithBacking(labels))
	return x, y
}

func makeBlockMask(from, to int) tensor.Tensor {
	// Allow attention only within [from,to)
	// Everything else gets a very negative value.
	mask := tensor.New(tensor.WithShape(seqLen, seqLen), tensor.Of(tensor.Float32))
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			v := float32(-1e9)
			if i >= from && i < to && j >= from && j < to {
				v = 0
			}
			_ = mask.SetAt(v, i, j)
		}
	}
	return mask
}
