// cpu/gpu inference API based on the llama.cpp engine

package services

import (
	"fmt"
	"log/slog"
	"path/filepath"
	"strings"
	"time"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/kultivator-consulting/goharmony"
)

type Inferer interface {
	Infer(prompt string) string
	StreamInference(prompt string) chan string
	Init() error
	Close()
}

func NewInferenceEngine(env EnvProvider) *inferenceEngine {
	return &inferenceEngine{env: env}
}

type inferenceTask struct {
	prompt          string
	responseChannel chan string
}

// one implementation of Inferer interface
type inferenceEngine struct {
	promptChannel chan inferenceTask
	modelLoaded   bool
	env           EnvProvider
}

func (i *inferenceEngine) Init() error {
	err := i.initLlama()
	if err != nil {
		return fmt.Errorf("Error while initializing Inference service. %s", err)
	}
	return nil
}

func (ie inferenceEngine) Close() {
	llama.Close()
}

func (ie inferenceEngine) Infer(prompt string) string {
	return "abc"
}

// sends new inference request and returns channel with the resulting tokens
// tokens are streamed for real-time systems
func (ie inferenceEngine) StreamInference(prompt string) chan string {
	ch := make(chan string)
	task := inferenceTask{
		prompt:          prompt,
		responseChannel: ch,
	}

	ie.promptChannel <- task
	return task.responseChannel
}

type llamaModel struct {
	model llama.Model
	vocab llama.Vocab
}

// kind of hardcoded for now
func (lm *llamaModel) loadModel(e EnvProvider) error {
	if lm.model != 0 {
		return nil
	}
	modelDir := e.GetEnv().ModelDir
	modelName := e.GetEnv().ModelName
	llama.LogSet(llama.LogSilent())
	st := time.Now()
	model, err := llama.ModelLoadFromFile(filepath.Join(modelDir, modelName), llama.ModelDefaultParams())
	if err != nil {
		return fmt.Errorf("initLlama failed to load llama.so. %s", err)
	}
	slog.Info(fmt.Sprintf("%q model loaded in %.0e sec\n", modelName, time.Since(st).Seconds()))
	lm.model = model
	return nil
}

func (lm *llamaModel) loadVocab() {
	if lm.vocab != 0 {
		return
	}
	st := time.Now()
	vocab := llama.ModelGetVocab(lm.model)
	slog.Info(fmt.Sprintf("model's vocabular fetched in %.0e sec\n", time.Since(st).Seconds()))
	lm.vocab = vocab
}

type llamaContext struct {
	llamaModel
	context llama.Context
	batch   llama.Batch
	sampler llama.Sampler
}

func newContext(lm llamaModel) (llamaContext, error) {
	var lc llamaContext
	lc.llamaModel = lm
	if err := lc.createContext(); err != nil {
		return lc, err
	}
	lc.loadSampler()
	return lc, nil
}

func (lc *llamaContext) createContext() error {
	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = 20_000
	st := time.Now()
	ctx, err := llama.InitFromModel(lc.model, ctxParams)
	slog.Info(fmt.Sprintf("model's context created in %.0e sec\n", time.Since(st).Seconds()))
	if err != nil {
		return fmt.Errorf("initLlama failed to create new context %s", err)
	}
	lc.context = ctx
	return nil
}

func (lc *llamaContext) loadSampler() {
	if lc.sampler != 0 {
		return
	}
	st := time.Now()
	sampler := llama.SamplerChainInit(llama.SamplerChainDefaultParams())
	llama.SamplerChainAdd(sampler, llama.SamplerInitGreedy())
	slog.Info(fmt.Sprintf("Sampler created in %.0e sec\n", time.Since(st).Seconds()))
	lc.sampler = sampler
}

// loads llama .so libs and inits it
func (i *inferenceEngine) initLlama() error {
	st := time.Now()
	if err := i.loadLlamaLibs(); err != nil {
		return fmt.Errorf("initLlama failed to load llama.so. %s", err)
	}
	slog.Info(fmt.Sprintf("Llama libs loaded in %.0e sec\n", time.Since(st).Seconds()))

	st = time.Now()
	llama.Init()
	slog.Info(fmt.Sprintf("Llama engine initialized in %.0e sec\n", time.Since(st).Seconds()))

	ch, err := deployModels(i.env)
	if err != nil {
		return fmt.Errorf("initLlama failed to deploy models.  %s", err)
	}
	i.promptChannel = ch
	i.modelLoaded = true
	return nil
}

func (i *inferenceEngine) loadLlamaLibs() error {
	st := time.Now()

	err := llama.Load(i.env.GetEnv().LlamaLibs)
	if err != nil {
		slog.Error("Can't load llama .so libs: %s", err)
		return err
	}
	slog.Info(fmt.Sprintf("llama.cpp .so libs loaded in %.0e sec\n", time.Since(st).Seconds()))
	return nil
}

// only one atm but with two contexts
func deployModels(e EnvProvider) (chan inferenceTask, error) {
	var lm llamaModel
	if err := lm.loadModel(e); err != nil {
		slog.Error("Can't load model: %s", err)
		return nil, err
	}
	lm.loadVocab()
	inferenceChannel := make(chan inferenceTask)
	// only two goroutines
	for range 5 {
		go listenAndInfer(inferenceChannel, lm)
	}
	return inferenceChannel, nil
}

func genTokens(prompt string, ctx llamaContext) func(func(string) bool) {
	return func(yield func(string) bool) {
		parser := goharmony.NewParser()
		var analysis, answer, final_answer strings.Builder
		fmt.Println("Starting inference")
		// st := time.Now()
		tokens := llama.Tokenize(ctx.vocab, prompt, true, false)
		batch := llama.BatchGetOne(tokens)
		ctx.batch = batch
		for {
			llama.Decode(ctx.context, batch)                           // one inference step
			token := llama.SamplerSample(ctx.sampler, ctx.context, -1) // grab the generated token from the KV

			if llama.VocabIsEOG(ctx.vocab, token) {
				return
			}
			decodedToken := make([]byte, 50)
			len := llama.TokenToPiece(ctx.vocab, token, decodedToken, 0, true)
			decodedToken = decodedToken[:len]
			answer.WriteString(string(decodedToken))
			// fmt.Println(answer.String())
			batch = llama.BatchGetOne([]llama.Token{token})
			messages, err := parser.ParseResponse(answer.String())
			if err != nil {
				fmt.Println("error: ", err)
				continue // Wait for more data
			}
			if !strings.Contains(answer.String(), ">final<") {
				continue
			}
			for _, msg := range messages {
				if msg.Channel == goharmony.ChannelFinal || msg.Content == "" {
					final_answer.Write(decodedToken)
					if !yield(string(decodedToken)) {
						return
					}
				}
				if msg.Channel == goharmony.ChannelAnalysis || msg.Content == "" {
					analysis.Write(decodedToken)
				}
			}
		}
	}
}

func listenAndInfer(c chan inferenceTask, lm llamaModel) {
	for {
		ctx, err := newContext(lm)
		if err != nil {
			slog.Warn("Error creating new llama context; skipping", err)
			return
		}
		select {
		case task := <-c:
			task.prompt = CreateGPTOSSPrompt("", "", task.prompt)
			for s := range genTokens(task.prompt, ctx) {
				task.responseChannel <- s
			}
			close(task.responseChannel)
			llama.Free(ctx.context)
		}
	}
}
