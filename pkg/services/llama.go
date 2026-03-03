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

// text to text inference interface
type Inferer interface {
	Infer(prompt string) string
	StreamInference(prompt string) chan string
	Init() error
	Close()
}

type inferenceTask struct {
	prompt          string
	responseChannel chan string
}

// one implementation of Inferer interface
type inferenceEngine struct {
	promptChannel chan inferenceTask
	model         *llamaModel
	env           Env
}

func NewInferenceEngine(env Env) *inferenceEngine {
	return &inferenceEngine{env: env}
}

func (i *inferenceEngine) Init() error {
	err := i.initLlama()
	if err != nil {
		return fmt.Errorf("Error while initializing Inference service. %s", err)
	}
	return nil
}

func (ie *inferenceEngine) Close() {
	llama.Close()
}

func (ie *inferenceEngine) Infer(prompt string) string {
	return "abc"
}

// starts new inference request and returns channel with the resulting tokens
// tokens are streamed for real-time systems
// blocks until the request inference starts
func (ie *inferenceEngine) StreamInference(prompt string) chan string {
	ch := make(chan string)
	task := inferenceTask{
		prompt:          prompt,
		responseChannel: ch,
	}

	ie.promptChannel <- task
	return task.responseChannel
}

// loads llama .so libs and inits it
func (i *inferenceEngine) initLlama() error {
	st := time.Now()
	if err := i.loadLlamaLibs(); err != nil {
		return fmt.Errorf("initLlama failed to load llama.so. %s", err)
	}
	slog.Info(fmt.Sprintf("Llama libs loaded in %.0e sec\n", time.Since(st).Seconds()))
	llama.Init()

	llama.LogSet(llama.LogSilent()) // TODO: dedice on llama's log content

	lm := newLlamaModel(i.env)
	ch, err := lm.deployModel()
	if err != nil {
		return fmt.Errorf("initLlama failed to deploy models.  %s", err)
	}
	i.model = lm
	i.promptChannel = ch
	slog.Info(fmt.Sprintf("Llama engine initialized in %.0e sec\n", time.Since(st).Seconds()))
	return nil
}

func (i *inferenceEngine) loadLlamaLibs() error {
	st := time.Now()

	err := llama.Load(i.env.LlamaLibs)
	if err != nil {
		slog.Error("Can't load llama .so libs: %s", err)
		return err
	}
	slog.Info(fmt.Sprintf("llama.cpp .so libs loaded in %.0e sec\n", time.Since(st).Seconds()))
	return nil
}

type llamaModel struct {
	model    llama.Model
	vocab    llama.Vocab
	contexts []llamaContext
	env      Env
}

func newLlamaModel(env Env) *llamaModel {
	return &llamaModel{env: env}
}

// load model to gpu and return the prompt channel
func (lm *llamaModel) deployModel() (chan inferenceTask, error) {
	if err := lm.loadModel(); err != nil {
		return nil, err
	}
	lm.loadVocab()
	if err := lm.createContexts(); err != nil {
		return nil, err
	}
	c, err := lm.run()
	if err != nil {
		return nil, err
	}
	return c, nil
}

func (lm *llamaModel) loadModel() error {
	st := time.Now()
	modelPath := filepath.Join(lm.env.ModelDir, lm.env.ModelName)
	model, err := llama.ModelLoadFromFile(modelPath, llama.ModelDefaultParams())
	if err != nil {
		return fmt.Errorf("initLlama failed to load llama.so. %s", err)
	}
	slog.Info(fmt.Sprintf("%q model loaded in %.0e sec\n", lm.env.ModelName, time.Since(st).Seconds()))
	lm.model = model
	return nil
}

func (lm *llamaModel) loadVocab() {
	st := time.Now()
	vocab := llama.ModelGetVocab(lm.model)
	slog.Info(fmt.Sprintf("model's vocabular fetched in %.0e sec\n", time.Since(st).Seconds()))
	lm.vocab = vocab
}

func (lm *llamaModel) createContexts() error {
	st := time.Now()
	contexts := make([]llamaContext, 0, 10)
	for range lm.env.ModelNCtx {
		context, err := newLlamaContext(*lm)
		if err != nil {
			return nil
		}
		contexts = append(contexts, context)
	}
	slog.Info(fmt.Sprintf("Created %d model's contexts in %.0e sec\n", lm.env.ModelNCtx, time.Since(st).Seconds()))
	lm.contexts = contexts
	return nil
}

type llamaContext struct {
	model   llama.Model
	vocab   llama.Vocab
	context llama.Context
	batch   llama.Batch
	sampler llama.Sampler
	env     Env
}

func newLlamaContext(lm llamaModel) (llamaContext, error) {
	lc := new(llamaContext{model: lm.model, vocab: lm.vocab, env: lm.env})
	if err := lc.createContext(); err != nil {
		return *lc, err
	}
	lc.loadSampler()
	return *lc, nil
}

func (lc *llamaContext) createContext() error {
	st := time.Now()
	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = uint32(lc.env.ModelCtxSize)

	ctx, err := llama.InitFromModel(lc.model, ctxParams)
	slog.Info(fmt.Sprintf("model's context created in %.0e sec\n", time.Since(st).Seconds()))
	if err != nil {
		return fmt.Errorf("initLlama failed to create new context %s", err)
	}
	lc.context = ctx
	return nil
}

func (lc *llamaContext) loadSampler() {
	st := time.Now()
	sampler := llama.SamplerChainInit(llama.SamplerChainDefaultParams())
	llama.SamplerChainAdd(sampler, llama.SamplerInitGreedy())
	slog.Info(fmt.Sprintf("Sampler created in %.0e sec\n", time.Since(st).Seconds()))
	lc.sampler = sampler
}

func (lm *llamaModel) run() (chan inferenceTask, error) {
	inferenceChannel := make(chan inferenceTask)
	for _, c := range lm.contexts {
		go listenAndInfer(inferenceChannel, &c)
	}
	return inferenceChannel, nil
}

func genTokens(prompt string, ctx *llamaContext) func(func(string) bool) {
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
			fmt.Println(answer.String())
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

func listenAndInfer(c chan inferenceTask, lc *llamaContext) {
	for {
		select {
		case task := <-c:
			task.prompt = CreateGPTOSSPrompt("", "", task.prompt)
			for s := range genTokens(task.prompt, lc) {
				task.responseChannel <- s
			}
			close(task.responseChannel)
			llama.Free(lc.context)
			lc.createContext()
		}
	}
}
