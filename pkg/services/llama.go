// cpu/gpu inference API based on the llama.cpp engine

package services

import (
	"bytes"
	"context"
	"fmt"
	"iter"
	"log/slog"
	"path/filepath"
	"time"

	"github.com/hybridgroup/yzma/pkg/llama"
)

type inferenceType string

const (
	streamingInference inferenceType = "streaming"
	absoluteInference  inferenceType = "absolute"
)

type inferenceTask struct {
	prompt          []Message
	ctx             context.Context
	responseChannel chan string
	inference       inferenceType
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

func (ie *inferenceEngine) Infer(prompt []Message, ctx context.Context) chan string {
	ch := make(chan string)
	task := inferenceTask{
		prompt:          prompt,
		responseChannel: ch,
		ctx:             ctx,
		inference:       absoluteInference,
	}
	ie.promptChannel <- task
	return task.responseChannel

}

// starts new inference request and returns channel with the resulting tokens
// tokens are streamed for real-time systems
// blocks until the inference request starts
func (ie *inferenceEngine) StreamInference(prompt []Message, ctx context.Context) chan string {
	ch := make(chan string)
	task := inferenceTask{
		prompt:          prompt,
		responseChannel: ch,
		ctx:             ctx,
		inference:       streamingInference,
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
		fmt.Printf("Can't load llama .so libs: %s", err.Error())
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

func (lm *llamaModel) run() (chan inferenceTask, error) {
	inferenceChannel := make(chan inferenceTask)
	// gpu workers pool
	for _, c := range lm.contexts {
		go listenAndInfer(inferenceChannel, &c)
	}
	return inferenceChannel, nil
}

type llamaContext struct {
	model   llama.Model
	vocab   llama.Vocab
	context llama.Context
	batch   llama.Batch
	sampler llama.Sampler
	ct      *chatTemplate
	env     Env
}

func newLlamaContext(lm llamaModel) (llamaContext, error) {
	lc := &llamaContext{
		model: lm.model,
		vocab: lm.vocab,
		env:   lm.env,
	}
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

func (lc *llamaContext) createChatTemplate(m []Message, i inferenceType) {
	lc.ct = newChatTemplate(lc.env, m, i)
}

func (lc *llamaContext) genTokens(prompt string) iter.Seq[[]byte] {
	return func(yield func([]byte) bool) {
		fmt.Println("Starting inference")
		tokens := llama.Tokenize(lc.vocab, prompt, true, false)
		batch := llama.BatchGetOne(tokens)
		lc.batch = batch
		for {
			// one inference step
			llama.Decode(lc.context, batch)
			// grab the generated token from the KV
			token := llama.SamplerSample(lc.sampler, lc.context, -1)

			if llama.VocabIsEOG(lc.vocab, token) {
				return
			}
			decodedToken := make([]byte, 50)
			len := llama.TokenToPiece(lc.vocab, token, decodedToken, 0, true)
			decodedToken = decodedToken[:len]
			if !yield(decodedToken) {
				return
			}
			batch = llama.BatchGetOne([]llama.Token{token})
		}
	}
}

type chatTemplateType string

const (
	harmonyTemplate chatTemplateType = "harmony"
	chatMLTemplate  chatTemplateType = "chatML"
)

type chatTemplate struct {
	chatType     chatTemplateType
	inference    inferenceType
	messages     []Message
	lastStreamed string       // stores last streaming inference result
	rawAnswer    bytes.Buffer // stores entire AI output
}

func newChatTemplate(e Env, m []Message, i inferenceType) *chatTemplate {
	ct := &chatTemplate{messages: m, inference: i}
	switch e.ModelChatTemplate {
	case string(harmonyTemplate):
		ct.chatType = harmonyTemplate
	default:
		ct.chatType = chatMLTemplate
	}
	return ct
}

// serializes user facing messages
func (ct *chatTemplate) serializePrompt() string {
	switch ct.chatType {
	case harmonyTemplate:
		return createHarmonyPrompt(ct.messages)
	case chatMLTemplate:
		return createChatMLPrompt(ct.messages)
	default:
		panic(fmt.Sprintf("Can't serialize prompt, unkown chat template: %q", ct.chatType))
	}
}

func (ct *chatTemplate) addWord(word []byte) {
	ct.rawAnswer.Write(word)
	// fmt.Println(ct.rawAnswer.String())
	if ct.chatType == harmonyTemplate {
		// TODO: this doesn't have to be checked all of the time, fix
		ready := parseHarmonyStreamIsFinal(ct.rawAnswer.String())
		if !ready {
			ct.lastStreamed = ""
		} else {
			ct.lastStreamed = string(word)
		}
	}
}

func (ct *chatTemplate) isAbsolute() bool {
	return ct.inference == absoluteInference
}

func (ct *chatTemplate) isStreamed() bool {
	return ct.inference == streamingInference
}

func (ct *chatTemplate) getAnswer() string {
	switch ct.inference {
	case streamingInference:
		return ct.lastStreamed
	default:
		return parseHarmonyAbsIsFinal(ct.rawAnswer.String())
	}
}

func listenAndInfer(c chan inferenceTask, lc *llamaContext) {
	for {
		task := <-c
		fmt.Printf("Task %q received\n", task.prompt)
		lc.createChatTemplate(task.prompt, task.inference)
	gen:
		for word := range lc.genTokens(lc.ct.serializePrompt()) {
			select {
			case <-task.ctx.Done():
				slog.Info("Inference Cancelled")
				break gen
			default:
				lc.ct.addWord(word)
				if lc.ct.isAbsolute() {
					continue
				}
				if s := lc.ct.getAnswer(); s != "" {
					task.responseChannel <- s
				}
			}
		}
		if lc.ct.isAbsolute() {
			task.responseChannel <- lc.ct.getAnswer()
		}
		close(task.responseChannel)
		llama.Free(lc.context)
		lc.createContext()
	}

}
