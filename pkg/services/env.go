package services

import (
	"fmt"
	"os"
	"reflect"

	"github.com/caarlos0/env/v11"
	"github.com/mitchellh/go-homedir"
)

type Env struct {
	LlamaLibs    string `env:"LLAMA_LIBS,notEmpty" envDefault:"~/Projects/local_ai/llama.cpp/builds/vulkan/bin"`
	ModelDir     string `env:"MODEL_DIR,notEmpty" envDefault:"~/.cache/llama.cpp/"`
	ModelName    string `env:"MODEL_NAME,notEmpty" envDefault:"ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf"`
	ModelNCtx    int    `env:"MODEL_NCTX" envDefault:"2"`
	ModelCtxSize int    `env:"MODEL_CTX_SIZE" envDefault:"20000"`
}

func NewEnv() Env {
	appEnv, err := env.ParseAs[Env]()
	if err != nil {
		fmt.Println("Cane parse the OS env vars.", err)
		os.Exit(1)
	}
	envVal := reflect.ValueOf(&appEnv)
	envVal = envVal.Elem()
	for _, field := range envVal.Fields() {
		if field.Kind() != reflect.String || !field.CanSet() {
			continue
		}
		expanded, err := homedir.Expand(field.String())
		if err != nil {
			fmt.Println("Can't expand env var", err)
			os.Exit(1)
		}
		field.SetString(expanded)
	}
	return appEnv
}
