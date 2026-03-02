package services

import (
	"fmt"
	"os"
	"reflect"

	"github.com/caarlos0/env/v11"
	"github.com/mitchellh/go-homedir"
)

type EnvProvider interface {
	GetEnv() Env
}

type Env struct {
	loaded    bool
	LlamaLibs string `env:"LLAMA_LIBS,notEmpty" envDefault:"~/Projects/local_ai/llama.cpp/builds/vulkan/bin"`
	ModelDir  string `env:"MODEL_DIR,notEmpty" envDefault:"~/.cache/llama.cpp/"`
	ModelName string `env:"MODEL_NAME,notEmpty" envDefault:"ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf"`
}

func (e Env) GetEnv() Env {
	if e.loaded {
		return e
	}
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
	appEnv.loaded = true
	return appEnv
}
