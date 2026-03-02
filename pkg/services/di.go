// Dependency Injection API

package services

type DI struct {
	Env             EnvProvider
	InferenceEngine Inferer
}

func NewDI() DI {
	var di DI
	var env Env
	di.Env = env

	ie := NewInferenceEngine(env)
	if err := ie.Init(); err != nil {
		panic(err.Error())
	}
	di.InferenceEngine = ie
	return di
}
