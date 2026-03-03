// Dependency Injection API

package services

type DI struct {
	Env             Env
	InferenceEngine Inferer
}

func NewDI() DI {
	var di DI

	env := NewEnv()
	di.Env = env

	ie := NewInferenceEngine(env)
	if err := ie.Init(); err != nil {
		panic(err.Error())
	}
	di.InferenceEngine = ie
	return di
}
