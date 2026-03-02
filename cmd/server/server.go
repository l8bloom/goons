package main

import (
	"goons/pkg/services"

	"github.com/gin-gonic/gin"
)

func main() {
	di := services.NewDI()
	defer di.InferenceEngine.Close()

	r := gin.Default()
	registerHandlers(r, di)

	r.Run(":9000")
}
