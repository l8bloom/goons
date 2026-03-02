package main

import (
	"goons/pkg/services"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"
)

func StreamingInferenceHandler(i services.Inferer) func(c *gin.Context) {
	return func(c *gin.Context) {
		var q AIPrompt
		c.ShouldBindJSON(&q)
		resChannel := i.StreamInference(q.Question)
		c.Stream(func(w io.Writer) bool {
			select {
			case word, ok := <-resChannel:
				if !ok {
					return false
				}
				w.Write([]byte(word))
				return true
			case <-c.Request.Context().Done():
				return false
			}
		})
	}
}

func health(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"message": "OK"})
}

func registerHandlers(r *gin.Engine, di services.DI) {
	r.GET("/health", health)
	r.POST("/infer", StreamingInferenceHandler(di.InferenceEngine))
}
