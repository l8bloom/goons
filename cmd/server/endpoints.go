package main

import (
	"context"
	"goons/pkg/services"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"
)

func streamingInferenceHandler(i services.Inferer) func(c *gin.Context) {
	return func(c *gin.Context) {
		var q AIPrompt
		c.ShouldBindJSON(&q)
		ctx, cancel := context.WithCancel(c.Copy().Request.Context())
		defer cancel()
		msgs := make([]services.Message, 0, 10)
		msgs = append(msgs, services.NewMessage(q.Question, ""))
		resChannel := i.StreamInference(msgs, ctx)
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

func absoluteInferenceHandler(i services.Inferer) func(c *gin.Context) {
	return func(c *gin.Context) {
		var q AIPrompt
		c.ShouldBindJSON(&q)
		ctx, cancel := context.WithCancel(c.Copy().Request.Context())
		defer cancel()
		msgs := make([]services.Message, 0, 10)
		msgs = append(msgs, services.NewMessage(q.Question, ""))
		resChannel := i.Infer(msgs, ctx)
		answer := <-resChannel
		c.JSON(http.StatusOK, gin.H{"final": answer})
	}
}

func health(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"message": "OK"})
}

func registerHandlers(r *gin.Engine, di services.DI) {
	r.GET("/health", health)
	r.POST("/infer", streamingInferenceHandler(di.InferenceEngine))
	r.POST("/absolute", absoluteInferenceHandler(di.InferenceEngine))
}
