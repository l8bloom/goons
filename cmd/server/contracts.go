package main

type AIPrompt struct {
	Question string `json:"question" binding:"required,min=3"`
}
