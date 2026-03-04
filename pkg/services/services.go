package services

import "context"

// text to text inference interface
type Inferer interface {
	Infer(prompt []Message, ctx context.Context) chan string
	StreamInference(prompt []Message, ctx context.Context) chan string
	Init() error
	Close()
}

type Message struct {
	user string
	ai   string
}

func NewMessage(fromUser string, fromAI string) Message {
	return Message{user: fromUser, ai: fromAI}
}
