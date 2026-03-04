package services

import (
	"fmt"
	"strings"
)

var systemChatMLPrompt = `You are a friendly assistant.`

var userChatMLPrompt = ``

var assistantChatMLPrompt = `
<|im_start|>assistant
`

// generates prompt dynamically for flexibiliy
// TODO: fix this
func createChatMLPrompt(msgs []Message) string {
	header := "<|im_start|>%s\n%s<|im_end|>\n"
	var prompt strings.Builder
	fmt.Fprintf(&prompt, header, "system", systemChatMLPrompt)

	// fmt.Fprintf(&prompt, header, "user", up)
	// fmt.Fprint(&prompt, assistantChatMLPrompt)

	// fmt.Println(prompt.String())
	return prompt.String()
}
