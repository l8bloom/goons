package services

import (
	"fmt"
	"strings"
)

var SystemGPTOSSprompt = `
Knowledge cutoff: 2024-06
Current date: 2026-03-02

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.
`

var DeveloperGPTOSSprompt = `
# Instructions
Provide "N\A" for all questions answered in English and German.
`

var UserGPTOSSprompt = ``

var AssistantGPTOSSprompt = `
<|start|>assistant
`

// generates prompt dynamically for flexibiliy
func CreateGPTOSSPrompt(sp string, dp string, up string) string {
	header := "<|start|>%s<|message|>%s<|end|>"
	var prompt strings.Builder
	if sp == "" {
		fmt.Fprintf(&prompt, header, "system", SystemGPTOSSprompt)
	} else {
		fmt.Fprintf(&prompt, header, "system", sp)
	}

	if dp == "" {
		fmt.Fprintf(&prompt, header, "developer", DeveloperGPTOSSprompt)
	} else {
		fmt.Fprintf(&prompt, header, "developer", dp)
	}

	fmt.Fprintf(&prompt, header, "user", up)
	fmt.Fprint(&prompt, AssistantGPTOSSprompt)

	fmt.Println(prompt.String())
	return prompt.String()
}
