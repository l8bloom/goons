package services

import (
	"fmt"
	"strings"
	"time"

	"github.com/kultivator-consulting/goharmony"
)

var systemHarmonyPrompt = `
<|start|>user<|message|>
Knowledge cutoff: 2024-06
Current date: %s

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>
`

var developerHarmonyPrompt = `
<|start|>developer<|message|>
# Instructions
You are a friendly assistant.
Answer questions concisely, if you
are not sure ask for more details or say "I don't know".<|end|>
`

var userHarmonyPrompt = `
<|start|>developer<|message|>
%s<|end|>
`

var assistantHarmonyPromptStart = `
<|start|>assistant
`

var assistantHarmonyPromptComplete = `
<|start|>assistant<|channel|>final<|message|>%s<|return|>
`

// generates prompt dynamically for flexibiliy
func createHarmonyPrompt(msgs []Message) string {
	var prompt strings.Builder
	now := time.Now().Format("2006-01")
	fmt.Fprintf(&prompt, systemHarmonyPrompt, now)
	fmt.Fprint(&prompt, developerHarmonyPrompt)
	for _, msg := range msgs {
		fmt.Fprintf(&prompt, userHarmonyPrompt, msg.user)
		// TODO: fix for ai msg which errored out,
		// this check should be on the last received message instead
		if msg.ai == "" {
			fmt.Fprint(&prompt, assistantHarmonyPromptStart)
			fmt.Println(prompt.String())
			return prompt.String()
		}
		fmt.Fprintf(&prompt, assistantHarmonyPromptComplete, msg.ai)
	}
	fmt.Println(prompt.String())
	return prompt.String()
}

// only final messages atm, no reasoning

// bug(?) in goharmony for parsing streaming answers
func parseHarmonyStreamIsFinal(answer string) bool {
	// naive implementation atm
	finalHeader := `<|start|>assistant<|channel|>final<|message|>`
	return strings.Contains(answer, finalHeader) && !strings.HasSuffix(answer, finalHeader)
}

func parseHarmonyAbsIsFinal(answer string) string {
	parser := goharmony.NewParser()
	return parser.ExtractFinalMessage(answer)
}
