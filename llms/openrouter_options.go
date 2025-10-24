package llms

type OpenRouterOptions struct {
	ReasoningEffort    string `json:"reasoning_effort,omitempty"`
	ReasoningMaxTokens int    `json:"reasoning_max_tokens,omitempty"`
}
