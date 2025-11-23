// Synth Task App Example - Go Implementation
//
// A minimal but complete Task App implementing the Synth contract for prompt optimization.
//
// ## Building
//
//	go build -o synth-task-app
//
// ## Running
//
//	./synth-task-app
//	ENVIRONMENT_API_KEY=secret PORT=8001 ./synth-task-app
//
// ## Cross-compiling
//
//	GOOS=linux GOARCH=amd64 go build -o synth-task-app-linux
//	GOOS=darwin GOARCH=arm64 go build -o synth-task-app-macos
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// =============================================================================
// Types (matching OpenAPI contract)
// =============================================================================

type Sample struct {
	Text  string `json:"text"`
	Label string `json:"label"`
}

type Dataset struct {
	Samples []Sample `json:"samples"`
	Labels  []string `json:"labels"`
}

type RolloutRequest struct {
	RunID  string     `json:"run_id"`
	Env    EnvSpec    `json:"env"`
	Policy PolicySpec `json:"policy"`
	Mode   string     `json:"mode"`
}

type EnvSpec struct {
	Seed   *int64                 `json:"seed"`
	Config map[string]interface{} `json:"config"`
}

type PolicySpec struct {
	PolicyID   string                 `json:"policy_id"`
	PolicyName string                 `json:"policy_name"`
	Config     map[string]interface{} `json:"config"`
}

type RolloutResponse struct {
	RunID        string       `json:"run_id"`
	Trajectories []Trajectory `json:"trajectories"`
	Metrics      Metrics      `json:"metrics"`
	Aborted      bool         `json:"aborted"`
	OpsExecuted  int          `json:"ops_executed"`
}

type Trajectory struct {
	EnvID        string                 `json:"env_id"`
	PolicyID     string                 `json:"policy_id"`
	Steps        []Step                 `json:"steps"`
	Length       int                    `json:"length"`
	InferenceURL string                 `json:"inference_url"`
}

type Step struct {
	Obs       map[string]interface{} `json:"obs"`
	ToolCalls []ToolCall             `json:"tool_calls"`
	Reward    float64                `json:"reward"`
	Done      bool                   `json:"done"`
	Info      map[string]interface{} `json:"info"`
}

type ToolCall struct {
	ID       string   `json:"id"`
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

type Function struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type Metrics struct {
	EpisodeReturns []float64 `json:"episode_returns"`
	MeanReturn     float64   `json:"mean_return"`
	NumSteps       int       `json:"num_steps"`
	NumEpisodes    int       `json:"num_episodes"`
	OutcomeScore   float64   `json:"outcome_score"`
}

type HealthResponse struct {
	Healthy bool `json:"healthy"`
}

type ErrorResponse struct {
	Detail string `json:"detail"`
}

// TaskInfo types
type TaskInfo struct {
	Task        TaskSpec      `json:"task"`
	Environment string        `json:"environment"`
	Dataset     DatasetInfo   `json:"dataset"`
	Rubric      RubricSpec    `json:"rubric"`
	Inference   InferenceSpec `json:"inference"`
	Limits      LimitsSpec    `json:"limits"`
}

type TaskSpec struct {
	TaskID      string `json:"task_id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Version     string `json:"version"`
}

type DatasetInfo struct {
	Seeds      []int `json:"seeds"`
	TrainCount int   `json:"train_count"`
	ValCount   int   `json:"val_count"`
	TestCount  int   `json:"test_count"`
}

type RubricSpec struct {
	ScoringCriteria string    `json:"scoring_criteria"`
	MetricPrimary   string    `json:"metric_primary"`
	MetricRange     []float64 `json:"metric_range"`
}

type InferenceSpec struct {
	Mode           string   `json:"mode"`
	SupportedTools []string `json:"supported_tools"`
}

type LimitsSpec struct {
	MaxResponseTokens int `json:"max_response_tokens"`
	TimeoutSeconds    int `json:"timeout_seconds"`
}

// LLM types
type ChatRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Tools       []Tool        `json:"tools"`
	ToolChoice  string        `json:"tool_choice"`
	Temperature float64       `json:"temperature"`
	MaxTokens   int           `json:"max_tokens"`
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

type ChatResponse struct {
	Choices []Choice `json:"choices"`
}

type Choice struct {
	Message ResponseMessage `json:"message"`
}

type ResponseMessage struct {
	Content   string             `json:"content"`
	ToolCalls []ResponseToolCall `json:"tool_calls"`
}

type ResponseToolCall struct {
	ID       string           `json:"id"`
	Function ResponseFunction `json:"function"`
}

type ResponseFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// =============================================================================
// Globals
// =============================================================================

var dataset Dataset
var apiKey string
var llmApiKey string

// =============================================================================
// Dataset Loading
// =============================================================================

func loadDataset() Dataset {
	// Try loading from file
	paths := []string{
		"../data/banking77.json",
		"data/banking77.json",
		"../../data/banking77.json",
	}

	for _, path := range paths {
		absPath, _ := filepath.Abs(path)
		data, err := os.ReadFile(absPath)
		if err == nil {
			var ds Dataset
			if err := json.Unmarshal(data, &ds); err == nil {
				log.Printf("Loaded %d samples from %s", len(ds.Samples), absPath)
				return ds
			}
		}
	}

	// Fallback to embedded samples
	log.Println("Dataset file not found, using embedded samples")
	return Dataset{
		Samples: []Sample{
			{Text: "How do I reset my PIN?", Label: "change_pin"},
			{Text: "My card hasn't arrived yet", Label: "card_arrival"},
			{Text: "I want to cancel my card", Label: "terminate_account"},
			{Text: "How do I activate my new card?", Label: "activate_my_card"},
			{Text: "I need to dispute a transaction", Label: "transaction_charged_twice"},
			{Text: "Can I get a refund?", Label: "request_refund"},
			{Text: "How do I transfer money?", Label: "transfer_into_account"},
			{Text: "I lost my card", Label: "lost_or_stolen_card"},
			{Text: "Is there a fee for this?", Label: "transfer_fee_charged"},
		},
		Labels: []string{
			"change_pin", "card_arrival", "terminate_account", "activate_my_card",
			"transaction_charged_twice", "request_refund", "transfer_into_account",
			"lost_or_stolen_card", "transfer_fee_charged",
		},
	}
}

func getSample(seed int) Sample {
	return dataset.Samples[seed%len(dataset.Samples)]
}

// =============================================================================
// Prompt Rendering
// =============================================================================

func renderTemplate(template string, placeholders map[string]string) string {
	result := template
	for key, value := range placeholders {
		pattern := regexp.MustCompile(`\{` + regexp.QuoteMeta(key) + `\}`)
		result = pattern.ReplaceAllString(result, value)
	}
	return result
}

func buildMessages(policyConfig map[string]interface{}, sample Sample) []ChatMessage {
	placeholders := map[string]string{
		"query":   sample.Text,
		"intents": strings.Join(dataset.Labels, ", "),
	}

	// Check for prompt_template
	if promptTemplate, ok := policyConfig["prompt_template"].(map[string]interface{}); ok {
		var sections []interface{}
		if s, ok := promptTemplate["prompt_sections"].([]interface{}); ok {
			sections = s
		} else if s, ok := promptTemplate["sections"].([]interface{}); ok {
			sections = s
		}

		if sections != nil {
			var messages []ChatMessage
			for _, s := range sections {
				section := s.(map[string]interface{})
				role := "user"
				if r, ok := section["role"].(string); ok {
					role = r
				}

				content := ""
				if c, ok := section["content"].(string); ok {
					content = c
				} else if p, ok := section["pattern"].(string); ok {
					content = p
				}

				messages = append(messages, ChatMessage{
					Role:    role,
					Content: renderTemplate(content, placeholders),
				})
			}
			return messages
		}
	}

	// Default messages
	return []ChatMessage{
		{
			Role:    "system",
			Content: "You are an expert banking assistant. Classify queries using the classify tool.",
		},
		{
			Role:    "user",
			Content: fmt.Sprintf("Query: %s\nIntents: %s\nClassify this query.", sample.Text, strings.Join(dataset.Labels, ", ")),
		},
	}
}

// =============================================================================
// LLM Client
// =============================================================================

func callLLM(inferenceURL, model string, messages []ChatMessage, providedKey string) (*ChatResponse, error) {
	// Build URL - handle query params correctly
	// inference_url may be "http://host/path?query" - we need "http://host/path/chat/completions?query"
	var url string
	if queryIdx := strings.Index(inferenceURL, "?"); queryIdx != -1 {
		base := strings.TrimSuffix(inferenceURL[:queryIdx], "/")
		query := inferenceURL[queryIdx:]
		url = base + "/chat/completions" + query
	} else {
		url = strings.TrimSuffix(inferenceURL, "/") + "/chat/completions"
	}

	log.Printf("LLM call: inference_url=%s full_url=%s model=%s", inferenceURL, url, model)

	tool := Tool{
		Type: "function",
		Function: ToolFunction{
			Name:        "classify",
			Description: "Classify the customer query into an intent category",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"intent": map[string]interface{}{
						"type":        "string",
						"description": "The classified intent",
					},
				},
				"required": []string{"intent"},
			},
		},
	}

	reqBody := ChatRequest{
		Model:       model,
		Messages:    messages,
		Tools:       []Tool{tool},
		ToolChoice:  "required",
		Temperature: 0,
		MaxTokens:   100,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	if providedKey != "" {
		req.Header.Set("X-API-Key", providedKey)
	}
	// Add Bearer auth for OpenAI-compatible APIs
	if llmApiKey != "" {
		req.Header.Set("Authorization", "Bearer "+llmApiKey)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("LLM request failed: %d - %s", resp.StatusCode, string(body))
	}

	var chatResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, err
	}

	return &chatResp, nil
}

func extractPrediction(resp *ChatResponse) (string, []ToolCall) {
	var toolCalls []ToolCall
	var predicted string

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		for _, call := range choice.Message.ToolCalls {
			toolCalls = append(toolCalls, ToolCall{
				ID:   call.ID,
				Type: "function",
				Function: Function{
					Name:      call.Function.Name,
					Arguments: call.Function.Arguments,
				},
			})

			if call.Function.Name == "classify" {
				var args map[string]interface{}
				if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err == nil {
					if intent, ok := args["intent"].(string); ok {
						predicted = intent
					}
				}
			}
		}

		// Fallback to content
		if predicted == "" && choice.Message.Content != "" {
			predicted = strings.TrimSpace(choice.Message.Content)
		}
	}

	return predicted, toolCalls
}

// =============================================================================
// Handlers
// =============================================================================

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(HealthResponse{Healthy: true})
}

func taskInfoHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	// Check authentication
	if apiKey != "" {
		providedKey := r.Header.Get("X-API-Key")
		if providedKey != apiKey {
			w.WriteHeader(http.StatusUnauthorized)
			json.NewEncoder(w).Encode(ErrorResponse{Detail: "Invalid or missing API key"})
			return
		}
	}

	// Parse seeds from query string - handles both ?seed=0&seed=1 and ?seeds=0&seeds=1
	var requestedSeeds []int
	query := r.URL.RawQuery
	for _, param := range strings.Split(query, "&") {
		parts := strings.SplitN(param, "=", 2)
		if len(parts) == 2 && (parts[0] == "seed" || parts[0] == "seeds") {
			if seed, err := strconv.Atoi(parts[1]); err == nil {
				requestedSeeds = append(requestedSeeds, seed)
			}
		}
	}

	datasetSize := len(dataset.Samples)

	// Build response - one TaskInfo per requested seed, or all seeds if none specified
	var infos []TaskInfo
	if len(requestedSeeds) > 0 {
		for _, seed := range requestedSeeds {
			infos = append(infos, TaskInfo{
				Task: TaskSpec{
					TaskID:      "banking77-go",
					Name:        "Banking77 Intent Classification (Go)",
					Description: "Classify banking customer queries into intent categories",
					Version:     "1.0.0",
				},
				Environment: "banking77",
				Dataset: DatasetInfo{
					Seeds:      []int{seed},
					TrainCount: datasetSize,
					ValCount:   0,
					TestCount:  0,
				},
				Rubric: RubricSpec{
					ScoringCriteria: "exact_match",
					MetricPrimary:   "accuracy",
					MetricRange:     []float64{0.0, 1.0},
				},
				Inference: InferenceSpec{
					Mode:           "tool_call",
					SupportedTools: []string{"classify"},
				},
				Limits: LimitsSpec{
					MaxResponseTokens: 100,
					TimeoutSeconds:    30,
				},
			})
		}
	} else {
		// Return all seeds
		allSeeds := make([]int, datasetSize)
		for i := 0; i < datasetSize; i++ {
			allSeeds[i] = i
		}
		infos = append(infos, TaskInfo{
			Task: TaskSpec{
				TaskID:      "banking77-go",
				Name:        "Banking77 Intent Classification (Go)",
				Description: "Classify banking customer queries into intent categories",
				Version:     "1.0.0",
			},
			Environment: "banking77",
			Dataset: DatasetInfo{
				Seeds:      allSeeds,
				TrainCount: datasetSize,
				ValCount:   0,
				TestCount:  0,
			},
			Rubric: RubricSpec{
				ScoringCriteria: "exact_match",
				MetricPrimary:   "accuracy",
				MetricRange:     []float64{0.0, 1.0},
			},
			Inference: InferenceSpec{
				Mode:           "tool_call",
				SupportedTools: []string{"classify"},
			},
			Limits: LimitsSpec{
				MaxResponseTokens: 100,
				TimeoutSeconds:    30,
			},
		})
	}

	json.NewEncoder(w).Encode(infos)
}

func rolloutHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	// Check authentication
	if apiKey != "" {
		providedKey := r.Header.Get("X-API-Key")
		if providedKey != apiKey {
			w.WriteHeader(http.StatusUnauthorized)
			json.NewEncoder(w).Encode(ErrorResponse{Detail: "Invalid or missing API key"})
			return
		}
	}

	// Parse request
	var req RolloutRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{Detail: "Invalid request body"})
		return
	}

	// Get sample
	seed := 0
	if req.Env.Seed != nil {
		seed = int(*req.Env.Seed)
	}
	sample := getSample(seed)

	log.Printf("Rollout: run_id=%s seed=%d query=%s", req.RunID, seed, sample.Text)

	// Get inference URL
	inferenceURL := ""
	for _, key := range []string{"inference_url", "api_base", "base_url"} {
		if v, ok := req.Policy.Config[key].(string); ok && v != "" {
			inferenceURL = v
			break
		}
	}

	if inferenceURL == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{Detail: "Missing inference_url in policy.config"})
		return
	}

	model := "gpt-4o-mini"
	if m, ok := req.Policy.Config["model"].(string); ok && m != "" {
		model = m
	}

	// Build messages and call LLM
	messages := buildMessages(req.Policy.Config, sample)
	providedKey := r.Header.Get("X-API-Key")

	llmResp, err := callLLM(inferenceURL, model, messages, providedKey)
	if err != nil {
		log.Printf("LLM call failed: %v", err)
		w.WriteHeader(http.StatusBadGateway)
		json.NewEncoder(w).Encode(ErrorResponse{Detail: fmt.Sprintf("LLM call failed: %v", err)})
		return
	}

	// Extract prediction
	predicted, toolCalls := extractPrediction(llmResp)

	// Compute reward
	isCorrect := strings.EqualFold(predicted, sample.Label)
	reward := 0.0
	if isCorrect {
		reward = 1.0
	}

	log.Printf("Result: expected=%s predicted=%s correct=%v reward=%.1f", sample.Label, predicted, isCorrect, reward)

	// Build response
	policyID := req.Policy.PolicyID
	if policyID == "" {
		policyID = req.Policy.PolicyName
	}
	if policyID == "" {
		policyID = "policy"
	}

	resp := RolloutResponse{
		RunID: req.RunID,
		Trajectories: []Trajectory{
			{
				EnvID:    fmt.Sprintf("task::train::%d", seed),
				PolicyID: policyID,
				Steps: []Step{
					{
						Obs: map[string]interface{}{
							"query": sample.Text,
							"index": seed,
						},
						ToolCalls: toolCalls,
						Reward:    reward,
						Done:      true,
						Info: map[string]interface{}{
							"expected":  sample.Label,
							"predicted": predicted,
							"correct":   isCorrect,
						},
					},
				},
				Length:       1,
				InferenceURL: inferenceURL,
			},
		},
		Metrics: Metrics{
			EpisodeReturns: []float64{reward},
			MeanReturn:     reward,
			NumSteps:       1,
			NumEpisodes:    1,
			OutcomeScore:   reward,
		},
		Aborted:     false,
		OpsExecuted: 1,
	}

	json.NewEncoder(w).Encode(resp)
}

// =============================================================================
// Main
// =============================================================================

func main() {
	// Load configuration
	port := os.Getenv("PORT")
	if port == "" {
		port = "8001"
	}

	apiKey = os.Getenv("ENVIRONMENT_API_KEY")
	if apiKey != "" {
		log.Println("API key authentication enabled")
	} else {
		log.Println("WARNING: No ENVIRONMENT_API_KEY set - running without authentication")
	}

	// Load LLM API key for Bearer auth
	llmApiKey = os.Getenv("GROQ_API_KEY")
	if llmApiKey == "" {
		llmApiKey = os.Getenv("OPENAI_API_KEY")
	}
	if llmApiKey != "" {
		log.Println("LLM API key configured")
	} else {
		log.Println("WARNING: No GROQ_API_KEY or OPENAI_API_KEY set - LLM calls may fail")
	}

	// Load dataset
	dataset = loadDataset()
	log.Printf("Dataset loaded: %d samples", len(dataset.Samples))

	// Set up routes
	http.HandleFunc("/health", healthHandler)
	http.HandleFunc("/task_info", taskInfoHandler)
	http.HandleFunc("/rollout", rolloutHandler)

	// Start server
	addr := ":" + port
	log.Printf("Starting task app on %s", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatal(err)
	}
}
