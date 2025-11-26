package openaiclient

import (
	"encoding/json"
	"testing"
)

func TestChatRequestMarshalJSONWithExtraBody(t *testing.T) {
	tests := []struct {
		name     string
		request  ChatRequest
		expected map[string]interface{}
	}{
		{
			name: "basic request with extra body",
			request: ChatRequest{
				Model:       "gpt-3.5-turbo",
				Temperature: 0.7,
				Messages: []*ChatMessage{
					{
						Role:    "user",
						Content: "Hello",
					},
				},
				ExtraBody: map[string]any{
					"custom_field": "custom_value",
					"user_id":      12345,
				},
			},
			expected: map[string]interface{}{
				"model":       "gpt-3.5-turbo",
				"temperature": 0.7,
				"messages": []interface{}{
					map[string]interface{}{
						"role":    "user",
						"content": "Hello",
					},
				},
				"custom_field": "custom_value",
				"user_id":      12345,
			},
		},
		{
			name: "request with numeric precision test",
			request: ChatRequest{
				Model:       "gpt-3.5-turbo",
				Temperature: 0.5,
				MaxTokens:   100,
				Messages: []*ChatMessage{
					{
						Role:    "user",
						Content: "Test",
					},
				},
				ExtraBody: map[string]any{
					"large_int":   int64(9223372036854775807), // max int64
					"float_value": 123.456789012345,
					"normal_int":  42,
				},
			},
			expected: map[string]interface{}{
				"model":       "gpt-3.5-turbo",
				"temperature": 0.5,
				"max_tokens":  100,
				"messages": []interface{}{
					map[string]interface{}{
						"role":    "user",
						"content": "Test",
					},
				},
				"large_int":   int64(9223372036854775807),
				"float_value": 123.456789012345,
				"normal_int":  42,
			},
		},
		{
			name: "reasoning model without temperature",
			request: ChatRequest{
				Model:       "o1-preview",
				Temperature: 0.8, // This should be omitted for reasoning models
				Messages: []*ChatMessage{
					{
						Role:    "user",
						Content: "Reasoning test",
					},
				},
				ExtraBody: map[string]any{
					"reasoning_effort": "high",
				},
			},
			expected: map[string]interface{}{
				"model": "o1-preview",
				"messages": []interface{}{
					map[string]interface{}{
						"role":    "user",
						"content": "Reasoning test",
					},
				},
				"reasoning_effort": "high",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Marshal the request
			jsonData, err := json.Marshal(tt.request)
			if err != nil {
				t.Fatalf("Failed to marshal request: %v", err)
			}

			// Unmarshal into a map to check the result
			var result map[string]interface{}
			if err := json.Unmarshal(jsonData, &result); err != nil {
				t.Fatalf("Failed to unmarshal result: %v", err)
			}

			// Check expected fields
			for key, expectedValue := range tt.expected {
				if value, exists := result[key]; exists {
					// For numeric comparisons, we need to handle type conversions
					switch expectedValue.(type) {
					case int64:
						// JSON numbers are float64 by default, but our UseNumber() should preserve precision
						if floatVal, ok := value.(float64); ok {
							if floatVal != float64(expectedValue.(int64)) {
								t.Errorf("Field %s: expected %v, got %v", key, expectedValue, value)
							}
						} else {
							t.Errorf("Field %s: expected numeric type, got %T", key, value)
						}
					case float64:
						if floatVal, ok := value.(float64); ok {
							if floatVal != expectedValue.(float64) {
								t.Errorf("Field %s: expected %v, got %v", key, expectedValue, value)
							}
						} else {
							t.Errorf("Field %s: expected float64, got %T", key, value)
						}
					case string, bool:
						if value != expectedValue {
							t.Errorf("Field %s: expected %v, got %v", key, expectedValue, value)
						}
					case []interface{}:
						// For arrays, we can't directly compare, but we can check they exist and have expected length
						if arr, ok := value.([]interface{}); ok {
							if len(arr) != len(expectedValue.([]interface{})) {
								t.Errorf("Field %s: expected array length %d, got %d", key, len(expectedValue.([]interface{})), len(arr))
							}
						} else {
							t.Errorf("Field %s: expected array type, got %T", key, value)
						}
					default:
						// Skip complex types for now
						t.Logf("Field %s: skipping comparison for complex type %T", key, expectedValue)
					}
				} else {
					t.Errorf("Expected field %s not found in result", key)
				}
			}

			// Verify that extra body fields are present
			if tt.request.ExtraBody != nil {
				for key := range tt.request.ExtraBody {
					if _, exists := result[key]; !exists {
						t.Errorf("Extra body field %s not found in result", key)
					}
				}
			}
		})
	}
}

func TestChatRequestConvenienceMethods(t *testing.T) {
	request := &ChatRequest{
		Model:       "gpt-3.5-turbo",
		Temperature: 0.7,
		Messages: []*ChatMessage{
			{
				Role:    "user",
				Content: "Test",
			},
		},
	}

	// Test SetExtraBody
	request.SetExtraBody(map[string]any{
		"field1": "value1",
		"field2": 42,
	})

	if request.ExtraBody["field1"] != "value1" {
		t.Errorf("SetExtraBody failed for field1")
	}
	if request.ExtraBody["field2"] != 42 {
		t.Errorf("SetExtraBody failed for field2")
	}

	// Test AddExtraBodyField
	request.AddExtraBodyField("field3", "value3")
	request.AddExtraBodyField("field4", 3.14)

	if request.ExtraBody["field3"] != "value3" {
		t.Errorf("AddExtraBodyField failed for field3")
	}
	if request.ExtraBody["field4"] != 3.14 {
		t.Errorf("AddExtraBodyField failed for field4")
	}

	// Test that all fields are included in JSON
	jsonData, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("Failed to marshal request: %v", err)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(jsonData, &result); err != nil {
		t.Fatalf("Failed to unmarshal result: %v", err)
	}

	expectedFields := []string{"field1", "field2", "field3", "field4"}
	for _, field := range expectedFields {
		if _, exists := result[field]; !exists {
			t.Errorf("Field %s not found in JSON output", field)
		}
	}
}

func TestChatRequestWithoutExtraBody(t *testing.T) {
	request := ChatRequest{
		Model:       "gpt-3.5-turbo",
		Temperature: 0.7,
		Messages: []*ChatMessage{
			{
				Role:    "user",
				Content: "Hello",
			},
		},
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("Failed to marshal request: %v", err)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(jsonData, &result); err != nil {
		t.Fatalf("Failed to unmarshal result: %v", err)
	}

	// Should have standard fields
	if result["model"] != "gpt-3.5-turbo" {
		t.Errorf("Model field incorrect")
	}
	if result["temperature"] != 0.7 {
		t.Errorf("Temperature field incorrect")
	}
}

// TestExtraBodyFieldOverride tests that ExtraBody fields can override main request fields
func TestExtraBodyFieldOverride(t *testing.T) {
	request := ChatRequest{
		Model:       "gpt-3.5-turbo",
		Temperature: 0.7,
		MaxTokens:   100,
		Messages: []*ChatMessage{
			{
				Role:    "user",
				Content: "Test override",
			},
		},
		ExtraBody: map[string]any{
			"temperature": 0.9, // Override temperature
			"max_tokens":  200, // Override max_tokens
			"custom":      "value",
		},
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("Failed to marshal request: %v", err)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(jsonData, &result); err != nil {
		t.Fatalf("Failed to unmarshal result: %v", err)
	}

	// ExtraBody should override main fields
	if result["temperature"] != 0.9 {
		t.Errorf("Temperature should be overridden by ExtraBody: expected 0.9, got %v", result["temperature"])
	}
	// JSON numbers are float64, so compare as float64
	if maxTokens, ok := result["max_tokens"].(float64); !ok || maxTokens != 200 {
		t.Errorf("MaxTokens should be overridden by ExtraBody: expected 200, got %v (type %T)", result["max_tokens"], result["max_tokens"])
	}
	if result["custom"] != "value" {
		t.Errorf("Custom field should be present: expected 'value', got %v", result["custom"])
	}
}

// TestExtraBodyWithComplexTypes tests ExtraBody with complex data types
func TestExtraBodyWithComplexTypes(t *testing.T) {
	request := ChatRequest{
		Model: "gpt-3.5-turbo",
		Messages: []*ChatMessage{
			{
				Role:    "user",
				Content: "Complex types test",
			},
		},
		ExtraBody: map[string]any{
			"nested_object": map[string]any{
				"key1": "value1",
				"key2": 42,
				"key3": true,
			},
			"array_field": []any{"item1", "item2", 123},
			"null_field":  nil,
		},
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("Failed to marshal request: %v", err)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(jsonData, &result); err != nil {
		t.Fatalf("Failed to unmarshal result: %v", err)
	}

	// Check nested object
	if nested, ok := result["nested_object"].(map[string]interface{}); ok {
		if nested["key1"] != "value1" {
			t.Errorf("Nested object key1 incorrect")
		}
		// JSON numbers are float64, so compare as float64
		if key2, ok := nested["key2"].(float64); !ok || key2 != 42 {
			t.Errorf("Nested object key2 incorrect: expected 42, got %v (type %T)", nested["key2"], nested["key2"])
		}
		if nested["key3"] != true {
			t.Errorf("Nested object key3 incorrect")
		}
	} else {
		t.Errorf("Nested object not found or incorrect type")
	}

	// Check array field
	if array, ok := result["array_field"].([]interface{}); ok {
		if len(array) != 3 {
			t.Errorf("Array field length incorrect: expected 3, got %d", len(array))
		}
	} else {
		t.Errorf("Array field not found or incorrect type")
	}

	// Check null field
	if result["null_field"] != nil {
		t.Errorf("Null field should be nil, got %v", result["null_field"])
	}
}

// TestExtraBodyEdgeCases tests edge cases for ExtraBody
func TestExtraBodyEdgeCases(t *testing.T) {
	tests := []struct {
		name    string
		request ChatRequest
		check   func(t *testing.T, result map[string]interface{})
	}{
		{
			name: "empty ExtraBody",
			request: ChatRequest{
				Model:       "gpt-3.5-turbo",
				Temperature: 0.7,
				ExtraBody:   map[string]any{},
				Messages: []*ChatMessage{
					{Role: "user", Content: "Empty extra body"},
				},
			},
			check: func(t *testing.T, result map[string]interface{}) {
				if len(result) != 3 { // model, temperature, messages
					t.Errorf("Expected 3 fields, got %d", len(result))
				}
			},
		},
		{
			name: "nil ExtraBody",
			request: ChatRequest{
				Model:       "gpt-3.5-turbo",
				Temperature: 0.7,
				ExtraBody:   nil,
				Messages: []*ChatMessage{
					{Role: "user", Content: "Nil extra body"},
				},
			},
			check: func(t *testing.T, result map[string]interface{}) {
				if len(result) != 3 { // model, temperature, messages
					t.Errorf("Expected 3 fields, got %d", len(result))
				}
			},
		},
		{
			name: "ExtraBody with special characters",
			request: ChatRequest{
				Model: "gpt-3.5-turbo",
				Messages: []*ChatMessage{
					{Role: "user", Content: "Special chars"},
				},
				ExtraBody: map[string]any{
					"unicode_key":   "🚀 Unicode test",
					"special_chars": "Hello\nWorld\t!",
					"quotes":        `"quoted" and 'single'`,
				},
			},
			check: func(t *testing.T, result map[string]interface{}) {
				if result["unicode_key"] != "🚀 Unicode test" {
					t.Errorf("Unicode key incorrect")
				}
				if result["special_chars"] != "Hello\nWorld\t!" {
					t.Errorf("Special chars incorrect")
				}
				if result["quotes"] != `"quoted" and 'single'` {
					t.Errorf("Quotes incorrect")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			jsonData, err := json.Marshal(tt.request)
			if err != nil {
				t.Fatalf("Failed to marshal request: %v", err)
			}

			var result map[string]interface{}
			if err := json.Unmarshal(jsonData, &result); err != nil {
				t.Fatalf("Failed to unmarshal result: %v", err)
			}

			tt.check(t, result)
		})
	}
}
