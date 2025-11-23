//! Synth Task App Example - Zig Implementation
//!
//! A minimal but complete Task App implementing the Synth contract for prompt optimization.
//!
//! ## Building
//! ```bash
//! zig build -Doptimize=ReleaseFast
//! ```
//!
//! ## Running
//! ```bash
//! ./zig-out/bin/synth-task-app
//! # Or with env vars:
//! ENVIRONMENT_API_KEY=secret PORT=8001 ./zig-out/bin/synth-task-app
//! ```
//!
//! ## Cross-compiling
//! ```bash
//! zig build -Doptimize=ReleaseFast -Dtarget=x86_64-linux-musl
//! zig build -Doptimize=ReleaseFast -Dtarget=aarch64-macos
//! ```

const std = @import("std");
const net = std.net;
const http = std.http;
const mem = std.mem;
const json = std.json;
const Allocator = std.mem.Allocator;

// =============================================================================
// Dataset
// =============================================================================

const Sample = struct {
    text: []const u8,
    label: []const u8,
};

// Embedded samples matching banking77.json (first 12 entries)
const samples = [_]Sample{
    .{ .text = "How do I reset my PIN?", .label = "change_pin" },
    .{ .text = "I need to change my PIN code", .label = "change_pin" },
    .{ .text = "Can I change my card PIN?", .label = "change_pin" },
    .{ .text = "My card hasn't arrived yet", .label = "card_arrival" },
    .{ .text = "When will my card be delivered?", .label = "card_arrival" },
    .{ .text = "I've been waiting for my card for 2 weeks", .label = "card_arrival" },
    .{ .text = "I want to cancel my card", .label = "terminate_account" },
    .{ .text = "How do I close my account?", .label = "terminate_account" },
    .{ .text = "I want to terminate my account", .label = "terminate_account" },
    .{ .text = "How do I activate my new card?", .label = "activate_my_card" },
    .{ .text = "I received my card, how do I activate it?", .label = "activate_my_card" },
    .{ .text = "My card needs to be activated", .label = "activate_my_card" },
};

const labels = [_][]const u8{
    "activate_my_card",
    "card_arrival",
    "change_pin",
    "terminate_account",
    "transaction_charged_twice",
    "pending_transfer",
};

fn getSample(seed: usize) *const Sample {
    return &samples[seed % samples.len];
}

// =============================================================================
// JSON Helpers
// =============================================================================

fn jsonString(allocator: Allocator, obj: anytype) ![]u8 {
    return try json.stringifyAlloc(allocator, obj, .{});
}

// =============================================================================
// HTTP Server
// =============================================================================

const Context = struct {
    api_key: ?[]const u8,
    llm_api_key: ?[]const u8,
    allocator: Allocator,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Read configuration from environment
    const port_str = std.posix.getenv("PORT") orelse "8001";
    const port = try std.fmt.parseInt(u16, port_str, 10);
    const api_key = std.posix.getenv("ENVIRONMENT_API_KEY");

    // LLM API key - try GROQ_API_KEY first, then OPENAI_API_KEY
    const llm_api_key = std.posix.getenv("GROQ_API_KEY") orelse std.posix.getenv("OPENAI_API_KEY");

    if (api_key) |_| {
        std.log.info("API key authentication enabled", .{});
    } else {
        std.log.warn("No ENVIRONMENT_API_KEY set - running without authentication", .{});
    }

    if (llm_api_key) |_| {
        std.log.info("LLM API key configured", .{});
    } else {
        std.log.warn("No GROQ_API_KEY or OPENAI_API_KEY set - LLM calls will fail", .{});
    }

    const ctx = Context{
        .api_key = api_key,
        .llm_api_key = llm_api_key,
        .allocator = allocator,
    };

    // Create server
    const address = net.Address.initIp4(.{ 0, 0, 0, 0 }, port);
    var server = try address.listen(.{ .reuse_address = true });
    defer server.deinit();

    std.log.info("Task app listening on port {d}", .{port});
    std.log.info("Dataset loaded: {d} samples", .{samples.len});

    // Accept connections
    while (true) {
        const conn = server.accept() catch |err| {
            std.log.err("Accept error: {}", .{err});
            continue;
        };

        handleConnection(allocator, conn, ctx) catch |err| {
            std.log.err("Connection error: {}", .{err});
        };
    }
}

fn handleConnection(allocator: Allocator, conn: net.Server.Connection, ctx: Context) !void {
    defer conn.stream.close();

    // Simple HTTP parsing
    var request_buf: [8192]u8 = undefined;
    const bytes_read = conn.stream.read(&request_buf) catch return;
    if (bytes_read == 0) return;

    const request_data = request_buf[0..bytes_read];

    // Parse first line: "METHOD PATH HTTP/1.x"
    const first_line_end = mem.indexOf(u8, request_data, "\r\n") orelse return;
    const first_line = request_data[0..first_line_end];

    var parts = mem.splitScalar(u8, first_line, ' ');
    const method = parts.next() orelse return;
    const target = parts.next() orelse return;

    // Parse headers to find x-api-key
    var api_key: ?[]const u8 = null;
    var headers_section = request_data[first_line_end + 2 ..];
    while (mem.indexOf(u8, headers_section, "\r\n")) |line_end| {
        const header_line = headers_section[0..line_end];
        if (header_line.len == 0) break; // Empty line = end of headers

        if (mem.indexOf(u8, header_line, ":")) |colon_idx| {
            const key = mem.trim(u8, header_line[0..colon_idx], " ");
            const val = mem.trim(u8, header_line[colon_idx + 1 ..], " ");
            if (std.ascii.eqlIgnoreCase(key, "x-api-key")) {
                api_key = val;
            }
        }
        headers_section = headers_section[line_end + 2 ..];
    }

    // Find body (after double CRLF)
    const body_start = mem.indexOf(u8, request_data, "\r\n\r\n");
    const body = if (body_start) |idx| request_data[idx + 4 ..] else "";

    // Route request
    const query_start = mem.indexOf(u8, target, "?") orelse target.len;
    const path = target[0..query_start];
    const query_string = if (query_start < target.len) target[query_start + 1 ..] else "";

    if (mem.eql(u8, path, "/health")) {
        try sendResponse(conn.stream, "200 OK", "{\"healthy\": true}");
    } else if (mem.eql(u8, path, "/task_info")) {
        try handleTaskInfoSimple(allocator, conn.stream, ctx, api_key, query_string);
    } else if (mem.eql(u8, path, "/rollout") and mem.eql(u8, method, "POST")) {
        try handleRolloutSimple(allocator, conn.stream, ctx, api_key, body);
    } else {
        try sendResponse(conn.stream, "404 Not Found", "{\"detail\": \"Not Found\"}");
    }
}

fn sendResponse(stream: net.Stream, status: []const u8, body: []const u8) !void {
    var response_buf: [8192]u8 = undefined;
    const response = std.fmt.bufPrint(&response_buf,
        "HTTP/1.1 {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
        .{ status, body.len, body }
    ) catch return;
    _ = stream.write(response) catch {};
}

fn sendResponseAlloc(allocator: Allocator, stream: net.Stream, status: []const u8, body: []const u8) !void {
    const header = try std.fmt.allocPrint(allocator,
        "HTTP/1.1 {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n",
        .{ status, body.len }
    );
    defer allocator.free(header);
    _ = stream.write(header) catch {};
    _ = stream.write(body) catch {};
}

fn handleTaskInfoSimple(allocator: Allocator, stream: net.Stream, ctx: Context, api_key: ?[]const u8, query_string: []const u8) !void {
    // Check authentication
    if (ctx.api_key) |expected_key| {
        if (api_key == null or !mem.eql(u8, api_key.?, expected_key)) {
            try sendResponse(stream, "401 Unauthorized", "{\"detail\": \"Invalid or missing API key\"}");
            return;
        }
    }

    // Parse seeds from query string
    var requested_seeds = std.ArrayListUnmanaged(usize){};
    defer requested_seeds.deinit(allocator);

    var params_iter = mem.splitScalar(u8, query_string, '&');
    while (params_iter.next()) |param| {
        var kv_iter = mem.splitScalar(u8, param, '=');
        const key = kv_iter.next() orelse continue;
        const val = kv_iter.next() orelse continue;

        if (mem.eql(u8, key, "seed") or mem.eql(u8, key, "seeds")) {
            if (std.fmt.parseInt(usize, val, 10)) |seed| {
                try requested_seeds.append(allocator, seed);
            } else |_| {}
        }
    }

    // Build response
    var response_buf = std.ArrayListUnmanaged(u8){};
    defer response_buf.deinit(allocator);
    const writer = response_buf.writer(allocator);

    try writer.writeAll("[");
    if (requested_seeds.items.len > 0) {
        for (requested_seeds.items, 0..) |seed, i| {
            if (i > 0) try writer.writeAll(",");
            try std.fmt.format(writer,
                \\{{"task":{{"task_id":"banking77-zig","name":"Banking77 (Zig)","description":"Classify banking queries","version":"1.0.0"}},"environment":"banking77","dataset":{{"seeds":[{d}],"train_count":{d},"val_count":0,"test_count":0}},"rubric":{{"scoring_criteria":"exact_match","metric_primary":"accuracy","metric_range":[0.0,1.0]}},"inference":{{"mode":"tool_call","supported_tools":["classify"]}},"limits":{{"max_response_tokens":100,"timeout_seconds":30}}}}
            , .{ seed, samples.len });
        }
    } else {
        try writer.writeAll("{\"task\":{\"task_id\":\"banking77-zig\",\"name\":\"Banking77 (Zig)\",\"description\":\"Classify banking queries\",\"version\":\"1.0.0\"},\"environment\":\"banking77\",\"dataset\":{\"seeds\":[");
        for (0..samples.len) |i| {
            if (i > 0) try writer.writeAll(",");
            try std.fmt.format(writer, "{d}", .{i});
        }
        try std.fmt.format(writer, "],\"train_count\":{d},\"val_count\":0,\"test_count\":0}},\"rubric\":{{\"scoring_criteria\":\"exact_match\",\"metric_primary\":\"accuracy\",\"metric_range\":[0.0,1.0]}},\"inference\":{{\"mode\":\"tool_call\",\"supported_tools\":[\"classify\"]}},\"limits\":{{\"max_response_tokens\":100,\"timeout_seconds\":30}}}}", .{samples.len});
    }
    try writer.writeAll("]");

    try sendResponseAlloc(allocator, stream, "200 OK", response_buf.items);
}

fn handleRolloutSimple(allocator: Allocator, stream: net.Stream, ctx: Context, api_key: ?[]const u8, body: []const u8) !void {
    // Check authentication
    if (ctx.api_key) |expected_key| {
        if (api_key == null or !mem.eql(u8, api_key.?, expected_key)) {
            try sendResponse(stream, "401 Unauthorized", "{\"detail\": \"Invalid or missing API key\"}");
            return;
        }
    }

    // Parse request
    const parsed = json.parseFromSlice(json.Value, allocator, body, .{}) catch {
        try sendResponse(stream, "400 Bad Request", "{\"detail\": \"Invalid JSON\"}");
        return;
    };
    defer parsed.deinit();
    const root = parsed.value;

    const run_id = if (root.object.get("run_id")) |v| v.string else "unknown";

    const seed: usize = blk: {
        if (root.object.get("env")) |env| {
            if (env.object.get("seed")) |s| {
                break :blk @intCast(s.integer);
            }
        }
        break :blk 0;
    };

    const sample = getSample(seed);
    std.log.info("Rollout: run_id={s} seed={d} query={s}", .{ run_id, seed, sample.text });

    // Get inference_url
    const inference_url = blk: {
        if (root.object.get("policy")) |policy| {
            if (policy.object.get("config")) |config| {
                if (config.object.get("inference_url")) |url| break :blk url.string;
                if (config.object.get("api_base")) |url| break :blk url.string;
            }
        }
        break :blk null;
    };

    if (inference_url == null) {
        try sendResponse(stream, "400 Bad Request", "{\"detail\": \"Missing inference_url\"}");
        return;
    }

    // Call LLM
    const prediction_result = callLlmAndPredict(allocator, inference_url.?, sample, ctx) catch {
        try sendResponse(stream, "502 Bad Gateway", "{\"detail\": \"LLM call failed\"}");
        return;
    };
    defer if (prediction_result.predicted) |p| allocator.free(p);

    const predicted = prediction_result.predicted orelse "";
    const correct = mem.eql(u8, predicted, sample.label);
    const reward: f64 = if (correct) 1.0 else 0.0;

    std.log.info("Result: expected={s} predicted={s} correct={} reward={d:.1}", .{ sample.label, predicted, correct, reward });

    // Build response
    const response_json = try std.fmt.allocPrint(allocator,
        \\{{"run_id":"{s}","trajectories":[{{"env_id":"task::train::{d}","policy_id":"policy","steps":[{{"obs":{{"query":"{s}","index":{d}}},"tool_calls":[],"reward":{d:.1},"done":true,"info":{{"expected":"{s}","predicted":"{s}","correct":{s}}}}}],"length":1,"inference_url":"{s}"}}],"metrics":{{"episode_returns":[{d:.1}],"mean_return":{d:.1},"num_steps":1,"num_episodes":1,"outcome_score":{d:.1}}},"aborted":false,"ops_executed":1}}
    , .{ run_id, seed, sample.text, seed, reward, sample.label, predicted, if (correct) "true" else "false", inference_url.?, reward, reward, reward });
    defer allocator.free(response_json);

    try sendResponseAlloc(allocator, stream, "200 OK", response_json);
}


const PredictionResult = struct {
    predicted: ?[]u8,
};

fn callLlmAndPredict(allocator: Allocator, inference_url: []const u8, sample: *const Sample, ctx: Context) !PredictionResult {
    // Build LLM request
    const labels_str = blk: {
        var buf = std.ArrayListUnmanaged(u8){};
        for (labels, 0..) |label, i| {
            if (i > 0) try buf.appendSlice(allocator, ", ");
            try buf.appendSlice(allocator, label);
        }
        break :blk try buf.toOwnedSlice(allocator);
    };
    defer allocator.free(labels_str);

    const request_body = try std.fmt.allocPrint(allocator,
        \\{{
        \\  "model": "gpt-4o-mini",
        \\  "messages": [
        \\    {{"role": "system", "content": "You are an expert banking assistant. Classify queries using the classify tool."}},
        \\    {{"role": "user", "content": "Query: {s}\\nIntents: {s}\\nClassify this query."}}
        \\  ],
        \\  "tools": [{{
        \\    "type": "function",
        \\    "function": {{
        \\      "name": "classify",
        \\      "description": "Classify the query",
        \\      "parameters": {{
        \\        "type": "object",
        \\        "properties": {{"intent": {{"type": "string"}}}},
        \\        "required": ["intent"]
        \\      }}
        \\    }}
        \\  }}],
        \\  "tool_choice": "required",
        \\  "temperature": 0
        \\}}
    , .{ sample.text, labels_str });
    defer allocator.free(request_body);

    // Build URL - handle query params correctly
    // inference_url may be "http://host/path?query" - we need "http://host/path/chat/completions?query"
    const url_str = blk: {
        if (mem.indexOf(u8, inference_url, "?")) |query_idx| {
            const base = mem.trimRight(u8, inference_url[0..query_idx], "/");
            const query = inference_url[query_idx..];
            break :blk try std.fmt.allocPrint(allocator, "{s}/chat/completions{s}", .{ base, query });
        } else {
            const base = mem.trimRight(u8, inference_url, "/");
            break :blk try std.fmt.allocPrint(allocator, "{s}/chat/completions", .{base});
        }
    };
    defer allocator.free(url_str);

    std.log.debug("LLM call: {s}", .{url_str});

    // Make HTTP request using fetch API
    var client = http.Client{ .allocator = allocator };
    defer client.deinit();

    // Prepare extra headers - use Bearer auth for OpenAI-compatible APIs
    var extra_headers_list = std.ArrayListUnmanaged(http.Header){};
    defer extra_headers_list.deinit(allocator);
    try extra_headers_list.append(allocator, .{ .name = "Content-Type", .value = "application/json" });

    // Build auth header if we have an LLM API key
    var auth_value: ?[]u8 = null;
    defer if (auth_value) |av| allocator.free(av);
    if (ctx.llm_api_key) |key| {
        auth_value = try std.fmt.allocPrint(allocator, "Bearer {s}", .{key});
        try extra_headers_list.append(allocator, .{ .name = "Authorization", .value = auth_value.? });
    }

    // Use low-level request API
    const uri = std.Uri.parse(url_str) catch {
        std.log.warn("Failed to parse URL: {s}", .{url_str});
        return PredictionResult{ .predicted = null };
    };

    var req = client.request(.POST, uri, .{
        .extra_headers = extra_headers_list.items,
    }) catch |err| {
        std.log.warn("Failed to create request: {}", .{err});
        return PredictionResult{ .predicted = null };
    };
    defer req.deinit();

    // Send request body
    req.transfer_encoding = .{ .content_length = request_body.len };
    var body_writer = req.sendBody(&.{}) catch |err| {
        std.log.warn("Failed to send body: {}", .{err});
        return PredictionResult{ .predicted = null };
    };
    body_writer.writer.writeAll(request_body) catch |err| {
        std.log.warn("Failed to write body: {}", .{err});
        return PredictionResult{ .predicted = null };
    };
    body_writer.end() catch |err| {
        std.log.warn("Failed to end body: {}", .{err});
        return PredictionResult{ .predicted = null };
    };
    req.connection.?.flush() catch |err| {
        std.log.warn("Failed to flush: {}", .{err});
        return PredictionResult{ .predicted = null };
    };

    // Receive response
    var redirect_buffer: [8192]u8 = undefined;
    var response = req.receiveHead(&redirect_buffer) catch |err| {
        std.log.warn("Failed to receive head: {}", .{err});
        return PredictionResult{ .predicted = null };
    };

    if (response.head.status != .ok) {
        std.log.warn("LLM request failed with status: {}", .{response.head.status});
        return PredictionResult{ .predicted = null };
    }

    // Read response body with decompression support
    var transfer_buffer: [4096]u8 = undefined;
    var decompress_buffer: [std.compress.flate.max_window_len]u8 = undefined;
    var decompress: http.Decompress = undefined;
    var body_reader = response.readerDecompressing(&transfer_buffer, &decompress, &decompress_buffer);
    const response_body = body_reader.allocRemaining(allocator, std.Io.Limit.limited(1024 * 1024)) catch |err| {
        std.log.warn("Failed to read response body: {}", .{err});
        return PredictionResult{ .predicted = null };
    };
    defer allocator.free(response_body);

    std.log.debug("LLM response ({d} bytes)", .{response_body.len});

    // Parse and extract intent
    const llm_parsed = json.parseFromSlice(json.Value, allocator, response_body, .{}) catch |err| {
        std.log.warn("Failed to parse LLM response JSON: {}", .{err});
        return PredictionResult{ .predicted = null };
    };
    defer llm_parsed.deinit();

    // Extract from tool_calls
    if (llm_parsed.value.object.get("choices")) |choices| {
        if (choices.array.items.len > 0) {
            if (choices.array.items[0].object.get("message")) |msg| {
                if (msg.object.get("tool_calls")) |tool_calls| {
                    if (tool_calls.array.items.len > 0) {
                        if (tool_calls.array.items[0].object.get("function")) |func| {
                            if (func.object.get("arguments")) |args_str| {
                                const args_parsed = json.parseFromSlice(json.Value, allocator, args_str.string, .{}) catch |err| {
                                    std.log.warn("Failed to parse tool call arguments: {}", .{err});
                                    return PredictionResult{ .predicted = null };
                                };
                                defer args_parsed.deinit();

                                if (args_parsed.value.object.get("intent")) |intent| {
                                    return PredictionResult{
                                        .predicted = try allocator.dupe(u8, intent.string),
                                    };
                                } else {
                                    std.log.warn("No 'intent' field in tool call arguments", .{});
                                }
                            } else {
                                std.log.warn("No 'arguments' in function", .{});
                            }
                        } else {
                            std.log.warn("No 'function' in tool_call", .{});
                        }
                    } else {
                        std.log.warn("tool_calls array is empty", .{});
                    }
                } else {
                    std.log.warn("No 'tool_calls' in message", .{});
                }
            } else {
                std.log.warn("No 'message' in choice", .{});
            }
        } else {
            std.log.warn("choices array is empty", .{});
        }
    } else {
        std.log.warn("No 'choices' in LLM response", .{});
    }

    return PredictionResult{ .predicted = null };
}

test "basic sample retrieval" {
    const s = getSample(0);
    try std.testing.expectEqualStrings("How do I reset my PIN?", s.text);
    try std.testing.expectEqualStrings("change_pin", s.label);
}

test "sample wrapping" {
    const s = getSample(samples.len + 1);
    // samples.len is 12, so index 13 % 12 = 1
    try std.testing.expectEqualStrings("I need to change my PIN code", s.text);
}
