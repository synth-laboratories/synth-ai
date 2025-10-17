CREATE TABLE experiments (
	experiment_id VARCHAR NOT NULL, 
	name VARCHAR NOT NULL, 
	description TEXT, 
	created_at DATETIME, 
	updated_at DATETIME, 
	configuration TEXT, 
	metadata TEXT, 
	PRIMARY KEY (experiment_id)
);
CREATE INDEX idx_experiment_name ON experiments (name);
CREATE INDEX idx_experiment_created ON experiments (created_at);
CREATE VIEW model_usage_stats AS
        SELECT 
            model_name,
            provider,
            COUNT(*) as usage_count,
            SUM(input_tokens) as total_input_tokens,
            SUM(output_tokens) as total_output_tokens,
            SUM(total_tokens) as total_tokens,
            SUM(cost_usd) / 100.0 as total_cost_usd,
            AVG(latency_ms) as avg_latency_ms,
            MIN(created_at) as first_used,
            MAX(created_at) as last_used
        FROM events
        WHERE event_type = 'cais' AND model_name IS NOT NULL
        GROUP BY model_name, provider
/* model_usage_stats(model_name,provider,usage_count,total_input_tokens,total_output_tokens,total_tokens,total_cost_usd,avg_latency_ms,first_used,last_used) */;
CREATE VIEW session_summary AS
        SELECT 
            s.session_id,
            s.created_at,
            s.num_timesteps,
            s.num_events,
            s.num_messages,
            e.experiment_id,
            e.name as experiment_name,
            COUNT(DISTINCT ev.model_name) as unique_models_used,
            SUM(CASE WHEN ev.event_type = 'cais' THEN ev.cost_usd ELSE 0 END) / 100.0 as total_cost_usd
        FROM session_traces s
        LEFT JOIN experiments e ON s.experiment_id = e.experiment_id
        LEFT JOIN events ev ON s.session_id = ev.session_id
        GROUP BY s.session_id
/* session_summary(session_id,created_at,num_timesteps,num_events,num_messages,experiment_id,experiment_name,unique_models_used,total_cost_usd) */;
CREATE TABLE systems (
	system_id VARCHAR NOT NULL, 
	name VARCHAR NOT NULL, 
	system_type VARCHAR, 
	description TEXT, 
	created_at DATETIME, 
	metadata TEXT, 
	PRIMARY KEY (system_id)
);
CREATE INDEX idx_system_name ON systems (name);
CREATE INDEX idx_system_type ON systems (system_type);
CREATE TABLE session_traces (
	session_id VARCHAR NOT NULL, 
	created_at DATETIME NOT NULL, 
	num_timesteps INTEGER NOT NULL, 
	num_events INTEGER NOT NULL, 
	num_messages INTEGER NOT NULL, 
	metadata TEXT, 
	experiment_id VARCHAR, 
	embedding VECTOR, 
	PRIMARY KEY (session_id), 
	FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id)
);
CREATE INDEX idx_session_created ON session_traces (created_at);
CREATE INDEX idx_session_experiment ON session_traces (experiment_id);
CREATE TABLE system_versions (
	version_id VARCHAR NOT NULL, 
	system_id VARCHAR NOT NULL, 
	version_number VARCHAR NOT NULL, 
	commit_hash VARCHAR, 
	created_at DATETIME, 
	configuration TEXT, 
	metadata TEXT, 
	PRIMARY KEY (version_id), 
	CONSTRAINT uq_system_version UNIQUE (system_id, version_number), 
	FOREIGN KEY(system_id) REFERENCES systems (system_id)
);
CREATE INDEX idx_version_created ON system_versions (created_at);
CREATE INDEX idx_version_system ON system_versions (system_id);
CREATE TABLE session_timesteps (
	id INTEGER NOT NULL, 
	session_id VARCHAR NOT NULL, 
	step_id VARCHAR NOT NULL, 
	step_index INTEGER NOT NULL, 
	turn_number INTEGER, 
	started_at DATETIME, 
	completed_at DATETIME, 
	num_events INTEGER, 
	num_messages INTEGER, 
	step_metadata TEXT, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_session_step UNIQUE (session_id, step_id), 
	FOREIGN KEY(session_id) REFERENCES session_traces (session_id)
);
CREATE INDEX idx_timestep_turn ON session_timesteps (turn_number);
CREATE INDEX idx_timestep_session_step ON session_timesteps (session_id, step_id);
CREATE TABLE experimental_systems (
	id INTEGER NOT NULL, 
	experiment_id VARCHAR NOT NULL, 
	system_id VARCHAR NOT NULL, 
	version_id VARCHAR NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_experiment_system UNIQUE (experiment_id, system_id), 
	FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id), 
	FOREIGN KEY(system_id) REFERENCES systems (system_id), 
	FOREIGN KEY(version_id) REFERENCES system_versions (version_id)
);
CREATE INDEX idx_experimental_system ON experimental_systems (experiment_id, system_id);
CREATE TABLE outcome_rewards (
	id INTEGER NOT NULL, 
	session_id VARCHAR NOT NULL, 
	total_reward INTEGER NOT NULL, 
	achievements_count INTEGER NOT NULL, 
	total_steps INTEGER NOT NULL, 
	created_at DATETIME NOT NULL, 
	reward_metadata TEXT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(session_id) REFERENCES session_traces (session_id)
);
CREATE INDEX idx_outcome_rewards_session ON outcome_rewards (session_id);
CREATE INDEX idx_outcome_rewards_total ON outcome_rewards (total_reward);
CREATE TABLE events (
	id INTEGER NOT NULL, 
	session_id VARCHAR NOT NULL, 
	timestep_id INTEGER, 
	event_type VARCHAR NOT NULL, 
	system_instance_id VARCHAR, 
	event_time FLOAT, 
	message_time INTEGER, 
	created_at DATETIME, 
	model_name VARCHAR, 
	provider VARCHAR, 
	input_tokens INTEGER, 
	output_tokens INTEGER, 
	total_tokens INTEGER, 
	cost_usd INTEGER, 
	latency_ms INTEGER, 
	span_id VARCHAR, 
	trace_id VARCHAR, 
	call_records TEXT, 
	reward FLOAT, 
	terminated BOOLEAN, 
	truncated BOOLEAN, 
	system_state_before TEXT, 
	system_state_after TEXT, 
	metadata TEXT, 
	event_metadata TEXT, 
	embedding VECTOR, 
	PRIMARY KEY (id), 
	CONSTRAINT check_event_type CHECK (event_type IN ('cais', 'environment', 'runtime')), 
	FOREIGN KEY(session_id) REFERENCES session_traces (session_id), 
	FOREIGN KEY(timestep_id) REFERENCES session_timesteps (id)
);
CREATE INDEX idx_event_type ON events (event_type);
CREATE INDEX idx_event_created ON events (created_at);
CREATE INDEX idx_event_model ON events (model_name);
CREATE INDEX idx_event_session_step ON events (session_id, timestep_id);
CREATE INDEX idx_event_trace ON events (trace_id);
CREATE TABLE messages (
	id INTEGER NOT NULL, 
	session_id VARCHAR NOT NULL, 
	timestep_id INTEGER, 
	message_type VARCHAR NOT NULL, 
	content TEXT NOT NULL, 
	timestamp DATETIME, 
	event_time FLOAT, 
	message_time INTEGER, 
	metadata TEXT, 
	embedding VECTOR, 
	PRIMARY KEY (id), 
	CONSTRAINT check_message_type CHECK (message_type IN ('user', 'assistant', 'system', 'tool_use', 'tool_result')), 
	FOREIGN KEY(session_id) REFERENCES session_traces (session_id), 
	FOREIGN KEY(timestep_id) REFERENCES session_timesteps (id)
);
CREATE INDEX idx_message_session_step ON messages (session_id, timestep_id);
CREATE INDEX idx_message_timestamp ON messages (timestamp);
CREATE INDEX idx_message_type ON messages (message_type);
CREATE TABLE event_rewards (
	id INTEGER NOT NULL, 
	event_id INTEGER NOT NULL, 
	session_id VARCHAR NOT NULL, 
	message_id INTEGER, 
	turn_number INTEGER, 
	reward_value FLOAT NOT NULL, 
	reward_type VARCHAR, 
	"key" VARCHAR, 
	annotation TEXT, 
	source VARCHAR, 
	created_at DATETIME NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(event_id) REFERENCES events (id), 
	FOREIGN KEY(session_id) REFERENCES session_traces (session_id), 
	FOREIGN KEY(message_id) REFERENCES messages (id)
);
CREATE INDEX idx_event_rewards_event ON event_rewards (event_id);
CREATE INDEX idx_event_rewards_type ON event_rewards (reward_type);
CREATE INDEX idx_event_rewards_session ON event_rewards (session_id);
CREATE INDEX idx_event_rewards_key ON event_rewards ("key");
CREATE VIEW experiment_overview AS
        SELECT 
            e.experiment_id,
            e.name,
            e.description,
            e.created_at,
            COUNT(DISTINCT s.session_id) as session_count,
            SUM(s.num_events) as total_events,
            SUM(s.num_messages) as total_messages,
            AVG(s.num_timesteps) as avg_timesteps_per_session
        FROM experiments e
        LEFT JOIN session_traces s ON e.experiment_id = s.experiment_id
        GROUP BY e.experiment_id
/* experiment_overview(experiment_id,name,description,created_at,session_count,total_events,total_messages,avg_timesteps_per_session) */;
