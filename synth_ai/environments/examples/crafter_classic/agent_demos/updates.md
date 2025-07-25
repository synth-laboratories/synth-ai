# Agent State
# Environment State
# Agent: Context (inbound message), Tool Call (outbound message)
# Environment: Action (inbound message), Env State (outbound message)
# Runtime Events: Tool calls -> Actions (K messages), Observations -> Context (1 message)

Events + Message Queue
Event is messages going into and out of a Markov blanket?
In general, you actually can't make that assumption
One message queue
One notion of messages 
One notion of event
Global Time, Local Time





# New Tracing
# New Achievement -> Event Reward Signal
# Inventory Increase -> Other Event Metadata
# Health Decrease -> Other Event Metadata
# Trace Tool Params sent to Model
# Trace Tool calls emitted (CAIS), tool calls parsed into actions (RuntimeEvent), Actions into observations (EnvironmentEvents), New Observations into Context (RuntimeEvent)
# 1 session time step -> llm call + resulting environment steps
# Multiple Hypotheticals
    # Sampling
    # Sub-Trajectories

# Easy Replication?
# Check last MCTS demo
# Scope out if above is *sufficient* with some kind of compatibility standards in the agent and environment
# Ideally we can just add decorators and/or wrappers around environments and agents and enable agent restarts



Synth Experiments
- Experiments
    - local duckdb
    - synth experiments api

- Synth Managed Experiments
    - modalized, registered with decorators
    - UpdatePrompt(versions=[A,B,C],priority=high,turnaround=deferred)
    - core idea:
        - simple code upload, containerization, data as data
        - agents as code
- Synth Error Hooks
    Traces (full, sub)
    Agent-Debugger
    
- synth_app = Synth.ADAS(dataset=XYZ, spec=ABC)
