# TYPE: Stream Writer (Live Progress Updates)
# DESCRIPTION: stream_writer lets a tool send live messages to the user
# WHILE it is still running — before returning its final result.
# Great for long-running tools where the user would otherwise just wait.
# Access via runtime.stream_writer inside the tool.

from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
import time

@tool
def analyze_report(report_id: str, runtime: ToolRuntime) -> str:
    """Analyze a large report — sends progress updates while working."""
    writer = runtime.stream_writer  # get the stream writer

    # Send live updates as work progresses
    writer(f"Loading report {report_id}...")
    time.sleep(1)   # simulating work

    writer(f"Scanning for keywords...")
    time.sleep(1)

    writer(f"Calculating statistics...")
    time.sleep(1)

    writer(f"Analysis complete!")

    # Return the final result to the model
    return f"Report {report_id}: 3 key findings, 95% confidence score."

@tool
def fetch_data(source: str, runtime: ToolRuntime) -> str:
    """Fetch data from an external source with progress updates."""
    writer = runtime.stream_writer

    writer(f"Connecting to {source}...")
    time.sleep(0.5)
    writer(f"Downloading data...")
    time.sleep(0.5)
    writer(f"Processing...")

    return f"Data from {source}: 1,200 records fetched."

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[analyze_report, fetch_data],
)

# Stream the agent — user sees live updates as tools run
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Analyze report ID-42"}]},
    stream_mode="custom"   # needed to receive stream_writer output
):
    print(chunk)   # prints live tool progress messages

# WITHOUT stream_writer: user stares at blank screen for 3 seconds
# WITH stream_writer:    user sees step-by-step progress updates live