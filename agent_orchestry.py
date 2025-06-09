from agents import Agent, Runner
from agents.teaching_agent import TeachingAgent
from agents.video_gen_agent import VideoAgent
agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)