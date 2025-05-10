from ragen.env.deep_research.env import DeepResearchEnv
env = DeepResearchEnv(); obs = env.reset()
action = '<tool_use>\n  <name>search</name>\n  <arguments>{\"query\": [\"Russia Ukraine war news latest\"]}</arguments>\n</tool_use>'
print(env.step(action))
print(obs)
