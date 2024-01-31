###############################################################################
# Prompt Architecture for Problem Solving
# Generates a solution proposal
# Author Daniel Albert, PhD
# 2024
###############################################################################
from openai_function.chatGPT import chatGPT
import prompts.prompts as prompt
from tools.tools import spawn_agents
from dependencies.functions import generate_agent_system_prompts
from dependencies.functions import report_generator
from audio.generate_audio_function import generate_audio


user_prompt = """"How can I eat more healthy at work, where all I can buy is take-out?
I cannot prepare food from home, so this is unfortunately not an option."""

# Reframe Task:
reframe = chatGPT(prompt = user_prompt,
                  system_prompt=prompt.project,
                  temperature=0.5)

# Spawn 3 Experts:
agent_call = chatGPT(prompt = reframe["text"],
                  system_prompt=prompt.interpreter,
                  temperature=0.5,
                  tools=spawn_agents)

# Get Experts to work:
agents_system_prompts = generate_agent_system_prompts(agent_call)

agent_proposals = []
for agent_prompt in agents_system_prompts:
    agent_proposals.append(chatGPT(prompt = reframe["text"],
            system_prompt=agent_prompt,
            temperature=0.7,
            model = "gpt-3.5-turbo",
            max_tokens=2048))
    
# Provide First-Round Feedback
devils_advocate = []
for proposal in agent_proposals:
    tmp_history = [{"role": "user", "content": reframe["text"]}]
    devils_advocate.append(chatGPT(prompt = proposal["text"],
            system_prompt=prompt.devil,
            conversation_history=tmp_history,
            temperature=0.5,
            model = "gpt-3.5-turbo",
            max_tokens=2048))

# Revise Proposal
revised_proposals = []
for i, feedback in enumerate(devils_advocate):
    tmp_proposal = agent_proposals[i]
    tmp_history = [{"role": "user", "content": reframe["text"]},
                   {"role": "assistant", "content": tmp_proposal["text"]}]
    tmp_prompt = feedback["text"] + "\nPlease revise your proposal accordingly,\
        be concise and to the point. Stay within 300 words."
    tmp_system = agents_system_prompts[i]
    revised_proposals.append(chatGPT(prompt = tmp_prompt,
            system_prompt=tmp_system,
            conversation_history=tmp_history,
            temperature=0.5,
            model = "gpt-3.5-turbo",
            max_tokens=2500))

# Consolidate Proposals
tmp_history = [{"role": "user", "content": "This is the original message from our client: " + user_prompt},
               {"role": "assistant", "content": "This is how I framed it to the team: " + reframe["text"]}]

for i, function_call in enumerate(agent_call['function_calls']):
    agent_role = function_call['arguments']['title']
    agent_content = revised_proposals[i]['text']
    tmp_history.append({
        "role": "user", 
        "content": f"This is what we got back from our {agent_role}: {agent_content}"
    })

exec_summary = chatGPT(prompt = "\nPlease go to work and devise the Executive Summary to our client.",
                      system_prompt=prompt.integration,
                      conversation_history=tmp_history,
                      model = "gpt-3.5-turbo",
                      max_tokens = 2048,
                      temperature = 0.9)

# Generate Report
experts_list = [{'role': fc['arguments']['title']} for fc in agent_call['function_calls']]
question_prompt = user_prompt
expert_responses = revised_proposals
report_name = "Draft"


report_generator(experts_list, question_prompt, revised_proposals, exec_summary, report_name, project_draft=None)

experts_list = [{'role': fc['arguments']['title']} for fc in agent_call['function_calls']]


generate_audio(exec_summary["text"], speed=1.2)
