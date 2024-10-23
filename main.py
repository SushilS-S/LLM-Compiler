import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from datetime import datetime
import pandas as pd
import traceback
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from core.planner import create_planner
from tools import GoogleAnalyticsTool, Respond
from core.planner import create_planner
from typing import Annotated, Any, Dict, List, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import AIMessage
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

load_dotenv()

ga_tool = GoogleAnalyticsTool()
respond = Respond()


dimensions = pd.read_csv('documentation/dimensions.csv')
metrics = pd.read_csv('documentation/metrics.csv')

with open("documentation/Samples.txt", 'r') as file:
    samples = file.read()

available_dimensions = ', '.join(dimensions.iloc[:, 0].to_list())
available_metrics = ', '.join(metrics.iloc[:, 0].to_list())

date = datetime.now().date()


prompt = ChatPromptTemplate.from_template("""
================================ System Message ================================

Given a user query about Google Analytics, create a plan to solve it with the utmost parallelizability. Each plan should comprise an action from the following {num_tools} types:
{tool_descriptions}
{num_tools}. respond: Formats and responds to the given analytics data based on the original query.

Guidelines:
- When the user provides a natural language query related to Google Analytics data, understand the required data dimensions, metrics, date ranges and sorting.

- Each action described above contains input/output types and description.
    - You must strictly adhere to the input and output types for each action.
    - The action descriptions contain the guidelines. You MUST strictly follow those guidelines when you use the actions.
- Each action in the plan should strictly be one of the above types. Follow the Python conventions for each action.
- Each action MUST have a unique ID, which is strictly increasing.
- Never introduce new actions other than the ones provided.

- Inputs for actions can either be constants or outputs from preceding actions. In the latter case, use the format $id to denote the ID of the previous action whose output will be the input.
- Always call respond as the last action in the plan.
- Ensure the plan maximizes parallelizability.
- Only use the provided action types. If a query cannot be addressed using these, invoke the respond action for the next steps.

**Components of the task**:
- **ga-tool**: Define the google analytics  tool. This tool will be used to fetch data from Google Analytics.
    - **dimensions**: The dimensions to retrieve (e.g., pagePath).
    - **metrics**: The metrics to retrieve (e.g., screenPageViews).
    - **date_ranges**: This is a compulsory field. The start and end dates for the data. (Let the default be last 1 month)
    - **order_bys**: Sorting options for the metrics.

- Use only the metrics and dimensions available in the GA4 API. Below are the available metrics:
{available_metrics} and available dimensions: {available_dimensions}
- When the user query contains keywords like users/visitors, preferably use the metric "totalUsers" over "sessions".
- When calculating the sale of products in terms of count, use products purchased and not the revenue.
- The current date is {date}. Use this for formulating appropriate date ranges when not specified by the user.

- Use the following sample queires and the generated task as a template.{samples}

- For ga-tool queries:
- Always include a date_range. If not specified by the user, default to 'lastMonth'.
- Ensure that metrics and dimensions used are valid for GA4.
- Include the arguments alone, don't include the available  metrics and dimensions in the task that is being generated.
- Add order_bys and dimesional_filter are optional, add them in the query that only requires them.

- For respond queries:
- Include the GA result, and format the response appropriately.

============================= Messages Placeholder =============================

{messages}

Remember, ONLY respond with the task list in the correct format! E.g.:
idx. tool(arg_name=args)
""")

prompt = prompt.partial(
    property_id=ga_tool.property_id,
    date=datetime.now().date(),
    available_metrics=available_metrics,
    available_dimensions=available_dimensions,
    samples=samples
)

llm = ChatOpenAI(model="gpt-4-turbo-preview")
tools = [ga_tool, respond]
planner = create_planner(llm, tools, prompt)


def plan_and_schedule(state: Dict[str, Any]) -> Dict[str, List[Any]]:
    messages = state["messages"]
    try:
        tasks = list(planner.invoke(messages))
        original_query = messages[0].content if messages else ""
        ga_results = {}
        final_response = None

        # Execute tasks
        for task in tasks:
            tool_name = task['tool'].name if isinstance(
                task['tool'], BaseTool) else task['tool']

            if tool_name == 'ga-tool':
                try:
                    metrics = task['args'].get('metrics', [])
                    dimensions = task['args'].get('dimensions', [])
                    if isinstance(metrics, str):
                        metrics = [m.strip(' "[]') for m in metrics.split(',')]
                    if isinstance(dimensions, str):
                        dimensions = [d.strip(' "[]') for d in dimensions.split(',')]
                    metrics = [m for m in metrics if m]
                    dimensions = [d for d in dimensions if d]

                    ga_args = {
                        'date_ranges': task['args'].get('date_ranges', 'lastMonth'),
                        'metrics': metrics,
                        'dimensions': dimensions,
                        'property_id': ga_tool.property_id,
                    }

                    if 'order_bys' in task['args']:
                        order_bys = task['args']['order_bys']
                        if isinstance(order_bys, str):
                            try:
                                order_bys = json.loads(order_bys)
                            except json.JSONDecodeError:
                                order_bys = []
                        ga_args['order_bys'] = order_bys

                    if 'dimensional_filter' in task['args']:
                        dimensional_filter = task['args']['dimensional_filter']
                        if isinstance(dimensional_filter, str):
                            try:
                                dimensional_filter = json.loads(dimensional_filter)
                            except json.JSONDecodeError:
                                dimensional_filter = None
                        ga_args['dimensional_filter'] = dimensional_filter

                    result = ga_tool._run(**ga_args)
                    ga_results[str(task['idx'])] = result

                except Exception as e:
                    error_message = f"GA Tool error: {str(e)}"
                    ga_results[str(task['idx'])] = {"error": error_message}

            elif tool_name == 'respond':
                try:
                    relevant_ga_results = {
                        str(dep): ga_results[str(dep)]
                        for dep in task['dependencies']
                        if str(dep) in ga_results
                    }

                    reply = task['args'].get('reply', '')
                    if isinstance(reply, str):
                        for idx, result in relevant_ga_results.items():
                            reply = reply.replace(f"${idx}", json.dumps(result))

                    respond_args = {
                        'reply': reply,
                        'query': original_query,
                        'ga_result': relevant_ga_results
                    }

                    final_response = respond._run(**respond_args)

                except Exception as e:
                    error_message = f"Respond tool error: {str(e)}"
                    final_response = error_message

        # Generate single, clean output
        output = [
            "=== ANALYTICS QUERY RESULTS ===",
            f"Original Query: {original_query}\n",
            "Execution Plan:",
        ]
        
        for task in tasks:
            tool_name = task['tool'].name if isinstance(
                task['tool'], BaseTool) else task['tool']
            task_info = [f"Task {task['idx']}: {tool_name}"]
            if task['dependencies']:
                task_info.append(f"Dependencies: {task['dependencies']}")
            output.extend(task_info)
        
        output.append("\nResults:")
        for idx, result in ga_results.items():
            output.extend([
                f"\nTask {idx}:",
                json.dumps(result, indent=2)
            ])
        
        if final_response:
            output.extend([
                "\nFinal Response:",
                final_response
            ])

        return {"messages": messages + [AIMessage(content="\n".join(output))]}

    except Exception as e:
        error_message = f"Plan and schedule error: {str(e)}\n" + traceback.format_exc()
        return {"messages": messages + [AIMessage(content=error_message)]}

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

graph_builder.add_node("plan_and_schedule", plan_and_schedule)

def should_continue(state):
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage):
        return END
    return "plan_and_schedule"

graph_builder.add_edge(START, "plan_and_schedule")
graph_builder.add_conditional_edges(
    "plan_and_schedule",
    should_continue,
    {
        "plan_and_schedule": "plan_and_schedule",
        END: END
    }
)

chain = graph_builder.compile()

query = "What is the bounce rate for my homepage?"
input_message = HumanMessage(content=query)
result = chain.invoke({"messages": [input_message]})
final_response = result['messages'][-1].content
print("Final response:")
print(final_response)