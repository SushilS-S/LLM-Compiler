import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, Iterable, List, Union
from langchain_core.runnables import chain as as_runnable
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from output_parser import Task
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import chain as as_runnable
from typing_extensions import TypedDict

def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            results[int(message.additional_kwargs["idx"])] = message.content
    return results

class SchedulerInput(TypedDict):
    messages: List[BaseMessage]
    tasks: Iterable[Task]

def _execute_task(task, observations, config):
    tool_to_use = task["tool"]
    if isinstance(tool_to_use, str):
        return tool_to_use
    args = task["args"]
    try:
        if isinstance(args, dict):
            resolved_args = {
                key: _resolve_arg(val, observations) for key, val in args.items()
            }
        else:
            resolved_args = _resolve_arg(args, observations)
    except Exception as e:
        return f"ERROR: Failed to resolve args. Error: {repr(e)}"
    
    try:
        return tool_to_use._run(**resolved_args)
    except Exception as e:
        return f"ERROR: Failed to execute tool. Error: {repr(e)}"


def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):
    ID_PATTERN = r"\$\{?(\d+)\}?"

    def replace_match(match):
        idx = int(match.group(1))
        return str(observations.get(idx, match.group(0)))

    if isinstance(arg, str):
        return re.sub(ID_PATTERN, replace_match, arg)
    elif isinstance(arg, list):
        return [_resolve_arg(a, observations) for a in arg]
    else:
        return str(arg)

@as_runnable
def schedule_task(task_inputs, config):
    task: Task = task_inputs["task"]
    observations: Dict[int, Any] = task_inputs["observations"]
    try:
        observation = _execute_task(task, observations, config)
    except Exception:
        import traceback
        observation = traceback.format_exception()
    observations[task["idx"]] = observation

def schedule_pending_task(
    task: Task, observations: Dict[int, Any], retry_after: float = 0.2
):
    while True:
        deps = task["dependencies"]
        if deps and (any([dep not in observations for dep in deps])):
            time.sleep(retry_after)
            continue
        schedule_task.invoke({"task": task, "observations": observations})
        break

@as_runnable
def schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
    tasks = scheduler_input["tasks"]
    args_for_tasks = {}
    messages = scheduler_input["messages"]
    observations = _get_observations(messages)
    task_names = {}
    originals = set(observations)
    futures = []
    retry_after = 0.25
    with ThreadPoolExecutor() as executor:
        for task in tasks:
            deps = task["dependencies"]
            task_names[task["idx"]] = (
                task["tool"] if isinstance(task["tool"], str) else task["tool"].name
            )
            args_for_tasks[task["idx"]] = task["args"]
            if deps and (any([dep not in observations for dep in deps])):
                futures.append(
                    executor.submit(
                        schedule_pending_task, task, observations, retry_after
                    )
                )
            else:
                schedule_task.invoke(dict(task=task, observations=observations))
        wait(futures)
    new_observations = {
        k: (task_names[k], args_for_tasks[k], observations[k])
        for k in sorted(observations.keys() - originals)
    }
    tool_messages = [
        FunctionMessage(
            name=name, content=str(obs), additional_kwargs={"idx": k, "args": task_args}, tool_call_id = k
        )
        for k, (name, task_args, obs) in new_observations.items()
    ]
    return tool_messages

