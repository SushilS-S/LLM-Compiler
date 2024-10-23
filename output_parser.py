import ast
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from typing_extensions import TypedDict

THOUGHT_PATTERN = r"Thought: ([^\n]*)"
ACTION_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"
ID_PATTERN = r"\$\{?(\d+)\}?"
END_OF_PLAN = "<END_OF_PLAN>"

def _ast_parse(arg: str) -> Any:
    try:
        return ast.literal_eval(arg)
    except:  # noqa
        return arg

def _parse_llm_compiler_action_args(args: str, tool: Union[str, BaseTool]) -> Dict[str, Any]:
    if args == "":
        return {}
    if isinstance(tool, str):
        return {}
    extracted_args = {}
    tool_key = None
    prev_idx = None
    for key in tool.args.keys():
        if f"{key}=" in args:
            idx = args.index(f"{key}=")
            if prev_idx is not None:
                extracted_args[tool_key] = _ast_parse(
                    args[prev_idx:idx].strip().rstrip(",")
                )
            args = args.split(f"{key}=", 1)[1]
            tool_key = key
            prev_idx = 0
    if prev_idx is not None:
        value = args[prev_idx:].strip().rstrip(",").rstrip(")")
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]  
        extracted_args[tool_key] = value
    return extracted_args

def default_dependency_rule(idx, args: str):
    matches = re.findall(ID_PATTERN, args)
    numbers = [int(match) for match in matches]
    return idx in numbers

def _get_dependencies_from_graph(
    idx: int, tool_name: str, args: Dict[str, Any]
) -> List[int]:
    if tool_name == "join":
        return list(range(1, idx))
    return [i for i in range(1, idx) if default_dependency_rule(i, str(args))]

class Task(TypedDict):
    idx: int
    tool: Union[BaseTool, str]
    args: Dict[str, Any]
    dependencies: List[int]
    thought: Optional[str]

def instantiate_task(
    tools: Sequence[BaseTool],
    idx: int,
    tool_name: str,
    args: Union[str, Any],
    thought: Optional[str] = None,
) -> Task:
    if tool_name == "join":
        tool = "join"
    else:
        try:
            tool = next(t for t in tools if t.name == tool_name)
        except StopIteration:
            raise OutputParserException(f"Tool {tool_name} not found.")
    
    tool_args = _parse_llm_compiler_action_args(args, tool)
    
    if isinstance(tool, BaseTool):
        optional_args = {'order_bys', 'dimensional_filter', 'property_id'}
        required_args = set(tool.args.keys()) - optional_args
        provided_args = set(tool_args.keys())
        missing_args = required_args - provided_args
        if missing_args:
            raise OutputParserException(f"Missing required arguments for {tool_name}: {', '.join(missing_args)}")
    
    dependencies = _get_dependencies_from_graph(idx, tool_name, tool_args)

    return Task(
        idx=idx,
        tool=tool,
        args=tool_args,
        dependencies=dependencies,
        thought=thought,
    )   
class LLMCompilerPlanParser(BaseTransformOutputParser[Dict[str, Any]], extra="allow"):
    tools: List[BaseTool]
    _idx: int = 1

    def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[Task]:
        texts = []
        thought = None
        for chunk in input:
            text = chunk if isinstance(chunk, str) else str(chunk.content)
            for task, thought in self.ingest_token(text, texts, thought):
                if task and task['tool'] != 'join':
                    yield task
        if texts:
            task, _ = self._parse_task("".join(texts), thought)
            if task and task['tool'] != 'join':
                yield task

    def parse(self, text: str) -> List[Task]:
        self._idx = 1
        return list(self._transform([text]))

    def stream(
        self,
        input: Union[str, BaseMessage],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Task]:
        yield from self.transform([input], config, **kwargs)

    def ingest_token(
        self, token: str, buffer: List[str], thought: Optional[str]
    ) -> Iterator[Tuple[Optional[Task], Optional[str]]]:
        buffer.append(token)
        if "\n" in token:
            buffer_ = "".join(buffer).split("\n")
            suffix = buffer_[-1]
            for line in buffer_[:-1]:
                task, thought = self._parse_task(line, thought)
                if task:
                    yield task, thought
            buffer.clear()
            buffer.append(suffix)

    def _parse_task(self, line: str, thought: Optional[str] = None) -> Tuple[Optional[Task], Optional[str]]:
        task = None
        if match := re.match(THOUGHT_PATTERN, line):
            thought = match.group(1)
        else:
            if "." in line:
                line = line.split(".", 1)[1].strip()
            parts = line.split("(", 1)
            if len(parts) == 2:
                tool_name, args = parts
                tool_name = tool_name.strip()
                args = args.rstrip(")").strip()
                if tool_name != 'join':
                    try:
                        tool = next(t for t in self.tools if t.name == tool_name)
                    except StopIteration:
                        raise OutputParserException(f"Tool {tool_name} not found.")                   
                    tool_args = _parse_llm_compiler_action_args(args, tool)
                    
                    if tool_name == 'ga-tool':
                        if 'metrics' not in tool_args:
                            raise OutputParserException("GA tool requires 'metrics' argument")
                        if 'date_ranges' not in tool_args:
                            tool_args['date_ranges'] = 'lastMonth'           
                            
                    elif tool_name == 'respond':
                        if 'reply' not in tool_args:
                            tool_args['reply'] = '' 
                        if 'query' not in tool_args:
                            tool_args['query'] = ''  
                        if 'ga_result' not in tool_args:
                            tool_args['ga_result'] = '{}'  
                                               
                    dependencies = _get_dependencies_from_graph(self._idx, tool_name, tool_args)                   
                    task = Task(
                        idx=self._idx,
                        tool=tool,
                        args=tool_args,
                        dependencies=dependencies,
                        thought=thought,
                    )
                    self._idx += 1
                thought = None
        return task, thought