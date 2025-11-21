import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, NamedTuple, Optional

from openai import OpenAI

MODEL_NAME = "gpt-5"

# Tools that will be passed to every model invocation. They are defined once so
# that the configuration lives in a single place.
TOOLS = [
    {
        "type": "custom",
        "name": "code_exec_python",
        "description": "Executes python code",
    },
    {
        "type": "custom",
        "name": "code_exec_cpp",
        "description": "Executes c++ code",
    },
    {
        "type": "custom",
        "name": "code_exec_java",
        "description": "Executes java code",
    },
]

client = OpenAI()


class ToolResult(NamedTuple):
    success: bool
    output: str


def _prefer_homebrew_java(tool: str) -> Optional[str]:
    homebrew_path = Path("/opt/homebrew/opt/openjdk/bin") / tool
    if homebrew_path.exists():
        return str(homebrew_path)
    return None


def _run_subprocess(command, *, cwd=None, timeout=30) -> subprocess.CompletedProcess:
    """Run a subprocess and capture stdio."""
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _execute_python(code: str) -> ToolResult:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(code)
        script_path = Path(handle.name)
    try:
        proc = _run_subprocess([sys.executable, str(script_path)])
    finally:
        script_path.unlink(missing_ok=True)

    if proc.returncode == 0:
        output = proc.stdout.strip() or proc.stderr.strip() or "Python script produced no output."
        return ToolResult(True, output)

    error = f"Python execution failed (exit code {proc.returncode}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return ToolResult(False, error)


def _execute_cpp(code: str) -> ToolResult:
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        return ToolResult(False, "No C++ compiler (clang++/g++) found on PATH.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src_path = tmp_path / "main.cpp"
        bin_path = tmp_path / "a.out"
        src_path.write_text(code)

        compile_proc = _run_subprocess(
            [compiler, "-std=c++17", "-O2", str(src_path), "-o", str(bin_path)],
            cwd=tmpdir,
        )
        if compile_proc.returncode != 0:
            error = (
                "C++ compilation failed "
                f"(exit code {compile_proc.returncode}).\nSTDOUT:\n{compile_proc.stdout}\n"
                f"STDERR:\n{compile_proc.stderr}"
            )
            return ToolResult(False, error)

        run_proc = _run_subprocess([str(bin_path)], cwd=tmpdir)
        if run_proc.returncode != 0:
            error = (
                "C++ program exited with a non-zero code "
                f"({run_proc.returncode}).\nSTDOUT:\n{run_proc.stdout}\nSTDERR:\n{run_proc.stderr}"
            )
            return ToolResult(False, error)

        output = run_proc.stdout.strip() or run_proc.stderr.strip() or "C++ program produced no output."
        return ToolResult(True, output)


def _execute_java(code: str) -> ToolResult:
    javac = _prefer_homebrew_java("javac") or shutil.which("javac")
    java = _prefer_homebrew_java("java") or shutil.which("java")
    if not (javac and java):
        return ToolResult(False, "Java toolchain (javac/java) not found on PATH.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src_path = tmp_path / "Main.java"
        src_path.write_text(code)

        compile_proc = _run_subprocess([javac, str(src_path)], cwd=tmpdir)
        if compile_proc.returncode != 0:
            error = (
                "Java compilation failed "
                f"(exit code {compile_proc.returncode}).\nSTDOUT:\n{compile_proc.stdout}\n"
                f"STDERR:\n{compile_proc.stderr}"
            )
            return ToolResult(False, error)

        run_proc = _run_subprocess([java, "-cp", tmpdir, "Main"], cwd=tmpdir)
        if run_proc.returncode != 0:
            error = (
                "Java program exited with a non-zero code "
                f"({run_proc.returncode}).\nSTDOUT:\n{run_proc.stdout}\nSTDERR:\n{run_proc.stderr}"
            )
            return ToolResult(False, error)

        output = run_proc.stdout.strip() or run_proc.stderr.strip() or "Java program produced no output."
        return ToolResult(True, output)


def execute_tool(tool_name: str, code: str) -> ToolResult:
    if tool_name == "code_exec_python":
        return _execute_python(code)
    if tool_name == "code_exec_cpp":
        return _execute_cpp(code)
    if tool_name == "code_exec_java":
        return _execute_java(code)
    return ToolResult(False, f"Unsupported tool '{tool_name}'.")

def create_response(
    input_messages: List[dict],
    previous_response_id: Optional[str] = None,
):
    """Wrapper around ``client.responses.create``.

    Parameters
    ----------
    input_messages: List[dict]
        The running conversation history to feed to the model.
    previous_response_id: str | None
        Pass the ``response.id`` from the *previous* call so the model can keep
        the thread of the conversation.  Omit on the very first request.
    """
    kwargs = {
        "model": MODEL_NAME,
        "input": input_messages,
        "text": {"format": {"type": "text"}},
        "tools": TOOLS,
    }
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    return client.responses.create(**kwargs)

# Recursive 
def run_conversation(
    input_messages: List[dict],
    previous_response_id: Optional[str] = None,
):
  
    response = create_response(input_messages, previous_response_id)

    # ``response.output`` is expected to be a list where element 0 is the model
    # message.  Element 1 (if present) denotes a tool call.  When the model is
    # done with tool calls, that element is omitted.
    tool_call = response.output[1] if len(response.output) > 1 else None

    if tool_call and tool_call.type == "custom_tool_call":
        print("--- tool name ---")
        print(tool_call.name)
        print("--- tool call argument (generated code) ---")
        print(tool_call.input)
        tool_result = execute_tool(tool_call.name, tool_call.input)
        print("--- tool output ---")
        print(tool_result.output)
        
        # Add a synthetic *tool result* so the model can continue the thread.
        
        input_messages.append(
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": tool_result.output,
            }
        )

        # Recurse with updated conversation and track the response id so the
        # model is aware of the prior turn.
        return run_conversation(input_messages, previous_response_id=response.id)
    else:
        # Base-case: no further tool call - return. 
        return 


prompt = """
Write code to sort the array of numbers in three languages: C++, Python and Java (10 times each)using code_exec functions.

ALWAYS CALL THESE THREE FUNCTIONS EXACTLY ONCE: code_exec_python, code_exec_cpp and code_exec_java tools to sort the array in each language. Stop once you've called these three functions in each language once.

Print only the time it takes to sort the array in milliseconds. 

[448, 986, 255, 884, 632, 623, 246, 439, 936, 925, 644, 159, 777, 986, 706, 723, 534, 862, 195, 686, 846, 880, 970, 276, 613, 736, 329, 622, 870, 284, 945, 708, 267, 327, 678, 807, 687, 890, 907, 645, 364, 333, 385, 262, 730, 603, 945, 358, 923, 930, 761, 504, 870, 561, 517, 928, 994, 949, 233, 137, 670, 555, 149, 870, 997, 809, 180, 498, 914, 508, 411, 378, 394, 368, 766, 486, 757, 319, 338, 159, 585, 934, 654, 194, 542, 188, 934, 163, 889, 736, 792, 737, 667, 772, 198, 971, 459, 402, 989, 949]
"""

# Initial developer message.
messages = [
    {
        "role": "developer",
        "content": prompt,
    }
]

if __name__ == "__main__":
    run_conversation(messages)
