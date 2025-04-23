import json
import requests
from typing import List, Dict, Any, Optional

# Placeholder unit test
fake_unit_test = [{"input": "0\r\n", "output": ["0"]}]

def invoke_tool(
    url: str,
    language: str,
    source_code: str,
    unittests: Optional[List[Dict[str, Any]]] = None,
    compile_cmd: Optional[str] = None,
    compile_flags: Optional[str] = None,
    execute_cmd: str = "",
    execute_flags: str = "",
    block_network: bool = True,
    stop_on_first_fail: bool = False,
    use_sanitizer: bool = False,
    limits: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Invoke a single tool endpoint with a standardized JSON payload.
    """
    payload = {
        "language":          language,
        "source_code":       source_code,
        "unittests":         unittests or [],
        "limits":            limits or {},
        "compile_cmd":       compile_cmd,
        "compile_flags":     compile_flags,
        "execute_cmd":       execute_cmd,
        "execute_flags":     execute_flags,
        "block_network":     block_network,
        "stop_on_first_fail": stop_on_first_fail,
        "use_sanitizer":     use_sanitizer
    }

    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, data=json.dumps(payload))
    if resp.status_code != 200:
        raise RuntimeError(f"Tool call failed [{resp.status_code}]: {resp.text}")
    return resp.json()


def eval_cpp(
    url: str,
    source_code: str,
    test_id: Optional[str] = None,
    unittests: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> Dict[str, Any]:
    return invoke_tool(
        url,
        language="GNU C++",
        source_code=source_code,
        unittests=unittests or fake_unit_test,
        compile_cmd="g++",
        compile_flags="-std=c++17 -O2 -Wall",
        execute_cmd="./a.out",
        **kwargs
    )


def eval_python(
    url: str,
    source_code: str,
    test_id: Optional[str] = None,
    unittests: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> Dict[str, Any]:
    return invoke_tool(
        url,
        language="PyPy 3",
        source_code=source_code,
        unittests=unittests or fake_unit_test,
        execute_cmd="python3",
        **kwargs
    )


def eval_java(
    url: str,
    source_code: str,
    test_id: Optional[str] = None,
    unittests: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> Dict[str, Any]:
    return invoke_tool(
        url,
        language="Java 17",
        source_code=source_code,
        unittests=unittests or fake_unit_test,
        **kwargs
    )

# Utility functions

def check_compile(response_json: Dict[str, Any]) -> int:
    entries = response_json.get("data", [])
    return 0 if any(e.get("exec_outcome") == "COMPILATION_ERROR" for e in entries) else 1


def calculate_wrong_answer_proportion(response_json: Dict[str, Any]) -> float:
    entries = response_json.get("data", [])
    if not entries:
        return 0.0
    wrong = sum(1 for e in entries if e.get("exec_outcome") == "WRONG_ANSWER")
    return wrong / len(entries)

# Main with original tests
if __name__ == "__main__":
    tool_url = "http://localhost:5000/api/execute_code"

    # C++ test
    cpp_test_src = """#include <iostream>
int main() {
    std::cout << \"Hello, world!\" << std::endl;
    return 0;
}
"""
    print("C++ response:", eval_cpp(tool_url, cpp_test_src, test_id="0"))

    # Java test
    java_test_id = "18d4fae42678823e57388b5359f3be61"
    java_test_src = """import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.StringTokenizer;

public class Archer {
    public static void main(String[] args) {
        MyScanner sc = new MyScanner();
        
        int A = sc.nextInt();
        int B = sc.nextInt();
        int C = sc.nextInt();
        int D = sc.nextInt();
        
        double r = (double) A / B;
        double z = (double) C / D;
        
        System.out.println(r / (1 - (1 - r) * (1 - z)));
    }

    public static class MyScanner {
        BufferedReader br;
        StringTokenizer st;

        public MyScanner() {
            br = new BufferedReader(new InputStreamReader(System.in));
        }

        int nextInt() {
            return Integer.parseInt(next());
        }
        
        String next() {
            while (st == null || !st.hasMoreElements()) {
                try {
                    st = new StringTokenizer(br.readLine());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            return st.nextToken();
        }
    }
}
"""
    print("Java response:", eval_java(tool_url, java_test_src, test_id=java_test_id))

    # Python test
    source_code = (
        "import sys\n\n"
        "def main():\n"
        "    data = sys.stdin.read().strip()\n"
        "    numbers = list(map(int, data.split()))\n"
        "    print(sum(numbers))\n\n"
        "if __name__ == '__main__':\n"
        "    main()"
    )
    print("Python response:", eval_python(tool_url, source_code, test_id="0"))
