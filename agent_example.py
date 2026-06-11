#!/usr/bin/env python3
"""
Mini Agent — a real tool-calling loop with Ollama.

This is the OPPOSITE of your video editor.
Here the LLM DECIDES which tools to call and in what order.
Your code just runs whatever the LLM asks for, then hands the result back.
The `while` loop is what makes it an agent.

Setup:
    ollama pull llama3.2
    python agent.py
"""

import json
import urllib.request

OLLAMA_HOST = "http://localhost:11434"
MODEL = "gemma3"   # change to llama3.2 or mistral to compare behavior


# ════════════════════════════════════════════════════════════════════
#  PART 1: THE TOOLS
#  These are just normal Python functions. The LLM can't run code —
#  it can only ASK us to run one of these by name. We do the running.
# ════════════════════════════════════════════════════════════════════

def calculator(expression: str) -> str:
    """Evaluate a math expression like '23 * 47'."""
    try:
        # Only allow safe math characters
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            return "Error: invalid characters in expression"
        result = eval(expression)
        return f"{result}"
    except Exception as e:
        return f"Error: {e}"


def word_counter(text: str) -> str:
    """Count the words in a piece of text."""
    return f"{len(text.split())} words"


def reverse_text(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


# A registry so we can look up a tool by the name the LLM gives us
TOOLS = {
    "calculator":   calculator,
    "word_counter": word_counter,
    "reverse_text": reverse_text,
}

# Plain-English description of each tool, given to the LLM so it knows
# what's available and how to call them.
TOOLS_DESCRIPTION = """
You have access to these tools:

1. calculator(expression)  — does math. Example: calculator("23 * 47")
2. word_counter(text)      — counts words. Example: word_counter("hello there world")
3. reverse_text(text)      — reverses a string. Example: reverse_text("hello")
"""


# ════════════════════════════════════════════════════════════════════
#  PART 2: TALK TO OLLAMA
# ════════════════════════════════════════════════════════════════════

def ask_ollama(prompt: str) -> str:
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1}   # low temp = more reliable JSON
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read()).get("response", "").strip()


def extract_json(raw: str) -> dict:
    """Pull a JSON object out of the model's reply, even if wrapped in text."""
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return json.loads(raw)


# ════════════════════════════════════════════════════════════════════
#  PART 3: THE AGENT LOOP  ← THIS is what makes it an agent
# ════════════════════════════════════════════════════════════════════

def run_agent(user_goal: str, max_steps: int = 6):
    print(f"\n🎯 GOAL: {user_goal}\n" + "─" * 55)

    # We keep a running "scratchpad" of everything that has happened.
    # On each loop we show the LLM the goal + history and ask: what next?
    history = ""
    seen_calls = set()   # safety net: track tool calls we've already made

    for step in range(1, max_steps + 1):
        prompt = f"""You are an agent that solves tasks step by step using tools.

{TOOLS_DESCRIPTION}

THE USER'S GOAL:
{user_goal}

WHAT HAS HAPPENED SO FAR:
{history if history else "(nothing yet — this is your first step)"}

STOP AND THINK FIRST: Look at "WHAT HAS HAPPENED SO FAR" above.
- Has every part of the goal already been answered there?
- If YES, you MUST finish now. Do NOT repeat a tool you already used.
- Only use a tool for a part of the goal that has NOT been done yet.

Decide your NEXT single action. Reply with ONLY a JSON object, nothing else.

To use a tool (ONLY for a part not yet done):
{{"action": "use_tool", "tool": "calculator", "input": "23 * 47", "thought": "why you chose this"}}

To finish (when all parts of the goal appear in the history above):
{{"action": "finish", "answer": "combine all the results into one final answer", "thought": "why you're done"}}

IMPORTANT: If the history already contains the word count, the multiplication
result, and the reversed phrase, then ALL parts are done — you MUST choose
"finish" and assemble those results into your answer. Do not loop.

Your JSON response:"""

        raw = ask_ollama(prompt)

        try:
            decision = extract_json(raw)
        except Exception:
            print(f"⚠️  Step {step}: couldn't parse model output:\n{raw[:200]}")
            break

        thought = decision.get("thought", "")

        # ── The LLM decided it's done ──
        if decision.get("action") == "finish":
            print(f"\n🧠 Step {step} — thought: {thought}")
            print(f"\n✅ FINAL ANSWER: {decision.get('answer')}")
            print("─" * 55)
            return decision.get("answer")

        # ── The LLM wants to use a tool ──
        tool_name = decision.get("tool")
        tool_input = decision.get("input", "")

        print(f"\n🧠 Step {step} — thought: {thought}")
        print(f"🔧 LLM chose tool: {tool_name}(\"{tool_input}\")")

        # ── SAFETY NET: detect a repeated tool call (the loop we saw before) ──
        call_signature = f"{tool_name}|{tool_input}"
        if call_signature in seen_calls:
            print(f"⚠️  The model tried to repeat a call it already made.")
            print(f"   Stopping the loop — all work is already in the history.")
            print(f"\n✅ COLLECTED RESULTS:{history}")
            print("─" * 55)
            return history
        seen_calls.add(call_signature)

        if tool_name not in TOOLS:
            result = f"Error: no tool named '{tool_name}'"
        else:
            result = TOOLS[tool_name](tool_input)   # ← WE run the tool

        print(f"📤 Tool result: {result}")

        # Feed the result back into history so the LLM sees it next loop
        history += (
            f"\nStep {step}: used {tool_name}(\"{tool_input}\") "
            f"→ got: {result}"
        )

    print("\n⚠️  Hit max steps without finishing.")
    return None


# ════════════════════════════════════════════════════════════════════
#  PART 4: TRY IT
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # A task that REQUIRES multiple tool calls in an order the LLM
    # must figure out itself. This is where an agent beats a script.
    run_agent(
        "Take the phrase 'automation is leverage'. "
        "Count how many words it has, then multiply that number by 100, "
        "then reverse the phrase. Give me all three results."
    )

    # Try your own:
    # run_agent("What is 1500 divided by 12, and how many words are in this sentence?")
