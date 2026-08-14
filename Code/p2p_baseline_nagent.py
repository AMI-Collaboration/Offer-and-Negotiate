# p2p_baseline_nagent.py
#
# Baseline Comparison (N-agent): Centralized / Independent
#
# 원본 p2p_baseline.py(2-agent)와 설계 원칙 100% 동일:
#   - can_do/cannot_do/can_provide/need_from_other 등 우리 방법론(Offer) 구조 전혀 없음
#   - observation은 자연어 묘사만 (이미지를 실제로 봄)
#   - few-shot은 출력 JSON 포맷만 가이드, 방법론적 개입 없음
#
# Centralized:
#   Step 1..N. VLM(img_i) → Room i 자연어 묘사 (병렬, N-agent)
#   Step N+1.  VLM(img_0) + N개 묘사 + task → joint plan (단일 플래너, handoff 없음)
#
# Independent:
#   Step 1..N. VLM(img_i) → Room i 자연어 묘사 (병렬)
#   Step N+1.  각자 묘사 + task → 각자 plan (병렬, 상대방 모름)
#   Step N+2.  rule-based merge (step_id 오프셋만 조정, 협상/보정 없음)
#
# step_id 오프셋은 p2p_config_nagent.step_offset()을 그대로 재사용해서
# 본 P2P 파이프라인(run_n)과 동일한 스킴을 쓴다 (agent_index * 1000).
#
# 실행:
#   from p2p_baseline_nagent import run_baseline_comparison_n
#   run_baseline_comparison_n(
#       task_id="task_003",
#       image_sets=[[img1,img2,img3,img4], ...],   # 태스크별 N장 이미지 세트 리스트
#   )

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from IPython.display import display

import p2p_vlm
from p2p_config_nagent import make_agent_ids, step_offset
from p2p_main import get_task
from p2p_phase_nagent import format_joint_plan_n
from p2p_phases import _banner, _log, _run_parallel
from p2p_tracker import tracker
from p2p_utils import extract_json


# ══════════════════════════════════════════════════════════════════════════
# OBSERVATION PROMPT (공통, 원본과 동일 — 방법론 구조 없음)
# ══════════════════════════════════════════════════════════════════════════

def _build_observation_prompt(task: str) -> str:
    return f"""Look at this image carefully.

Task: "{task}"

Describe the following in natural language:
1. What room is this?
2. What objects and areas do you see?
3. What actions can be done in this room to help with the task?

Be specific about visible objects. Keep it concise."""


def _observe_all_n(images: List[str], agent_ids: List[str], task: str, verbose: str = "full") -> Dict[str, str]:
    prompt = _build_observation_prompt(task)
    results = _run_parallel([(img, prompt, False) for img in images])
    obs = {}
    for aid, (text, _) in zip(agent_ids, results):
        obs[aid] = text
        if verbose == "full":
            _log(f"{aid} OBSERVATION", text)
    return obs


# ══════════════════════════════════════════════════════════════════════════
# CENTRALIZED (N-agent)
# ══════════════════════════════════════════════════════════════════════════

def _build_centralized_plan_prompt_n(task: str, obs: Dict[str, str], agent_ids: List[str]) -> str:
    rooms_block = "\n\n".join(f"Room {aid} ({aid}):\n{obs[aid]}" for aid in agent_ids)
    example_agent = agent_ids[0]
    offset_example = step_offset(1)  # 두 번째 agent의 오프셋을 예시로 보여줌
    few_shot = f"""
EXAMPLE OUTPUT FORMAT (repeat this "agent_id: [steps]" pattern for EVERY agent listed above):
<JSON>
{{
  "{example_agent}": [
    {{"step_id": 1, "time_min": 0,  "action": "take water bottle from refrigerator"}},
    {{"step_id": 2, "time_min": 5,  "action": "prepare sandwich on countertop"}}
  ],
  "{agent_ids[1] if len(agent_ids) > 1 else 'agent_B'}": [
    {{"step_id": {offset_example + 1}, "time_min": 0, "action": "take laptop from dresser"}},
    {{"step_id": {offset_example + 2}, "time_min": 5, "action": "tidy bed for cleaner environment"}}
  ]
}}
</JSON>"""

    step_id_rules = "\n".join(
        f"- step_id for {aid}: {step_offset(i) + 1}-{step_offset(i) + 999}"
        for i, aid in enumerate(agent_ids)
    )

    return f"""You are coordinating {len(agent_ids)} home agents to complete a task.

Task: "{task}"

{rooms_block}

{few_shot}

Generate a plan for ALL {len(agent_ids)} agents. Each agent works ONLY in their own room.
{step_id_rules}
- Generate 4-6 steps per agent
- No handoff or transfer between agents

Return ONLY valid JSON inside <JSON> tags, with one key per agent_id listed above."""


def run_centralized_n(
    task_id: Optional[str] = None, images: Optional[List[str]] = None,
    agent_ids: Optional[List[str]] = None, task: Optional[str] = None,
    verbose: str = "full",
) -> Dict:
    n_agents = len(images)
    agent_ids = agent_ids or make_agent_ids(n_agents)
    task_str = task or get_task(task_id)
    task_id = task_id or "custom_task"

    _banner(f"CENTRALIZED (N={n_agents}) - STEP 1..N: OBSERVATION (자연어, 구조화 없음)")
    obs = _observe_all_n(images, agent_ids, task_str, verbose=verbose)

    _banner(f"CENTRALIZED (N={n_agents}) - STEP N+1: JOINT PLAN (단일 플래너, handoff 없음)")
    prompt = _build_centralized_plan_prompt_n(task_str, obs, agent_ids)
    raw, _ = p2p_vlm.run_vlm(images[0], prompt)  # 단일 플래너 — 원본과 동일하게 대표 이미지 1장만 사용
    if verbose == "full":
        _log("CENTRALIZED RAW PLAN", raw)

    data = extract_json(raw)
    if not isinstance(data, dict):
        data = {}

    def _parse(steps_raw, agent_id, idx) -> List[Dict]:
        if not isinstance(steps_raw, list):
            return []
        offset = step_offset(idx)
        out = []
        for s in steps_raw:
            if not isinstance(s, dict) or "action" not in s:
                continue
            sid = s.get("step_id", len(out) + 1)
            out.append({
                "step_id": sid if sid >= offset else sid + offset,
                "time_min": s.get("time_min", 0),
                "agent_id": agent_id,
                "room": agent_id,
                "action": s.get("action", ""),
                "depends_on": [],
                "handoff_type": None,
                "target_agent": None,
            })
        return out

    all_steps: List[Dict] = []
    for idx, aid in enumerate(agent_ids):
        all_steps.extend(_parse(data.get(aid, []), aid, idx))
    joint_plan = sorted(all_steps, key=lambda s: (s.get("time_min", 0), s.get("step_id", 0)))

    counts = ", ".join(f"{aid}={sum(1 for s in joint_plan if s['agent_id']==aid)}" for aid in agent_ids)
    print(f"\n  {counts} | total: {len(joint_plan)}")
    print("\n" + "#" * 68)
    print(f"  FINAL JOINT PLAN - Centralized (N={n_agents})")
    print("#" * 68)
    print(format_joint_plan_n(joint_plan, agent_ids, task_str))

    return {
        "method": "Centralized", "task_id": task_id, "task": task_str,
        "n_agents": n_agents, "agent_ids": agent_ids, "joint_plan": joint_plan,
    }


# ══════════════════════════════════════════════════════════════════════════
# INDEPENDENT (N-agent)
# ══════════════════════════════════════════════════════════════════════════

_INDEPENDENT_FEW_SHOT = """
EXAMPLE OUTPUT FORMAT:
<JSON>
{
  "plan_steps": [
    {"step_id": 1, "time_min": 0,  "action": "take water bottle from refrigerator"},
    {"step_id": 2, "time_min": 5,  "action": "prepare sandwich on countertop"},
    {"step_id": 3, "time_min": 10, "action": "place food on kitchen counter"}
  ]
}
</JSON>"""


def _build_independent_plan_prompt(task: str, obs: str) -> str:
    return f"""You are a home agent working independently.

Task: "{task}"

Your room:
{obs}

{_INDEPENDENT_FEW_SHOT}

Generate a plan for YOUR room only.
- Only actions possible in your room
- Generate 4-6 steps
- No handoff or transfer to other agents

Return ONLY valid JSON inside <JSON> tags."""


def _rule_based_merge_n(steps_by_agent: Dict[str, List[Dict]], agent_ids: List[str]) -> List[Dict]:
    """협상/보정 없이 오프셋만 맞춰서 그대로 합침 (원본 _rule_based_merge의 N-agent 버전)."""
    merged: List[Dict] = []
    for aid in agent_ids:
        merged.extend(steps_by_agent.get(aid, []))
    merged.sort(key=lambda s: (s.get("time_min", 0), s.get("step_id", 0)))
    return merged


def run_independent_n(
    task_id: Optional[str] = None, images: Optional[List[str]] = None,
    agent_ids: Optional[List[str]] = None, task: Optional[str] = None,
    verbose: str = "full",
) -> Dict:
    n_agents = len(images)
    agent_ids = agent_ids or make_agent_ids(n_agents)
    task_str = task or get_task(task_id)
    task_id = task_id or "custom_task"

    _banner(f"INDEPENDENT (N={n_agents}) - STEP 1..N: OBSERVATION (자연어, 구조화 없음)")
    obs = _observe_all_n(images, agent_ids, task_str, verbose=verbose)

    _banner(f"INDEPENDENT (N={n_agents}) - STEP N+1: LOCAL PLANNING (상대방 모름)")
    prompts = [_build_independent_plan_prompt(task_str, obs[aid]) for aid in agent_ids]
    results = _run_parallel(list(zip(images, prompts, [False] * n_agents)))
    if verbose == "full":
        for aid, (raw, _) in zip(agent_ids, results):
            _log(f"{aid} RAW PLAN", raw)

    def _parse(raw: str, agent_id: str, idx: int) -> List[Dict]:
        data = extract_json(raw)
        if isinstance(data, list):
            data = {"plan_steps": data}
        if not isinstance(data, dict):
            return []
        offset = step_offset(idx)
        out = []
        for s in data.get("plan_steps", []):
            if not isinstance(s, dict) or "action" not in s:
                continue
            sid = s.get("step_id", len(out) + 1)
            out.append({
                "step_id": sid if sid >= offset else sid + offset,
                "time_min": s.get("time_min", 0),
                "agent_id": agent_id,
                "room": agent_id,
                "action": s.get("action", ""),
                "depends_on": [],
                "handoff_type": None,
                "target_agent": None,
            })
        return out

    steps_by_agent: Dict[str, List[Dict]] = {}
    for idx, (aid, (raw, _)) in enumerate(zip(agent_ids, results)):
        steps_by_agent[aid] = _parse(raw, aid, idx)

    counts = ", ".join(f"{aid}={len(steps_by_agent[aid])}" for aid in agent_ids)
    print(f"\n  {counts}")

    _banner(f"INDEPENDENT (N={n_agents}) - STEP N+2: RULE-BASED MERGE")
    joint_plan = _rule_based_merge_n(steps_by_agent, agent_ids)
    print(f"  Merged: {len(joint_plan)} steps total")
    print("\n" + "#" * 68)
    print(f"  FINAL JOINT PLAN - Independent (N={n_agents})")
    print("#" * 68)
    print(format_joint_plan_n(joint_plan, agent_ids, task_str))

    return {
        "method": "Independent", "task_id": task_id, "task": task_str,
        "n_agents": n_agents, "agent_ids": agent_ids, "joint_plan": joint_plan,
    }


# ══════════════════════════════════════════════════════════════════════════
# SAVE + BATCH RUNNER
# ══════════════════════════════════════════════════════════════════════════

def _save_result(result: Dict, pt: float, tc: int, run_idx: int) -> None:
    save_dir = Path("/content/KCC_CoRobot/results_nagent")
    save_dir.mkdir(exist_ok=True, parents=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    method = result["method"].replace(" ", "_")
    n = result["n_agents"]
    fname = save_dir / f"baseline_{result['task_id']}_N{n}_{method}_run{run_idx}_{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump({**result, "pt": pt, "tc": tc}, f, ensure_ascii=False, indent=2)
    print(f"  -> 저장: {fname}")


def run_baseline_comparison_n(
    task_id: Optional[str] = None,
    image_sets: Optional[List[List[str]]] = None,
    task: Optional[str] = None,
    verbose: str = "full",
) -> pd.DataFrame:
    """
    Args:
        task_id    : tasks.json에 등록된 task를 조회할 때 사용. task=를 직접 넘기면 무시됨.
        image_sets : [[img_A, img_B, ...], ...] — 태스크당 N장짜리 이미지 세트 리스트.
                     세트마다 이미지 개수(N)가 곧 agent 수가 된다 (N=2든 N=4든 그대로 지원).
        task       : task 설명을 문자열로 직접 전달. 지정하면 tasks.json 조회 없이 바로 이 텍스트를 쓴다.
                     예: run_baseline_comparison_n(task="...", image_sets=[[...]])
        verbose    : "full"(기본) - observation/raw plan 원문까지 다 콘솔에 출력.
                     "summary" - 단계 배너와 최종 플랜만 출력.
    """
    if not task and not task_id:
        raise ValueError("task 또는 task_id 중 하나는 지정해주세요.")
    task_id = task_id or "custom_task"

    conditions = [("Centralized", run_centralized_n), ("Independent", run_independent_n)]

    SEP = "=" * 68
    print(SEP)
    print(f"  BASELINE COMPARISON (N-agent)  |  task={task_id}  |  runs={len(image_sets)}")
    print(SEP)

    all_rows: Dict[str, List[Dict]] = {name: [] for name, _ in conditions}

    for run_idx, images in enumerate(image_sets, 1):
        n_agents = len(images)
        print(f"\n{'-'*68}")
        print(f"  [Run {run_idx}/{len(image_sets)}] N={n_agents}")
        print(f"  images: {images}")
        print(f"{'-'*68}")

        for method_name, run_fn in conditions:
            _banner(f"BASELINE - {method_name} (N={n_agents})")
            tracker.start()
            try:
                result = run_fn(task_id=task_id, images=images, task=task, verbose=verbose)
            except Exception as e:
                print(f"  [ERROR] {method_name}: {e}")
                tracker.stop()
                all_rows[method_name].append({"pt": 0.0, "tc": 0, "n_agents": n_agents})
                continue
            tracker.stop()

            pt, tc = tracker.elapsed, tracker.total_tokens
            print(tracker.summary(f"{method_name} N={n_agents}"))
            _save_result(result, pt, tc, run_idx)
            all_rows[method_name].append({"pt": pt, "tc": tc, "n_agents": n_agents})

    final_rows = []
    for method_name, _ in conditions:
        rows = all_rows[method_name]
        final_rows.append({
            "Method": method_name,
            "PT(s)": round(float(np.mean([r["pt"] for r in rows])), 2),
            "TC": int(np.mean([r["tc"] for r in rows])),
        })

    df = pd.DataFrame(final_rows)[["Method", "PT(s)", "TC"]]

    print("\n" + "#" * 68)
    print("  Table. Baseline Comparison (N-agent) - PT / TC")
    print("#" * 68)
    display(df.style.hide(axis="index").format({"PT(s)": "{:.2f}", "TC": "{:,}"})
            .set_properties(**{"text-align": "center"}))
    print("\n[Markdown]")
    print(df.to_markdown(index=False))
    print("\n※ 플랜 품질은 위 자연어 출력을 통해 확인하세요.")

    return df
