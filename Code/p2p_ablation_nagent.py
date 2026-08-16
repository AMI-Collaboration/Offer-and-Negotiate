# p2p_ablation_nagent.py
#
# Ablation Study (N-agent): Offer / Negotiation 각 구성요소의 기여도 측정
#
# run_n()의 플래그만 바꿔서 실행 — 알고리즘은 그대로, 조건만 다름.
# 측정: PT (elapsed time), TC (token cost)
#
# 조건 (요청하신 3개):
#   Full P2P (Ours)  : use_offer=True,  use_negotiation=True
#   w/o Offer        : use_offer=False, use_negotiation=True   (rule-based PASS 보정 끔)
#   w/o Negotiate    : use_offer=True,  use_negotiation=False  (conflict 있어도 협상 안 함)
#
# Human Query는 세 조건 모두 켜둔다 (원본 4-condition ablation에서도 HQ는
# 별도 조건으로 뺐지, Offer/Negotiate 조건에는 항상 켜져 있었음).
#
# 실행:
#   from p2p_ablation_nagent import run_ablation_study_n
#   run_ablation_study_n(
#       task_id="task_003",
#       image_sets=[[img1,img2,img3,img4], ...],   # N=4면 4장씩, N=2면 2장씩
#   )

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from IPython.display import display

from p2p_main_nagent import run_n
from p2p_tracker import tracker
from p2p_utils import _banner


ABLATION_CONDITIONS_N: List[Tuple[str, Dict]] = [
    ("Full P2P (Ours)", dict(use_offer=True,  use_negotiation=True)),
    ("w/o Offer",       dict(use_offer=False, use_negotiation=True)),
    ("w/o Negotiate",   dict(use_offer=True,  use_negotiation=False)),
]


def _save_result(result: Dict, condition: str, pt: float, tc: int, run_idx: int) -> None:
    save_dir = Path("/content/KCC_CoRobot/results_nagent")
    save_dir.mkdir(exist_ok=True, parents=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cond = condition.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "")
    n = result.get("n_agents", "?")
    fname = save_dir / f"ablation_{result['task_id']}_N{n}_{cond}_run{run_idx}_{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump({**result, "condition": condition, "pt": pt, "tc": tc}, f, ensure_ascii=False, indent=2)
    print(f"  -> 저장: {fname}")


def run_ablation_study_n(
    task_id: Optional[str] = None,
    image_sets: Optional[List[List[str]]] = None,
    task: Optional[str] = None,
    verbose: str = "full",
) -> pd.DataFrame:
    """
    Args:
        task_id    : tasks.json에 등록된 task를 조회할 때 사용. task=를 직접 넘기면 무시됨.
        image_sets : [[img_A, img_B, ...], ...] — 태스크당 N장짜리 이미지 세트 리스트.
                     세트마다 이미지 개수(N)가 곧 agent 수. N=2/N=4 섞어서 넣어도 됨.
        task       : task 설명을 문자열로 직접 전달. 지정하면 tasks.json 조회 없이 바로 이 텍스트를 쓴다.
                     예: run_ablation_study_n(task="...", image_sets=[[...]])
        verbose    : "full"(기본) - run_n을 직접 호출했을 때처럼 Offer/Draft/Coordinate
                     원문까지 다 콘솔에 출력. "summary" - 단계 배너와 최종 플랜만 출력.
    """
    if not task and not task_id:
        raise ValueError("task 또는 task_id 중 하나는 지정해주세요.")
    if not image_sets:
        raise ValueError("image_sets를 지정해주세요. 예: image_sets=[[img1,img2,img3,img4], ...]")
    task_id = task_id or "custom_task"

    SEP = "=" * 68
    print(SEP)
    print(f"  ABLATION STUDY (N-agent)  |  task={task_id}  |  runs={len(image_sets)}")
    print(SEP)

    all_rows: Dict[str, List[Dict]] = {cond: [] for cond, _ in ABLATION_CONDITIONS_N}

    for run_idx, images in enumerate(image_sets, 1):
        n_agents = len(images)
        print(f"\n{'-'*68}")
        print(f"  [Run {run_idx}/{len(image_sets)}] N={n_agents}")
        print(f"  images: {images}")
        print(f"{'-'*68}")

        for condition, flags in ABLATION_CONDITIONS_N:
            _banner(f"ABLATION (N={n_agents}) - {condition}")
            print(f"  Flags: {flags}")

            tracker.start()
            try:
                result = run_n(
                    task_id=task_id, task=task, images=images,
                    label=f"{condition} (N={n_agents})",
                    verbose=verbose,
                    **flags,
                )
            except Exception as e:
                print(f"  [ERROR] {condition}: {e}")
                tracker.stop()
                all_rows[condition].append({"pt": 0.0, "tc": 0, "n_agents": n_agents})
                continue
            tracker.stop()

            pt, tc = tracker.elapsed, tracker.total_tokens
            print(tracker.summary(f"{condition} N={n_agents}"))
            _save_result(result, condition, pt, tc, run_idx)
            all_rows[condition].append({"pt": pt, "tc": tc, "n_agents": n_agents})

    # ── 결과 테이블 (조건 x N 별로 분리 집계) ────────────────────────────────
    n_values = sorted({len(imgs) for imgs in image_sets})
    final_rows = []
    for condition, _ in ABLATION_CONDITIONS_N:
        rows = all_rows[condition]
        for n in n_values:
            sub = [r for r in rows if r["n_agents"] == n]
            if not sub:
                continue
            final_rows.append({
                "Condition": condition,
                "N": n,
                "PT(s)": round(float(np.mean([r["pt"] for r in sub])), 2),
                "TC": int(np.mean([r["tc"] for r in sub])),
            })

    df = pd.DataFrame(final_rows)[["Condition", "N", "PT(s)", "TC"]]

    print("\n" + "#" * 68)
    print("  Table. Ablation Study (N-agent) - PT / TC")
    print("#" * 68)
    display(df.style.hide(axis="index").format({"PT(s)": "{:.2f}", "TC": "{:,}"})
            .set_properties(**{"text-align": "center"}))
    print("\n[Markdown]")
    print(df.to_markdown(index=False))
    print("\n※ 플랜 품질(conflict/negotiation 통계 등)은 각 run의 저장된 JSON(metrics 필드)에서 확인하세요.")

    return df
