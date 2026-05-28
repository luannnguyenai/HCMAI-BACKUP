"""Generate `smoke_20.jsonl` with proper Vietnamese diacritics.

This script exists because some host toolchains (Windows + PowerShell on
this particular workstation, in particular) mangle non-ASCII content when
writing files through a CP1252 fallback path. To stay reproducible, the
Vietnamese strings below are encoded as Python ``\\uXXXX`` escapes -
pure ASCII source bytes that cannot be corrupted in transit. Python
decodes the escapes at runtime into proper Unicode, and we write the
JSONL via an explicit ``encoding="utf-8"`` so the on-disk file is real
UTF-8 with full diacritics.

Run from repo root:

    uv run python tests/mock_tasks/_generate_smoke.py

The output file is checked into git; this script is the authoritative
source. Re-run after edits.
"""

from __future__ import annotations

import json
from pathlib import Path

TASKS: list[dict] = [
    # --- KIS ---------------------------------------------------------------
    {
        "task_id": "KIS-0001",
        "task_type": "KIS",
        "query_vi": (
            "Tr\u1ebb em ch\u1ea1y nh\u1ea3y d\u01b0\u1edbi m\u01b0a "
            "\u1edf s\u00e2n ch\u01a1i tr\u01b0\u1eddng h\u1ecdc."
        ),
        "query_en": "Children running in rain at a school playground.",
        "time_limit_seconds": 300,
        "ground_truth": {"kis_frame_ids": ["vid_001_f000123", "vid_001_f000124"]},
        "metadata": {"difficulty": "easy", "place": "school", "source": "placeholder"},
    },
    {
        "task_id": "KIS-0002",
        "task_type": "KIS",
        "query_vi": (
            "Ph\u1ecfng v\u1ea5n n\u1eef ph\u00e1t thanh vi\u00ean VTV "
            "m\u1eb7c \u00e1o d\u00e0i \u0111\u1ecf."
        ),
        "query_en": "Interview with a female VTV anchor in a red ao dai.",
        "time_limit_seconds": 300,
        "ground_truth": {"kis_frame_ids": ["vid_017_f002411"]},
        "metadata": {"difficulty": "medium", "place": "studio", "source": "placeholder"},
    },
    {
        "task_id": "KIS-0003",
        "task_type": "KIS",
        "query_vi": (
            "C\u1ea3nh giao th\u00f4ng gi\u1edd cao \u0111i\u1ec3m tr\u00ean "
            "c\u1ea7u S\u00e0i G\u00f2n."
        ),
        "query_en": "Rush-hour traffic on Saigon Bridge.",
        "time_limit_seconds": 300,
        "ground_truth": {
            "kis_frame_ids": ["vid_042_f000812", "vid_042_f000813", "vid_042_f000814"],
        },
        "metadata": {"difficulty": "easy", "place": "outdoor_street", "source": "placeholder"},
    },
    {
        "task_id": "KIS-0004",
        "task_type": "KIS",
        "query_vi": (
            "L\u1ec5 h\u1ed9i ph\u1ed1 \u0111\u00eam \u1edf ch\u1ee3 "
            "B\u1ebfn Th\u00e0nh v\u1edbi \u0111\u00e8n l\u1ed3ng \u0111\u1ecf."
        ),
        "query_en": "Night-market scene at Ben Thanh with red lanterns.",
        "time_limit_seconds": 300,
        "ground_truth": {"kis_frame_ids": ["vid_064_f001100"]},
        "metadata": {"difficulty": "hard", "place": "market", "source": "placeholder"},
    },
    {
        "task_id": "KIS-0005",
        "task_type": "KIS",
        "query_vi": (
            "H\u1ecdc sinh x\u1ebfp h\u00e0ng ch\u00e0o c\u1edd t\u1ea1i s\u00e2n tr\u01b0\u1eddng."
        ),
        "query_en": "Pupils lined up for the flag ceremony.",
        "time_limit_seconds": 300,
        "ground_truth": {"kis_frame_ids": ["vid_082_f000045", "vid_082_f000046"]},
        "metadata": {"difficulty": "medium", "place": "school", "source": "placeholder"},
    },
    {
        "task_id": "KIS-0006",
        "task_type": "KIS",
        "query_vi": (
            "B\u1ea3n tin d\u1ef1 b\u00e1o th\u1eddi ti\u1ebft c\u00f3 b\u1ea3n "
            "\u0111\u1ed3 Vi\u1ec7t Nam."
        ),
        "query_en": "Weather forecast segment showing a map of Vietnam.",
        "time_limit_seconds": 300,
        "ground_truth": {"kis_frame_ids": ["vid_109_f000700"]},
        "metadata": {"difficulty": "easy", "place": "studio", "source": "placeholder"},
    },
    {
        "task_id": "KIS-0007",
        "task_type": "KIS",
        "query_vi": (
            "Ng\u01b0\u1eddi d\u00e2n b\u00e1n h\u00e0ng rong b\u00e1nh m\u00ec "
            "tr\u00ean v\u1ec9a h\u00e8."
        ),
        "query_en": "Street vendor selling banh mi on the sidewalk.",
        "time_limit_seconds": 300,
        "ground_truth": {"kis_frame_ids": ["vid_133_f002033"]},
        "metadata": {"difficulty": "medium", "place": "outdoor_street", "source": "placeholder"},
    },
    {
        "task_id": "KIS-0008",
        "task_type": "KIS",
        "query_vi": (
            "M\u01b0a ng\u1eadp \u0111\u01b0\u1eddng t\u1ea1i Qu\u1eadn 1 "
            "Th\u00e0nh ph\u1ed1 H\u1ed3 Ch\u00ed Minh."
        ),
        "query_en": "Flooded street in District 1 HCMC.",
        "time_limit_seconds": 300,
        "ground_truth": {"kis_frame_ids": ["vid_158_f000901"]},
        "metadata": {"difficulty": "hard", "place": "outdoor_street", "source": "placeholder"},
    },
    # --- QA ----------------------------------------------------------------
    {
        "task_id": "QA-0001",
        "task_type": "QA",
        "query_vi": (
            "C\u00f3 bao nhi\u00eau chi\u1ebfc xe m\u00e1y m\u00e0u \u0111\u1ecf "
            "trong c\u1ea3nh giao th\u00f4ng gi\u1edd cao \u0111i\u1ec3m?"
        ),
        "query_en": "How many red motorbikes are in the rush-hour traffic scene?",
        "time_limit_seconds": 180,
        "ground_truth": {
            "qa_answer": "ba",
            "qa_answer_acceptable": [
                "3",
                "three",
                "ba chi\u1ebfc",
                "3 chi\u1ebfc",
            ],
        },
        "metadata": {"difficulty": "medium", "place": "outdoor_street", "source": "placeholder"},
    },
    {
        "task_id": "QA-0002",
        "task_type": "QA",
        "query_vi": (
            "M\u00e0u \u00e1o d\u00e0i c\u1ee7a n\u1eef ph\u00e1t thanh "
            "vi\u00ean l\u00e0 m\u00e0u g\u00ec?"
        ),
        "query_en": "What colour is the anchor's ao dai?",
        "time_limit_seconds": 180,
        "ground_truth": {
            "qa_answer": "\u0111\u1ecf",
            "qa_answer_acceptable": ["red", "m\u00e0u \u0111\u1ecf"],
        },
        "metadata": {"difficulty": "easy", "place": "studio", "source": "placeholder"},
    },
    {
        "task_id": "QA-0003",
        "task_type": "QA",
        "query_vi": (
            "T\u00f4i \u0111\u00e3 \u0111\u1ebfn ch\u1ee3 ngo\u00e0i tr\u1eddi "
            "bao nhi\u00eau l\u1ea7n trong th\u00e1ng 2?"
        ),
        "query_en": "How many times did I visit an outdoor market in February?",
        "time_limit_seconds": 180,
        "ground_truth": {
            "qa_answer": "b\u1ed1n l\u1ea7n",
            "qa_answer_acceptable": ["4", "four times", "4 l\u1ea7n"],
        },
        "metadata": {"difficulty": "hard", "place": "market", "source": "placeholder"},
    },
    {
        "task_id": "QA-0004",
        "task_type": "QA",
        "query_vi": (
            "T\u00ean c\u1ee7a ch\u01b0\u01a1ng tr\u00ecnh th\u1eddi s\u1ef1 "
            "bu\u1ed5i t\u1ed1i tr\u00ean VTV1 l\u00e0 g\u00ec?"
        ),
        "query_en": "What is the name of VTV1's evening news programme?",
        "time_limit_seconds": 180,
        "ground_truth": {
            "qa_answer": "Th\u1eddi s\u1ef1 19h",
            "qa_answer_acceptable": ["19h news", "VTV1 Th\u1eddi s\u1ef1"],
        },
        "metadata": {"difficulty": "easy", "place": "studio", "source": "placeholder"},
    },
    {
        "task_id": "QA-0005",
        "task_type": "QA",
        "query_vi": (
            "T\u00ean b\u00e1n h\u00e0ng rong l\u00e0 g\u00ec theo d\u00f2ng "
            "ch\u1eef ch\u1ea1y d\u01b0\u1edbi m\u00e0n h\u00ecnh?"
        ),
        "query_en": "What is the vendor's name per the chyron?",
        "time_limit_seconds": 180,
        "ground_truth": {
            "qa_answer": "Nguy\u1ec5n V\u0103n A",
            "qa_answer_acceptable": [
                "Mr. Nguy\u1ec5n V\u0103n A",
                "Anh Nguy\u1ec5n V\u0103n A",
            ],
        },
        "metadata": {"difficulty": "hard", "place": "outdoor_street", "source": "placeholder"},
    },
    {
        "task_id": "QA-0006",
        "task_type": "QA",
        "query_vi": ("B\u00e3o s\u1ed1 9 \u0111\u1ed5 b\u1ed9 v\u00e0o t\u1ec9nh n\u00e0o?"),
        "query_en": "Which province did Storm No. 9 hit?",
        "time_limit_seconds": 180,
        "ground_truth": {
            "qa_answer": "B\u00ecnh \u0110\u1ecbnh",
            "qa_answer_acceptable": [
                "Binh Dinh province",
                "t\u1ec9nh B\u00ecnh \u0110\u1ecbnh",
            ],
        },
        "metadata": {"difficulty": "medium", "place": "studio", "source": "placeholder"},
    },
    # --- AD_HOC ------------------------------------------------------------
    {
        "task_id": "ADHOC-0001",
        "task_type": "AD_HOC",
        "query_vi": (
            "C\u00e1c c\u1ea3nh ph\u1ecfng v\u1ea5n ng\u01b0\u1eddi d\u00e2n "
            "ngo\u00e0i \u0111\u01b0\u1eddng v\u1ec1 t\u00ecnh h\u00ecnh giao th\u00f4ng."
        ),
        "query_en": "Street interviews with locals about traffic.",
        "time_limit_seconds": 180,
        "ground_truth": {
            "adhoc_frame_ids": [
                "vid_011_f000202",
                "vid_018_f000404",
                "vid_025_f000606",
                "vid_037_f000808",
                "vid_054_f001010",
            ],
        },
        "metadata": {"difficulty": "medium", "place": "outdoor_street", "source": "placeholder"},
    },
    {
        "task_id": "ADHOC-0002",
        "task_type": "AD_HOC",
        "query_vi": (
            "H\u00ecnh \u1ea3nh h\u1ecdc sinh \u0111ang h\u1ecdc b\u00e0i trong l\u1edbp h\u1ecdc."
        ),
        "query_en": "Pupils studying inside a classroom.",
        "time_limit_seconds": 180,
        "ground_truth": {
            "adhoc_frame_ids": [
                "vid_082_f000300",
                "vid_082_f000301",
                "vid_082_f000302",
                "vid_088_f000150",
            ],
        },
        "metadata": {"difficulty": "easy", "place": "school", "source": "placeholder"},
    },
    {
        "task_id": "ADHOC-0003",
        "task_type": "AD_HOC",
        "query_vi": (
            "C\u00e1c c\u1ea3nh b\u00e1n h\u00e0ng rong tr\u00ean v\u1ec9a h\u00e8 "
            "S\u00e0i G\u00f2n."
        ),
        "query_en": "Street-vendor scenes on Saigon sidewalks.",
        "time_limit_seconds": 180,
        "ground_truth": {
            "adhoc_frame_ids": [
                "vid_133_f002000",
                "vid_133_f002033",
                "vid_141_f000450",
                "vid_172_f001212",
            ],
        },
        "metadata": {"difficulty": "medium", "place": "outdoor_street", "source": "placeholder"},
    },
    # --- TRAKE -------------------------------------------------------------
    {
        "task_id": "TRAKE-0001",
        "task_type": "TRAKE",
        "query_vi": (
            "Ph\u00e1t thanh vi\u00ean gi\u1edbi thi\u1ec7u b\u1ea3n tin, "
            "sau \u0111\u00f3 l\u00e0 c\u1ea3nh ph\u1ecfng v\u1ea5n, "
            "r\u1ed3i chuy\u1ec3n sang d\u1ef1 b\u00e1o th\u1eddi ti\u1ebft, "
            "k\u1ebft th\u00fac b\u1eb1ng qu\u1ea3ng c\u00e1o."
        ),
        "query_en": "Anchor intro, then interview, then weather forecast, then advert.",
        "time_limit_seconds": 180,
        "ground_truth": {
            "trake_frame_ids": [
                "vid_109_f000100",
                "vid_109_f000400",
                "vid_109_f000700",
                "vid_109_f000900",
            ],
        },
        "metadata": {"difficulty": "hard", "place": "studio", "source": "placeholder"},
    },
    {
        "task_id": "TRAKE-0002",
        "task_type": "TRAKE",
        "query_vi": (
            "M\u01b0a b\u1eaft \u0111\u1ea7u r\u01a1i, ng\u01b0\u1eddi d\u00e2n "
            "ch\u1ea1y tr\u00fa \u1ea9n, \u0111\u01b0\u1eddng ng\u1eadp, "
            "n\u01b0\u1edbc r\u00fat d\u1ea7n."
        ),
        "query_en": "Rain starts, people seek shelter, street floods, water recedes.",
        "time_limit_seconds": 180,
        "ground_truth": {
            "trake_frame_ids": [
                "vid_158_f000800",
                "vid_158_f000850",
                "vid_158_f000901",
                "vid_158_f001000",
            ],
        },
        "metadata": {"difficulty": "hard", "place": "outdoor_street", "source": "placeholder"},
    },
    {
        "task_id": "TRAKE-0003",
        "task_type": "TRAKE",
        "query_vi": (
            "H\u1ecdc sinh x\u1ebfp h\u00e0ng, ch\u00e0o c\u1edd, h\u00e1t "
            "qu\u1ed1c ca, v\u00e0o l\u1edbp."
        ),
        "query_en": "Pupils line up, raise the flag, sing the anthem, enter the classroom.",
        "time_limit_seconds": 180,
        "ground_truth": {
            "trake_frame_ids": [
                "vid_082_f000045",
                "vid_082_f000080",
                "vid_082_f000120",
                "vid_082_f000200",
            ],
        },
        "metadata": {"difficulty": "medium", "place": "school", "source": "placeholder"},
    },
]


def main() -> None:
    out_path = Path(__file__).parent / "smoke_20.jsonl"
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for task in TASKS:
            fh.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"wrote {len(TASKS)} tasks to {out_path}")


if __name__ == "__main__":
    main()
