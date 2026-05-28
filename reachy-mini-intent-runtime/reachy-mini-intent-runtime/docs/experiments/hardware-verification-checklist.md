# Hardware Verification Checklist

Use this before claiming the system works on the real robot.

## Experiment 1: Native stop behavior

- Start official dance through SDK or conversation app.
- Immediately issue official `stop_dance`.
- Record whether the robot stops immediately, waits until current macro ends, or clears only queued future moves.

## Experiment 2: Chunked dance behavior

- Convert one long dance into chunks.
- Measure gap between chunks.
- Human evaluator marks smoothness: smooth / noticeable / unacceptable.

## Experiment 3: CPU budget

- Run motion + camera sampling + speech/VAD + stop loop.
- Record CPU utilization and missed heartbeat count.
- Repeat with off-board LLM/VLM and, separately, with any local vision attempt.

## Experiment 4: Hospital scenario

- Synthetic patient asks for orientation.
- Reachy answers and uses short emotion.
- User asks for dance.
- User interrupts with “停止跳舞” at T+5s.
- User asks for nurse.

Do not use real patient data.
