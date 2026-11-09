# Agent prompts

- `analysis.txt`: training anomaly detection and diagnosis.
- `planner.txt`: autonomous config patches, trial count/order and shared epoch count.
- `evaluation.txt`: comparison of control and trial snapshots.
- `local.txt`: additional instructions for the local Qwen backend.
- `response_format.txt`: shared JSON output instructions, followed by the generated schema.

Edit the text files to change instructions. The loader reads them on each request,
so changes apply to the next agent call, including an active run that already uses
this loader. Runs started before this refactor need to be restarted to load it.
The Qwen server does not need a restart.

The schema is generated from Python contracts and remains in code. Prompts guide
the LLM; changing a prompt does not change validators or engine capabilities.
