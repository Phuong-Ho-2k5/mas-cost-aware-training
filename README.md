# MAS — autonomous training trials

Demo huấn luyện có ba Agent: **Analysis → Planner → Evaluation**. Training chạy
trong worker; LLM đọc logs, lập thử nghiệm và so sánh kết quả control/trial.
Protocol v3 để Planner tự chọn giá trị cấu hình và số epoch kiểm chứng, không có
menu trial hoặc danh sách hyperparameter định sẵn trong YAML.

## Chạy nhanh trên máy đã chuẩn bị

Chạy từ thư mục project trên ổ D, có `.venv` và runtime vLLM đã được qualification.
Nếu chưa có môi trường, xem phần chuẩn bị bên dưới.

Terminal 1 — bật server, chờ script báo ready:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_vllm.ps1
```

Terminal 2 — chạy smoke 8 epoch:

```powershell
.\.venv\Scripts\python.exe -m mas_demo.cli run --config configs/autonomous_smoke.yaml --output "artifacts/smoke-$(Get-Date -Format yyyyMMdd-HHmmss)"
```

Terminal 3 — xem dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

Mở `http://localhost:8501` và chọn run vừa tạo. Dashboard chỉ đọc artifacts.
Mỗi output directory dành cho một run; luôn dùng tên mới khi chạy lại.

Backend local mặc định là **vLLM**, model **Qwen/Qwen3.5-0.8B**, API tại
`http://127.0.0.1:8000/v1`. Server không có giao diện chat. Training kết thúc
không tự tắt server; dừng bằng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/stop_vllm.ps1
```

## Phân công và luồng quyết định

1. Worker huấn luyện một epoch; collector gom tối đa `window` epoch gần nhất.
2. **Analysis quyết định bất thường từ training logs**: xu hướng loss, accuracy,
   gradient và quan sát theo epoch. Config/data chỉ hỗ trợ giải thích nguyên
   nhân có thể xảy ra, không tự quyết định status. `DIAGNOSED` là giả thuyết có
   evidence cần kiểm chứng; `ABSTAIN` là logs bình thường hoặc chưa đủ căn cứ.
3. Planner nhận diagnosis, cấu hình đang active, capabilities, lịch sử và tài
   nguyên còn lại. Nó tự chọn từ 0 đến `policy.max_trials` trial. **Giá trị 3 là
   trần, không phải số trial bắt buộc.**
4. Mỗi proposal là patch tối thiểu: `changes` chỉ chứa giá trị thực sự khác
   `current_config`. Planner chọn một `verification_epochs` chung cho control
   và mọi trial, không vượt số epoch còn lại.
5. Compiler kiểm cấu hình và admission reserve toàn bộ ngân sách trước khi chạy.
   Plan bị từ chối được gửi lại cho Planner sửa tối đa một lần. Code không tự bỏ
   trial, rút horizon hoặc tạo giá trị thay cho LLM.
6. Control/trial bắt đầu từ cùng checkpoint và chạy cùng số epoch. Evaluation
   chỉ so snapshot tại đúng comparison epoch; nhánh chưa đủ horizon không được promote.
7. Nhánh được chọn tiếp tục với config, optimizer/scheduler/RNG và revision/hash
   tương ứng. Analysis được gọi lại sau epoch main tiếp theo.

Analysis được gọi từ epoch 1 dù chưa đủ window, kể cả epoch cuối. Không mở trial
sau giới hạn epoch. Planner abstain không thay đổi diagnosis của Analysis.
Không có detector bất thường theo ngưỡng cố định hoặc Python tùy ý do model sinh.

## Chuẩn bị môi trường

### Windows training và biến môi trường

Cần Python 3.10 trở lên. Cài môi trường training:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Chỉnh `.env` theo máy; file này được Git bỏ qua. Biến process có sẵn ưu tiên hơn
`.env`. `model_id` nằm trong YAML và phải khớp model server; `LOCAL_LLM_MODEL`
chỉ phục vụ launcher Transformers cũ.

| Biến | Mặc định / mục đích |
| --- | --- |
| `LLM_LOCAL_BACKEND` | `vllm`; dùng `bundled` cho Transformers cũ. |
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1`. |
| `LLM_RUNTIME_MANIFEST` | `tmp/vllm/runtime-manifest.json`, bắt buộc xác minh runtime vLLM. |
| `LLM_TIMEOUT_SECONDS` / `LLM_DEADLINE_SECONDS` | 120 / 130 giây. |
| `LLM_MAX_OUTPUT_TOKENS` | 1024; nằm chung context budget với prompt. |
| `LLM_API_KEY` | Tùy chọn khi endpoint ngoài yêu cầu xác thực. |
| `OPENROUTER_API_KEY` | Cần khi `analysis_mode: openrouter`. |
| `HF_TOKEN` | Tùy chọn cho checkpoint private/gated; model public không yêu cầu. |
| `LOCAL_LLM_DEVICE`, `LOCAL_LLM_MODEL_CACHE` | Thiết bị/cache của launcher cũ. |
| `LOCAL_LLM_MODEL`, `LOCAL_LLM_REVISION`, `HF_HOME` | Model/revision/cache của launcher cũ; tách khỏi lựa chọn Agent trong YAML. |

### vLLM trong WSL2 trên ổ D

Launcher hiện gắn với môi trường Windows/WSL2 đã kiểm chứng: distro `MAS-vLLM`,
VHDX tại `D:\HK261\wsl\mas-vllm`, Linux Python 3.12 tại `/opt/mas-vllm/venv`.
Cache/checkpoint/log Linux nằm dưới `/opt/mas-vllm` trong VHDX trên D. Host cần
NVIDIA GPU dùng được từ WSL2, compute capability ít nhất 7.5.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_vllm_host.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_vllm.ps1
```

Host preflight mặc định kiểm distro `Ubuntu` có sẵn; sau setup có thể chọn
`-Distro MAS-vLLM -MinimumDFreeGiB 10` để kiểm distro dành riêng. Setup cần C
còn ít nhất 4 GiB, D ít nhất 35 GiB trước cài và 10 GiB sau cài. Nó tạo/cập nhật
môi trường riêng, không tự tải checkpoint hoặc khởi động API. Triton JIT còn cần
`build-essential` và `python3.12-dev` trong distro.

`requirements-vllm.in` là input của setup; setup resolve ra lock môi trường
Linux. `requirements-vllm.lock` ghi version/hash của môi trường đã qualification,
được launcher kiểm SHA256; **không cài file này vào Windows `.venv`**. Setup có
fallback resolve 0.20.0, nhưng môi trường fallback chưa tự được xem là đạt profile
serving đã khóa; launcher còn kiểm package/version thực tế.

`.gitattributes` giữ LF cho script `.sh` và file lock, kể cả checkout trên Windows
có bật `core.autocrlf`, để Bash chạy đúng và lock hash không đổi theo kiểu xuống dòng.

Profile serving đã khóa: Qwen3.5-0.8B revision
`2fc06364715b967f1860aea9cf38778875588b17`, FP16, không lượng tử hóa, context
4096, `max_num_seqs=1`, batch token 1024, GPU memory utilization 0.65, thinking
tắt và compact JSON qua XGrammar. `scripts/wsl/start_vllm.sh` dùng cache offline;
snapshot model phải được tải đầy đủ trước khi chạy probe.

Probe checkpoint chạy riêng cổng 8001, dùng output directory mới:

```powershell
wsl -d MAS-vLLM -u root --exec env HF_HOME=/opt/mas-vllm/cache/huggingface XDG_CACHE_HOME=/opt/mas-vllm/cache TMPDIR=/opt/mas-vllm/tmp /opt/mas-vllm/venv/bin/python /mnt/d/HK261/MAS/scripts/check_vllm_qwen.py --launch --gpu-memory-utilization 0.65 --output /opt/mas-vllm/probes/new-run-name
```

Probe ghi `probe.json`, snapshot hashes, responses, resources và server log;
khi kết thúc dừng nhóm process do nó tạo. Nạp được weights chưa đủ xác nhận API,
context, structured output hoặc khả năng hủy request hoạt động.

**Clone mới chưa có đủ dữ liệu để gọi start ngay.** Lifecycle cần các file local
`tmp/vllm/launch-profile.json` và `tmp/vllm/task2-qualification.json` từ môi trường
đã qualification, có lock hash và package metadata khớp. `qualified_runtime()`
có thể lấy qualification từ manifest cũ tương thích khi file riêng chưa tồn tại.
Setup/probe hiện chưa tự đưa toàn bộ dữ liệu này vào lifecycle launcher; chúng
không phải file mẫu được commit. Đây là workflow cho profile máy đã chuẩn bị,
chưa phải trình cài vLLM portable cho mọi máy.

Start chờ ready tối đa 600 giây rồi trả quyền điều khiển, để server chạy background.
Start/stop mặc định cổng 8000, hỗ trợ `-Port 8001` cho candidate. Stop kiểm ownership,
kernel birth identity, process group và Windows relay; không dừng process ngoài
nhóm đã xác minh. Restart bằng stop rồi start, chạy tuần tự.

### Transformers cũ và OpenRouter

Rollback Transformers dùng môi trường Windows riêng:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_local_llm.ps1
powershell -ExecutionPolicy Bypass -File scripts/stop_vllm.ps1
```

Đặt `LLM_LOCAL_BACKEND=bundled` và endpoint phù hợp trong `.env`; trong config run
mới đặt `model_id: Qwen/Qwen2.5-0.5B-Instruct`. Launcher cũ đọc
`LOCAL_LLM_MODEL`/revision trong `.env`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_local_llm.ps1
```

Hai server GPU chạy lần lượt. Trước khi phục hồi vLLM, dừng server cũ bằng
`scripts/stop_local_llm.ps1`, phục hồi `LLM_LOCAL_BACKEND=vllm`/model_id Qwen3.5
rồi gọi start_vllm. Dùng terminal mới hoặc cập nhật process env để tránh override
cấu hình `.env` vừa sửa. Không sửa model_id trong artifacts lịch sử.

OpenRouter dùng `analysis_mode: openrouter`, `model_id` tương thích endpoint và
`OPENROUTER_API_KEY` trong `.env`.

## Config, ngân sách và capability

| Config | Dữ liệu / giới hạn |
| --- | --- |
| `autonomous_smoke.yaml` | Synthetic 128/32, tối đa 8 epoch; kiểm tra luồng. |
| `autonomous_healthy.yaml` | CIFAR-10 subset 4.096/1.024, tối đa 20 epoch. |
| `autonomous_mixed.yaml` | Cùng subset, LR 0.1, momentum 0.99, decay 0.5; tối đa 100 epoch. |
| `autotrainer_oscillating_loss.yaml` | Blobs 700/300, MLP 2→50→3, SGD LR 0.01, momentum 0.9, batch 1; tối đa 200 epoch, worker budget 600 giây. |

Thay `--config` và dùng output mới, ví dụ:

```powershell
.\.venv\Scripts\python.exe -m mas_demo.cli run --config configs/autotrainer_oscillating_loss.yaml --output "artifacts/autotrainer-$(Get-Date -Format yyyyMMdd-HHmmss)"
```

AUTOTRAINER case là bản port PyTorch từ
[Oscillating Loss Case](https://github.com/shiningrain/AUTOTRAINER/tree/03a463ea05691a73403d7cff99a368fb4b6720cf/AutoTrainer/demo_case/Oscillating_Loss_Case),
dùng trọng số khởi tạo gốc; không bảo đảm tái tạo bitwise Keras. Blobs sinh tại
chỗ; CIFAR tải lần đầu hoặc tải trước bằng `scripts/download_cifar.py`.
Tên config không bảo đảm LLM phát hiện bất thường hoặc cải thiện training.

`training` là trạng thái ban đầu; `policy` là giới hạn run. Budget chỉ tính worker
wall time gồm startup/teardown; latency/tokens LLM và tổng wall time được báo riêng.
Run có thể hết budget trước max_epochs. Muốn kiểm một epoch không cần LLM:

```powershell
.\.venv\Scripts\python.exe -m mas_demo.cli profile --config configs/autotrainer_oscillating_loss.yaml --output "artifacts/profile-$(Get-Date -Format yyyyMMdd-HHmmss)"
```

Planner có adapter cho LR, momentum, decay, clipping, batch size, accumulation;
optimizer SGD/Adam/AdamW; scheduler none/StepLR/cosine và early stopping.
Validation kiểm kiểu dữ liệu, số hữu hạn, tùy chọn component và budget. Trial
không được sửa dataset/split/seed, metric, ngân sách hoặc max_epochs.

## Đọc logs và hiểu vì sao trial chưa chạy

`artifacts/<run>/` chứa `config.json`, `state.json`, config-revisions,
`events.jsonl`, memory SQLite, worker logs và checkpoint. Một proposal có trong
output LLM chưa có nghĩa đã được chạy:

| Event / mã lỗi | Ý nghĩa |
| --- | --- |
| `planning` | Raw/parsed proposal cùng metadata của Planner. |
| `plan_rejected` + `NO_OP` | Patch không đổi cấu hình hiện tại. |
| `DUPLICATE_CONFIG` | Nhiều proposal tạo cùng một cấu hình. |
| `INSUFFICIENT_BUDGET` | Không reserve được toàn bộ control/trial/continuation. |
| `trial_plan` | Plan đã qua compiler/admission; bắt đầu đường verification. |
| `decision` + `NOOP` | Vòng đó tiếp tục main, không có plan được nhận. |

Planner nhận lỗi để sửa một lần. Nếu vẫn không hợp lệ, hệ thống tiếp tục main;
không chạy lại nguyên cấu hình chỉ vì model đặt proposal_id mới. Xem các event
verification/evaluation và `rounds/<round>/admission.json` để kiểm trial thực tế.

`metadata.error` phân biệt fallback do lỗi với ABSTAIN hợp lệ. Analysis output
sai schema hoặc mâu thuẫn được repair một lần trên cùng evidence; ABSTAIN hợp lệ
không bị ép gọi lại. `metadata.calls` giữ từng phản hồi và chi phí tổng.

## Prompt, schema và context

Xem [README của prompts](src/mas_demo/prompts/README.md) để biết cách ghép từng
backend. Prompt được đọc lại mỗi request, không cần restart vLLM khi chỉ sửa text.

Schema sinh từ Python contracts; riêng vLLM được ràng buộc theo state trong
`output_schema.py`, gửi qua trường HTTP `response_format`. `response_format.txt`
chỉ là câu dẫn trước schema cho bundled/cloud, không phải nơi chứa schema vLLM.
Root reason local giới hạn 160 ký tự; Planner vLLM giới hạn patch.reason và
expected_effect 120 ký tự. Compiler/admission vẫn kiểm ngữ nghĩa sau generation.

vLLM preflight dùng `/tokenize`; bundled dùng `/v1/token-count`. Tổng prompt và
output reserve phải vừa context. Gateway có thể bỏ history cũ hoặc tham chiếu
output trước; giữ nguyên metrics/evidence và ghi `context_compaction`. Không còn
chỗ thì chặn generation. Sửa code server khác với sửa prompt: code server cần restart.

## Kiểm thử và đánh giá

Đặt thư mục tạm trên D vì các guard Windows yêu cầu dữ liệu kiểm tra ở ổ này:

```powershell
$env:TEMP = "$PWD\tmp\pytest"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
.\.venv\Scripts\python.exe -m pytest -q
```

Suite mặc định dùng HTTP stand-in và kiểm tra ownership/process; không tự chạy
benchmark model thật hoặc tạo distro. Test server Transformers có thể skip nếu
chưa cài dependency riêng; kiểm riêng bằng:

```powershell
.\tmp\llm-server-venv\Scripts\python.exe -m pytest tests/test_local_llm_server.py -q
```

Khi vLLM đang ready, kiểm ba role hoặc một nhóm Planner nhỏ:

```powershell
.\.venv\Scripts\python.exe scripts/check_local_agents.py --output "artifacts/agent-check-$(Get-Date -Format yyyyMMdd-HHmmss)"
.\.venv\Scripts\python.exe scripts/benchmark_local_agents.py --rounds 1 --cases tests/fixtures/planner_recovery_cases.json --case-ids p-recovery-unstable p-recovery-healthy p-recovery-budget --output "artifacts/planner-check-$(Get-Date -Format yyyyMMdd-HHmmss)"
```

Benchmark lưu warmup riêng, không chấm vào kết quả. Đọc `report.json` và raw
responses; benchmark CLI hiện không dùng exit code để báo mọi lỗi chất lượng.
`VALID` chấp nhận abstain hợp lệ, còn `NONEMPTY` đòi trial qua compiler/admission.
Fixture diagnosis kiểm Planner riêng, không chứng minh Analysis đúng. Nhóm case
nhỏ không thể làm full quality gate thành true.

`planner_outcomes` báo số trial/abstain hợp lệ, NO_OP, budget rejection và output
bị cắt. Model vẫn có thể chẩn đoán sai, tạo trial trùng hoặc không khả thi; JSON
hợp lệ chưa chứng minh cải thiện training. Kiểm tra trước/sau prompt đã có ca sinh
patch mới qua admission, nhưng ca budget thấp còn bị từ chối vì proposal trùng.

`compare_local_training.py` phục vụ paired pilot; chạy riêng với benchmark
inference để tránh tranh tài nguyên. Unit tests/mock pass chỉ xác nhận cơ chế,
không thay cho đo chất lượng ML. Script `check_vllm_simulation.py` là công cụ
Postman tùy chọn, cần CLI và tài nguyên local `postman/` vốn được Git bỏ qua;
nó không phải điều kiện để chạy training hoặc pytest thông thường.

## Stop, resume và migration

```powershell
.\.venv\Scripts\python.exe -m mas_demo.cli stop artifacts/<run>
.\.venv\Scripts\python.exe -m mas_demo.cli resume artifacts/<run>
.\.venv\Scripts\python.exe -m mas_demo.cli migrate-config artifacts/<old-run>/config.json --output tmp/migrated-v3.yaml
```

Thay `<run>`/`<old-run>` bằng tên thật. Stop chờ epoch/verification boundary;
run đã hoàn thành resume là no-op. Crash chưa ghi đủ chi phí trả INTERRUPTED,
không giả định chi phí bằng 0 rồi tự chạy lại. Artifact v1/v2 vẫn xem được trên
dashboard nhưng không resume bằng v3; migration chỉ chuyển initial config/policy,
không đoán lại trạng thái training.

`.env`, `tmp/`, `artifacts/`, dữ liệu, tài liệu cá nhân và `docs/`/`postman/` local
được Git bỏ qua. Không cần các báo cáo local đó để đọc hướng dẫn trong README này.
