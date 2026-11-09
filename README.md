# MAS — autonomous training trials

Demo local: training → Analysis → Planner → control/trials → Evaluation → tiếp tục.

Protocol v3 bỏ danh sách trial gợi ý trong YAML. Analysis tự nhận biết bất thường từ
metrics; Planner tự sinh giá trị config mới, số lượng trial và một số epoch chung
cho mọi trial cùng nhánh control trong lượt đó.

## Chuẩn bị trên Windows / PowerShell

Cần Python 3.10 trở lên. Chạy các lệnh bên dưới từ thư mục project.
Nếu máy đã có môi trường MAS và Qwen, có thể bỏ qua bước cài đặt.

Tạo `.env` nếu chưa có, rồi chỉnh giá trị theo máy của bạn:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Các biến chính:

| Biến | Ý nghĩa |
| --- | --- |
| `LLM_BASE_URL` | Endpoint cho agent; local mặc định `http://127.0.0.1:8000/v1`. |
| `LOCAL_LLM_DEVICE` | Thiết bị chạy Qwen: `cuda` hoặc `cpu`; mẫu mặc định `cuda`. |
| `LOCAL_LLM_MODEL_CACHE` | Đường dẫn cache riêng của máy; để trống dùng `tmp/model-cache`. |
| `LOCAL_LLM_MODEL`, `LOCAL_LLM_REVISION` | Model và revision được Qwen server nạp. |
| `LLM_TIMEOUT_SECONDS`, `LLM_DEADLINE_SECONDS` | Giới hạn chờ request và tổng thời gian gọi agent. |
| `LLM_MAX_OUTPUT_TOKENS` | Giới hạn output của agent local. |
| `OPENROUTER_API_KEY` | Chỉ cần khi dùng OpenRouter. |
| `LLM_API_KEY` | Chỉ cần khi endpoint ngoài yêu cầu xác thực. Server Qwen đi kèm không kiểm tra key này. |
| `HF_TOKEN` | Tùy chọn cho model Hugging Face private/gated; Qwen public không cần. |

`.env` chứa key, endpoint riêng và đường dẫn của máy, được Git bỏ qua.
`.env.example` chỉ chứa giá trị mẫu. CLI và script setup/start Qwen đọc `.env`;
biến môi trường có sẵn được ưu tiên. Tham số `-Device` và `-ModelCache` có thể
ghi đè lựa chọn mặc định của launcher; `HF_HOME` là override riêng cho Hugging Face.
Model agent được chọn bằng `model_id` trong YAML và phải khớp model server đang phục vụ.

Cài môi trường training:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

Qwen local dùng môi trường server riêng:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_local_llm.ps1
```

## Chạy AUTOTRAINER Oscillating Loss

Terminal 1 — bật Qwen và giữ terminal mở:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_local_llm.ps1
```

Launcher đọc thiết bị từ `.env`. Muốn dùng CPU cho Qwen, thêm `-Device cpu`.
Đợi log `Application startup complete` rồi chạy training.
API mặc định tại http://127.0.0.1:8000/v1; server chỉ bind loopback và không có giao diện chat.

Terminal 2 — chạy case AUTOTRAINER:

```powershell
.\.venv\Scripts\python.exe -m mas_demo.cli run --config configs/autotrainer_oscillating_loss.yaml --output artifacts/autotrainer-oscillating-v1
```

Config này là bản port PyTorch của `Oscillating_Loss_Case/config.pkl` trong
[AUTOTRAINER](https://github.com/shiningrain/AUTOTRAINER/tree/03a463ea05691a73403d7cff99a368fb4b6720cf/AutoTrainer/demo_case/Oscillating_Loss_Case), không cần TensorFlow/Keras:

- Blobs 1.000 mẫu, 2 đặc trưng, 3 lớp; 700 mẫu đầu train, 300 mẫu cuối validation.
- MLP 2 → 50 ReLU → 3, gồm 303 tham số, dùng trọng số khởi tạo gốc.
- SGD `lr=0.01`, momentum `0.9`, batch size `1`; training chạy CPU.
- Analysis sau mỗi epoch, window `5`. LLM tự đánh giá bất thường và đề xuất trial.
- Config hiện tại giới hạn **200 epoch** và **600 giây thời gian worker**.

Run có thể dừng vì hết budget trước epoch 200. Muốn demo ngắn, đặt
`policy.max_epochs: 20`; muốn chạy dài, tăng `policy.budget_seconds` theo kết quả profile.
Thời gian LLM chưa nằm trong budget worker nên tổng thời gian thực tế có thể dài hơn.
Đây là bản port, không tái tạo bitwise Keras vì RNG, shuffle và optimizer có thể khác.
Tên case không bảo đảm LLM sẽ phát hiện hoặc sửa được bất thường.

Mỗi output directory chỉ dành cho một run. Nếu `autotrainer-oscillating-v1` đã tồn tại,
đổi tên output thành `autotrainer-oscillating-v2` khi chạy mới.

Chỉ kiểm tra một epoch qua worker, không cần bật Qwen:

```powershell
.\.venv\Scripts\python.exe -m mas_demo.cli profile --config configs/autotrainer_oscillating_loss.yaml --output artifacts/autotrainer-oscillating-profile-v1
```

Kết quả đo nằm trong `artifacts/autotrainer-oscillating-profile-v1/profile.json`.

## Xem kết quả

Terminal 3 — bật dashboard:

```powershell
.venv/Scripts/python -m streamlit run app/dashboard.py
```

Mở http://localhost:8501 và chọn run mới. Dashboard chỉ đọc artifacts.
Kết quả AUTOTRAINER nằm trong `artifacts/autotrainer-oscillating-v1/`.

## Các config hiện có

| File | Dữ liệu và trạng thái ban đầu |
| --- | --- |
| `autotrainer_oscillating_loss.yaml` | Blobs 700/300; MLP gốc; SGD LR 0.01, momentum 0.9, batch 1; tối đa 200 epoch, budget worker 600 giây. |
| `autonomous_mixed.yaml` | CIFAR-10 subset 4.096/1.024; LR 0.1, momentum 0.99, decay 0.5; tối đa 100 epoch. |
| `autonomous_healthy.yaml` | Cùng subset; LR 0.01, momentum 0.9, decay 0; tối đa 20 epoch. |
| `autonomous_smoke.yaml` | Synthetic 128/32; kiểm tra luồng, không phải bằng chứng CIFAR. |

Chạy config khác bằng cách thay `--config` và chọn output mới:

```powershell
.venv/Scripts/python -m mas_demo.cli run --config configs/autonomous_mixed.yaml --output artifacts/autonomous-mixed-v3
.venv/Scripts/python -m mas_demo.cli run --config configs/autonomous_healthy.yaml --output artifacts/autonomous-healthy-v3
.venv/Scripts/python -m mas_demo.cli run --config configs/autonomous_smoke.yaml --output artifacts/autonomous-smoke-v3
```

CIFAR-10 được tải lần đầu; có thể tải trước bằng
`.venv/Scripts/python scripts/download_cifar.py`. Blobs được sinh tại chỗ, không cần tải dataset.

`training` là trạng thái khởi đầu; `policy` là giới hạn run. Không có
`allowed_interventions`, `lr_factors`, `trial_horizon` hoặc menu candidate.

`policy.max_trials: 3` là trần; LLM được sinh ít hơn hoặc abstain.
`policy.budget_seconds` tính thời gian worker gồm startup/teardown. LLM latency,
tokens và tổng wall time được báo riêng; đây chưa phải cap cho toàn bộ GPU LLM.

## Luồng từng epoch và quyền quyết định

1. Main train một epoch. Collector chỉ gom tối đa năm epoch gần nhất.
2. Analysis dùng kiến thức ML để trả DIAGNOSED hoặc ABSTAIN; không có detector theo ngưỡng.
3. DIAGNOSED chỉ cần giả thuyết có evidence; không cần chọn intervention.
4. Planner nhận config đang active, evidence, capability, lịch sử và tài nguyên còn lại.
5. Planner tự sinh các patch tuyệt đối, có thể sửa nhiều field, và một
   `verification_epochs` chung cho cả lượt.
6. Code kiểm tra plan và reserve toàn bộ budget trước khi chạy. Kế hoạch không hợp lệ
   hoặc thiếu budget được phản hồi cho Planner tối đa một lần. Code không tự bỏ trial
   hoặc rút ngắn horizon.
7. Control và tất cả trials bắt đầu từ cùng checkpoint, chạy cùng số epoch.
   Evaluation chỉ so metrics tại đúng comparison epoch.
8. Nhánh thắng được tiếp tục cùng config, optimizer/scheduler/RNG và revision/hash.
   Sau epoch main tiếp theo, Analysis tiếp tục nhận biết bất thường.

Analysis được gọi ngay từ epoch 1 dù chưa đầy window. Epoch cuối vẫn được phân tích;
không mở trial sau giới hạn epoch. Trial chưa chạy đủ horizon không đủ điều kiện promote.

## Các capability đã có adapter

Planner có thể tự chọn giá trị hợp lệ cho LR, momentum, decay, clipping, batch size,
gradient accumulation; đổi optimizer SGD/Adam/AdamW và scheduler none/StepLR/cosine.
Các options như betas/epsilon chỉ áp dụng cho optimizer tương ứng. Early stopping
được thực thi ở epoch boundary; nhánh dừng sớm không được so với snapshot khác epoch.

Giới hạn type, số hữu hạn, tương thích component và budget là validation thực thi,
không phải bộ rule nhận biết bất thường. LLM không được sửa dataset/split/seed, metric,
ngân sách hoặc max_epochs qua trial, và không được sinh/chạy Python tùy ý.

## Prompt và audit

Prompt nằm trong `src/mas_demo/prompts/`:

- `analysis.txt`: nhận biết và giải thích bất thường.
- `planner.txt`: tự sinh patch và horizon chung.
- `evaluation.txt`: so sánh kết quả thực nghiệm.
- `local.txt`: hướng dẫn ngắn cho Qwen.
- `response_format.txt`: JSON output.

File được đọc lại mỗi request. Schema sinh từ contracts Python. Sửa prompt không tự đổi
validator hoặc capability. Sau thay đổi protocol/schema, phải tạo run v3 mới.

`artifacts/<run>/` chứa state/config, config-revisions, events.jsonl, memory.sqlite,
worker requests/logs và checkpoint. Raw output/usage/error nằm trong metadata từng agent.
ABSTAIN do model và ABSTAIN dự phòng vì lỗi được phân biệt bằng `metadata.error`.

Nếu Analysis trả output sai schema hoặc tự mâu thuẫn (ví dụ `ABSTAIN` kèm
hypothesis), hệ thống gửi lỗi validation để LLM tự sửa tối đa một lần trên cùng
evidence. Code không tự đổi status hay tạo giả thuyết. ABSTAIN hợp lệ không bị gọi
lại; lỗi kết nối/xác thực giữ chính sách gateway, không kích hoạt lượt sửa output.
Nếu sửa vẫn thất bại, dashboard hiển thị lỗi và ABSTAIN chỉ là kết quả dự phòng,
không phải kết luận training bình thường. `metadata.calls` giữ output/error của
từng lần; latency, token usage và attempts ở metadata ngoài cùng tính tổng cả hai lần.

Server local có endpoint `/v1/token-count` dùng đúng tokenizer để kiểm tra context trước
suy luận. Khi cần, client bỏ history cũ hoặc dùng tham chiếu output sửa plan đã lưu;
metrics/evidence được giữ nguyên. Các lần rút gọn được ghi ở `context_compaction`.
Khởi động lại Qwen sau khi sửa code server. Endpoint ngoài không hỗ trợ preflight
vẫn có thể dùng nhưng phải tự cung cấp kiểm tra context của nó.

## Dừng, resume, profile và migration

```powershell
.venv/Scripts/python -m mas_demo.cli stop artifacts/autotrainer-oscillating-v1
.venv/Scripts/python -m mas_demo.cli resume artifacts/autotrainer-oscillating-v1
.venv/Scripts/python -m mas_demo.cli migrate-config artifacts/old-run/config.json --output tmp/migrated-v3.yaml
```

Stop chờ epoch/verification boundary. Run hoàn thành resume là no-op. Crash giữa công
việc chưa ghi đủ chi phí trả INTERRUPTED, không tự chạy lại và giả định cost bằng 0.

Artifact v1/v2 vẫn xem được trên dashboard, nhưng không resume bằng v3. Migration chỉ
chuyển initial settings/policy; bỏ action menus/horizon cũ, không đoán training state.
Nếu config cũ dùng fault theo epoch, phải chuyển rõ thành initial settings trước.

Dừng training không tắt Qwen. Dùng Ctrl+C trong terminal server, hoặc:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stop_local_llm.ps1
```

## Kiểm thử và đánh giá model

```powershell
.venv/Scripts/python -m pytest -q
tmp/llm-server-venv/Scripts/python -m pytest tests/test_local_llm_server.py -q
.venv/Scripts/python scripts/benchmark_config_agent.py --rounds 1 --output artifacts/config-benchmark-v3
.venv/Scripts/python scripts/compare_local_training.py --config configs/autonomous_mixed.yaml --output artifacts/config-pilot-v3
```

Chạy benchmark inference riêng với training để tránh tranh tài nguyên. Bộ tình huống
do tác giả đặt kỳ vọng; không phải ground truth nguyên nhân. Unit tests/mock pass chỉ
chứng minh cơ chế, không bảo đảm Qwen 0.5B phát hiện đúng hoặc sinh trial hữu ích.
Báo cáo raw schema, semantic validation và chất lượng quyết định riêng; không ép
DIAGNOSED, không biến lỗi thành trial giả để làm kết quả trông tốt hơn.

Dùng OpenRouter bằng cách đặt `analysis_mode: openrouter`, chọn `model_id` phù hợp
với endpoint và điền OPENROUTER_API_KEY trong .env. Không commit key hoặc .env.
