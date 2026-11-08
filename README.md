# MAS training demo

Demo local cho đồ án: **training → phát hiện → thử sửa → đánh giá → tiếp tục**.
CIFAR-10 + CNN nhỏ, một training worker tại một thời điểm, phù hợp mục tiêu GPU 4GB.
Không cần Slurm để chạy demo đầu. Interface backend nằm trong `execution/base.py`.

## Chạy trên Windows / PowerShell

Môi trường `.venv` được tạo trong project. Nếu cài lại trên máy khác:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

Chạy thử CIFAR-10 thật (lần đầu tải dataset, sau đó dùng bản local):

```powershell
.venv/Scripts/python -m mas_demo.cli run --config configs/high_lr.yaml --output artifacts/demo
```

Mở terminal thứ hai trong project để xem dashboard:

```powershell
.venv/Scripts/python -m streamlit run app/dashboard.py
```

Mở `http://localhost:8501`. Chọn run `demo`; dashboard tự cập nhật.
Mỗi output directory chỉ dành cho một run, tránh ghi đè. Đổi output hoặc dùng `resume`.

## Các chế độ

- `configs/high_lr.yaml`: CIFAR-10 subset cố định, LR không phù hợp `0.1` được đặt ngay từ đầu, có hiệu lực từ batch đầu tiên của epoch 1. Metrics đến từ training thật; action không được bảo đảm luôn thắng.
- `configs/healthy.yaml`: không inject fault; detector vẫn có thể phát hiện plateau tự nhiên.
- `configs/low_budget.yaml`: budget quá nhỏ, không cho bắt đầu segment thiếu dự phòng.
- `configs/smoke.yaml`: synthetic data/CPU để thử plumbing nhanh, LR gây lỗi `1.0` được đặt ngay từ đầu. Được gắn nhãn synthetic, không phải bằng chứng nghiên cứu.
- `configs/high_momentum.yaml`: momentum `0.99` từ đầu; cho phép thử `0.9` hoặc `0.5`.
- `configs/excessive_weight_decay.yaml`: weight decay `0.5` từ đầu; cho phép thử `0.001` hoặc `0.0001`.
- `configs/over_clipping.yaml`: clipping `0.01` từ đầu; cho phép thử ngưỡng `1.0` hoặc `5.0`.
- `configs/mixed_faults.yaml`: LR `0.1`, momentum `0.99` và weight decay `0.5` từ đầu; mỗi trial sửa một thông số, có thể sửa tiếp qua các vòng sau.

Các cấu hình mới là tình huống minh họa, không bảo đảm luôn phát hiện được lỗi hoặc trial luôn thắng.
Xem [hướng dẫn thí nghiệm nhiều thông số](docs/multi-parameter-experiments.md).

```powershell
.venv/Scripts/python -m mas_demo.cli profile --config configs/healthy.yaml --output artifacts/profile
.venv/Scripts/python -m mas_demo.cli run --config configs/smoke.yaml --output artifacts/smoke
.venv/Scripts/python -m mas_demo.cli stop artifacts/demo
.venv/Scripts/python -m mas_demo.cli resume artifacts/demo
.venv/Scripts/python -m pytest -q
```

`stop` yêu cầu dừng tại ranh giới segment/verification round. Resume giữ nguyên config/budget/checkpoint.
Nếu máy chủ CIFAR gốc tải chậm, chạy `.venv/Scripts/python scripts/download_cifar.py` trước
`prepare`/`run`: script dùng range requests tới chính máy chủ gốc và kiểm tra MD5 archive.
Run hoàn thành được resume như no-op. Nếu tiến trình bị kill trong lúc có công việc chưa commit,
hệ thống fail closed thành INTERRUPTED: phải đối chiếu kết quả/cost trước khi tạo continuation;
không tự chạy lại và giả định chi phí bằng 0.

## Dùng LLM thật

**Analysis, Planner và Evaluation chỉ dùng API LLM**, mỗi agent có request và JSON Schema riêng.
Tất cả cấu hình mặc định dùng OpenRouter; đã bỏ luồng heuristic và các hàm tự xếp hạng/chọn nhánh.
Detector, budget admission và kiểm tra tính hợp lệ của output vẫn do code thực hiện.

Điền `OPENROUTER_API_KEY` vào `.env` ở thư mục project (đã có `.env.example` làm mẫu).
CLI tự đọc file này; biến môi trường đã có trong terminal được ưu tiên.

```powershell
.venv/Scripts/python -m mas_demo.cli run --config configs/high_lr.yaml --output artifacts/demo-llm
```

Đổi `model_id` trong YAML để chọn model cho cả ba agent. Gateway validate JSON/schema;
code kiểm tra evidence refs, LR allowlist, cooldown và điều kiện accuracy/loss.
Analysis lỗi trả ABSTAIN; Planner lỗi bỏ qua trials; Evaluation lỗi trả INCONCLUSIVE
và giữ control đã hoàn thành. Không có chế độ heuristic để fallback.
Gọi Analysis ở mỗi cửa sổ quan sát đủ dữ liệu, kể cả chưa có tín hiệu bất thường từ detector;
LLM có thể ABSTAIN. Gọi Planner khi có diagnosis hợp lệ và còn horizon;
gọi Evaluation sau verification. Khi gateway retry có thể có thêm một HTTP attempt.
Output và metadata (model, usage, latency, attempts, error) nằm trong events.jsonl và dashboard.
CLI yêu cầu key trước khi run/resume, kể cả smoke CPU. `prepare`/`profile` chỉ chuẩn bị/đo training,
không gọi agent nên không cần key. `--analysis openrouter` vẫn được chấp nhận cho lệnh cũ;
`--analysis heuristic` và config heuristic bị từ chối. Run heuristic lịch sử vẫn xem được trên
dashboard nhưng không resume bằng phiên bản này; hãy tạo run mới. Không commit key hoặc `.env`.
Xem [schema output của từng agent](docs/llm-agent-contracts.md) cho ví dụ JSON và ràng buộc.

## Luồng và tính công bằng

1. Preprocessing cố định split/index/version; không dùng test set để chọn action.
2. Main train đến điểm quan sát; collector tổng hợp loss, accuracy, gradient và thông số thực tế, đồng thời phát hiện loss spike/plateau/oscillation.
3. Analysis hình thành hypothesis và can thiệp từ allowlist; Planner chọn và xếp thứ tự trials theo output LLM, history và cooldown.
4. Controller giữ budget cho control/trials/continuation. Budget thiếu thì giảm số trial hoặc
   horizon chung trước khi chạy; action không được thực thi nếu chưa reserve.
5. Control và trials khôi phục cùng model/optimizer/scheduler/RNG/data order, chạy cùng horizon.
   Local chạy tuần tự, đóng worker để nhả GPU trước nhánh kế tiếp.
6. Evaluation đọc đúng comparison epoch, chọn trial vượt minimum gain và loss guardrail,
   hoặc giữ main. Nhánh chưa hoàn thành không được xem là đã cải thiện.
7. Lưu outcome/memory theo từng can thiệp; tiếp tục từ checkpoint nhánh đã chọn, giữ cả momentum, weight decay và clipping đã áp dụng, rồi kiểm tra budget lại.

Số epoch, trial, horizon, thresholds và budget là config minh họa cần profile trước benchmark.
Config demo hiện đặt LR0.1 ngay từ đầu, không chờ epoch 3 mới đổi LR.
Các run cũ có fault theo epoch là dữ liệu lịch sử; cần tạo output mới để chạy cấu hình mới.
Mức1.0 trong pilot gây numerical error nên đã được giữ
trong artifacts như một lượt thất bại, không dùng để chứng minh sửa lỗi thành công.
`profile.json` đo thời gian/VRAM thật để chỉnh `epoch_seconds_estimate`.
Cost trong bản demo là **cận trên allocated-resource time bằng worker wall time**, gồm startup,
load dữ liệu và teardown; không phải NVML active GPU compute time. CPU run chỉ báo CPU/resource
seconds, không đổi thành GPU-hours. Watchdog/OS shutdown có thể làm vượt cap nhỏ, được báo `overrun`.
Thời gian LLM ở parent không giữ CUDA allocation vì worker đã đóng; token/latency báo riêng.

## Artifacts và giới hạn

`artifacts/<run>/`: config/manifest, state, audit JSONL, SQLite memory, segments, round admissions,
trial results, full-state checkpoints và worker logs. Checkpoint pickle chỉ load từ artifact local
tin cậy. Dashboard chỉ đọc, không tạo worker hoặc thực thi action.

Demo chưa chứng minh GPU savings hay lợi thế MAS so với fixed rule. Benchmark/ablation, Slurm thật,
AMP và telemetry NVML chi tiết thuộc giai đoạn sau. CIFAR subset không được gộp với full dataset.
Xem [runbook](docs/demo_runbook.md) và [implementation ledger](docs/implementation-progress.md).

Dependency CUDA theo [hướng dẫn phiên bản PyTorch chính thức](https://pytorch.org/get-started/previous-versions/).
Gateway theo [structured outputs của OpenRouter](https://openrouter.ai/docs/guides/features/structured-outputs).
