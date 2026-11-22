# Agent prompts

Các file được đọc mới mỗi request qua `load_prompt()`. Sửa text có hiệu lực ở
lượt gọi Agent tiếp theo, kể cả run đang hoạt động; không cần restart server.
Schema/validator và khả năng thực thi vẫn nằm trong Python.

Định dạng các file prompt `.txt`: mỗi đoạn nằm trên một dòng vật lý, không xuống
dòng cứng giữa đoạn; dùng dòng trống để ngăn các đoạn.

## Vai trò

| File | Trách nhiệm |
| --- | --- |
| `analysis.txt` | Prompt một bước cho adapter legacy `gateway.complete`; không dùng trong pipeline hai bước. |
| `analysis_detection.txt` | Đọc nguyên bản ghi epoch và viện dẫn bằng epoch; không nhận current_config riêng. |
| `analysis_hypotheses.txt` | Nhận detection đã khóa và config thực tế; sinh từ một đến ba cơ chế nguyên nhân nghi ngờ khác nhau. |
| `local_vllm_analysis.txt` | Wrapper JSON riêng cho hai bước Analysis, không chứa chỉ dẫn Planner/Evaluation. |
| `planner.txt` | Nhận diagnosis có rank, tạo đúng một trial cho mỗi hypothesis, sinh sparse patch và chọn horizon; controller quyết định ngân sách. |
| `evaluation.txt` | So control/trial sau khi chạy tại comparison epoch, chọn kết quả phù hợp. |
| `local.txt` | Hướng dẫn ngắn dùng chung cho Analysis/Evaluation vLLM. |
| `local_vllm.txt` | JSON gọn và chỉ dẫn bổ sung cho Analysis/Evaluation vLLM. |
| `local_vllm_planner.txt` | JSON gọn riêng Planner, mẫu proposal/abstain và giới hạn văn bản. |

Analysis trả từ một đến ba hypothesis về các nguyên nhân nghi ngờ khác nhau khi DIAGNOSED;
ABSTAIN trả danh sách rỗng. Mỗi hypothesis có ID riêng, rank 1–N duy nhất và
viện dẫn epoch có thật. Rank 1 được kiểm chứng trước, độc lập với confidence.
Analysis chọn status theo diễn biến logs. Config đáng ngờ không tự chứng minh
training bất thường. Planner dùng diagnosis để lập phép thử, không chẩn đoán lại;
quyết định ngân sách thuộc controller. `max_trials=3` là trần số trial thực thi.
Kế hoạch có trial phải bao phủ đúng một trial riêng cho mỗi hypothesis: N
hypothesis tạo N trial, với N cấu hình khác nhau. Controller tính thời gian,
xét rank/cooldown/policy và ngân sách rồi chọn 0–3 trial. Control/phần tiếp tục
chỉ tính một lần cho nhóm được chọn; không đổi patch hoặc horizon của Planner.
Planner nhận status và đầy đủ hypotheses (ID, rank, cause, mechanism, config_claims, confidence, evidence_epochs),
không nhận số dư ngân sách hoặc trần thực thi để tự loại proposal.
Không lặp lại mode/reason dài của Analysis trong payload Planner; bản gốc vẫn
có trong logs Analysis. Lý do abstain rỗng hoặc chỉ là placeholder bị từ chối
với INVALID_ABSTENTION_REASON để Planner sửa lại.
`changes` chỉ ghi field thực sự đổi, field bỏ qua tự giữ nguyên; đổi ID hoặc lời
giải thích không biến một config cũ thành trial mới.

Analysis nhận nguyên các bản ghi `EpochEvent` trong `evidence.recent_events`,
giữ đầy đủ field từ worker, kể cả null, thông số, thời gian, số mẫu và số bước
optimizer. Không gửi bảng facts/trends tính sẵn hoặc yêu cầu viện dẫn mã SHA256.
Cả hai bước viện dẫn bằng `evidence_epochs`. Cửa sổ theo config `window`, mặc định
training là 5; không phải toàn bộ lịch sử run hay stdout của worker. Wrapper ghi
file JSONL (time/kind/schema_version) không thuộc bản ghi EpochEvent được gửi.
Helper facts/trends và schema cũ vẫn giữ để đọc/kiểm thử giao thức legacy, không
được gọi để dựng đầu vào của luồng Analysis hiện tại.

## Ghép prompt theo role

| Role | System message |
| --- | --- |
| vLLM Planner | `planner.txt` + `local_vllm_planner.txt` |
| vLLM Analysis detection | `analysis_detection.txt` + `local_vllm_analysis.txt` |
| vLLM Analysis hypotheses | `analysis_hypotheses.txt` + `local_vllm_analysis.txt` |
| vLLM Analysis legacy (`gateway.complete`) | `analysis.txt` + `local.txt` + `local_vllm.txt` |
| vLLM Evaluation | `evaluation.txt` + `local.txt` + `local_vllm.txt` |

Gateway gửi schema qua trường HTTP `response_format` của vLLM.
Payload JSON được mã hóa gọn, giữ nguyên dữ liệu để giảm token đầu vào.
Expanded schema không lặp lại trong system message để tiết kiệm context; model
nhận quy tắc JSON trong wrapper. Wrapper Planner còn có mẫu cấu trúc ngắn.

## Schema và validation

- [contracts.py](../contracts.py) định nghĩa kiểu dữ liệu chung.
- [gateway.py](../llm/gateway.py) sinh schema và ghép request.
- [output_schema.py](../llm/output_schema.py) ràng buộc schema vLLM theo state:
  hash/hypothesis/epoch/branch, field đang hoạt động, abstain và shared horizon.
- Root reason của Analysis giới hạn 512 ký tự, prompt yêu cầu 2–3 câu ngắn.
  Root reason của Planner/Evaluation vẫn giới hạn 160 ký tự. Planner vLLM giới hạn
  patch.reason và expected_effect 120 ký tự; cause, expected_effect và reason của
  các vai trò khác vẫn có chỉ dẫn tối đa 15 từ trong prompt tương ứng.
- [configuration.py](../configuration.py) kiểm patch/config khác nhau thực sự;
  kiểm mỗi hypothesis có đúng một trial và gắn rank từ Analysis; DIAGNOSED hợp lệ
  bắt buộc có đúng một proposal cho mỗi hypothesis (1–3). Schema loại giá trị giữ nguyên khỏi sparse patch.
  Admission chọn
  nhóm trial vừa ngân sách theo rank, ghi rõ lý do bỏ qua từng trial.
  JSON đúng schema vẫn có thể bị HYPOTHESIS_COVERAGE, NO_OP hoặc duplicate.

Giá trị trial/horizon phải do model tự chọn theo đầu vào, không phải menu cố định.
Sửa prompt không tự đổi validator,
không bảo đảm model tuân thủ hay chứng minh chất lượng quyết định ML.

Xem [README chính](../../../../README.md) để chạy config, xem events và kiểm thử.

## Audit hai bước

Hai prompt Analysis và wrapper riêng được viết bằng tiếng Việt. Các trường văn bản
tự do được yêu cầu viết tiếng Việt; tên field JSON và giá trị quy ước giữ nguyên.

`diagnose()` gọi detection trước, chỉ gọi hypotheses khi DIAGNOSED. Mỗi bước được
repair output tối đa một lần. Hai call thành công bình thường không tính là repair.
Nếu bước nguyên nhân thất bại, diagnosis trả None; metadata.detection vẫn giữ
DIAGNOSED và outcome=hypothesis_generation_failed. Planner không được gọi.
ABSTAIN hợp lệ chỉ đến từ bước detection, không phải cách che lỗi sinh hypothesis.

Bước hypotheses giữ tối đa ba bản khác nhau sau chuẩn hóa cause/mechanism
(chữ hoa/thường, dấu câu, khoảng trắng). Khi trùng, giữ bản có rank ưu tiên cao hơn,
đánh số lại 1–N theo thứ tự ưu tiên rồi mới kiểm claim/evidence. Không thêm bản mới
cho đủ ba và không sửa claim sai để hợp thức hóa kết quả. metadata.calls[].output
giữ nguyên phản hồi model; selection ghi số đầu vào/giữ lại, bản bị bỏ và đổi rank;
selected_output ghi kết quả được đưa vào validator. Lọc chuỗi không chứng minh các
paraphrase khác nhau về mặt ngữ nghĩa.

Trạng thái phải dựa trên logs; telemetry không bị loại chỉ vì bốn metric loss/
accuracy/gradient thiếu. Epoch được viện dẫn phải có bản ghi worker thực tế.
Sau promote, history chỉ gồm các segment thuộc lineage đã chọn và giữ nguyên
branch_id worker, kể cả qua resume. Artifact cũ không được sửa ngược lại.
Config claims kiểm giá trị và component đang hoạt động, chưa chứng minh cause đúng.
Diagnosis giữ nguyên mechanism và config_claims cho Planner. Validator yêu cầu claim
cho tên field/alias cấu hình xuất hiện trong cause hoặc mechanism, và chặn một số
khẳng định thiếu/tắt trực tiếp bằng tiếng Việt/Anh khi worker đang áp dụng field đó,
cũng như khẳng định field đang hoạt động khi worker đã tắt nó. Các câu phủ định
thiếu vắng và mô tả giá trị chưa phù hợp không mặc nhiên đồng nghĩa field bị tắt.
Validator không phát hiện được mọi paraphrase hay khẳng định ngầm trong prose.
Replay phải chấm nội dung riêng, kể cả khi JSON và references đều hợp lệ.
Payload thực tế sau compact context nằm trong metadata.calls[].input_payload;
có thể đối chiếu trực tiếp evidence.recent_events với bản ghi worker.

Planner phải tạo patch tối thiểu theo từng mechanism; reason và expected_effect phải
khớp các field thực sự thay đổi. DUPLICATE_CONFIG trả duplicate_groups gồm config_hash,
proposal_ids, hypothesis_ids và patches của toàn bộ proposal trùng để sửa trong một lần.

Replay một round đã có Analysis và Planner mà không chạy training:

```powershell
python scripts/replay_round.py --run artifacts/<run-id> --round 4 --output artifacts/<new-replay-id>
```

Replay ghi inputs.json, responses.jsonl, report.json cùng provenance/runtime capture;
không sửa run gốc. Report kiểm output cũ bằng validator mới rồi gọi lại Analysis/Planner
trên cùng evidence/current_config. outcome=ready chỉ xác nhận đúng N trial qua compiler cho N hypothesis và
grounding không bị mất; manual_review_pending vẫn true cho đến khi đọc nội dung.
ABSTAIN hoặc lỗi Analysis/Planner trả exit code 2, không tính là replay thành công.

Nếu cần kiểm tra riêng Planner khi Analysis chưa tạo được đầu vào hợp lệ, truyền
`--hypotheses <reviewed-hypotheses.json>` với schema EpochLogHypothesesResponse.
Đầu vào này vẫn phải qua validator; report và manifest ghi mode=planner_only, toàn bộ
hypotheses đã cung cấp được lưu trong inputs.json. Kết quả đó không chứng minh
Analysis tự sinh giả thuyết thành công.
