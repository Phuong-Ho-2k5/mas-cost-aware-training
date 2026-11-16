# Agent prompts

Các file được đọc mới mỗi request qua `load_prompt()`. Sửa text có hiệu lực ở
lượt gọi Agent tiếp theo, kể cả run đang hoạt động; không cần restart server.
Schema/validator và khả năng thực thi vẫn nằm trong Python.

## Vai trò

| File | Trách nhiệm |
| --- | --- |
| `analysis.txt` | Quyết định DIAGNOSED/ABSTAIN từ training logs; config/data chỉ hỗ trợ giả thuyết nguyên nhân. |
| `planner.txt` | Nhận diagnosis, sinh sparse patch với giá trị mới, tự chọn số trial/horizon theo tài nguyên; hướng dẫn repair NO_OP/duplicate. |
| `evaluation.txt` | So control/trial sau khi chạy tại comparison epoch, chọn kết quả phù hợp. |
| `local.txt` | Hướng dẫn ngắn dùng chung cho backend local, ngoại trừ Planner vLLM. |
| `local_vllm.txt` | JSON gọn và chỉ dẫn bổ sung cho Analysis/Evaluation vLLM. |
| `local_vllm_planner.txt` | JSON gọn riêng Planner, mẫu proposal/abstain và giới hạn văn bản. |
| `response_format.txt` | Câu dẫn JSON trước schema cho bundled/cloud; không chứa schema cố định. |

Analysis chọn status theo diễn biến logs. Config đáng ngờ không tự chứng minh
training bất thường. Planner dùng diagnosis để lập phép thử, không chẩn đoán lại;
abstain tài nguyên không thay đổi kết luận của Analysis. `max_trials=3` là trần.
`changes` chỉ ghi field thực sự đổi, field bỏ qua tự giữ nguyên; đổi ID hoặc lời
giải thích không biến một config cũ thành trial mới.

## Ghép prompt theo backend

| Backend / role | System message |
| --- | --- |
| vLLM Planner | `planner.txt` + `local_vllm_planner.txt` |
| vLLM Analysis | `analysis.txt` + `local.txt` + `local_vllm.txt` |
| vLLM Evaluation | `evaluation.txt` + `local.txt` + `local_vllm.txt` |
| Bundled local | Prompt role + `local.txt` + `response_format.txt` + schema JSON |
| OpenRouter | Prompt role + `response_format.txt` + schema JSON |

Trường HTTP `response_format` vẫn được gửi ở mọi đường gateway. Riêng vLLM,
expanded schema không lặp lại trong system message để tiết kiệm context; model
nhận mẫu cấu trúc ngắn trong wrapper. Vì vậy file `response_format.txt` chỉ có
một câu dẫn là chủ ý, không phải thiếu contract.

## Schema và validation

- [contracts.py](../contracts.py) định nghĩa kiểu dữ liệu chung.
- [gateway.py](../llm/gateway.py) sinh schema và ghép request.
- [output_schema.py](../llm/output_schema.py) ràng buộc schema vLLM theo state:
  hash/hypothesis/epoch/branch, field đang hoạt động, abstain và shared horizon.
- Local root reason giới hạn 160 ký tự. Planner vLLM giới hạn patch.reason và
  expected_effect 120 ký tự; chỉ dẫn 15 từ trong prompt là yêu cầu bổ sung.
- [configuration.py](../configuration.py) kiểm patch/config khác nhau thực sự;
  admission kiểm budget. JSON đúng schema vẫn có thể bị NO_OP hoặc duplicate.

Ví dụ dùng placeholder để minh họa hình thức; giá trị trial/horizon phải do model
tự chọn theo đầu vào, không phải menu cố định. Sửa prompt không tự đổi validator,
không bảo đảm model tuân thủ hay chứng minh chất lượng quyết định ML.

Xem [README chính](../../../README.md) để chạy config, xem events và kiểm thử.
