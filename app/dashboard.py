from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MAS Training Demo", page_icon="📈", layout="wide")
st.title("MAS · Training intervention")
st.caption("Theo dõi → phát hiện → thử sửa → đánh giá → tiếp tục")
root = Path(__file__).resolve().parents[1]
default_runs = root / "artifacts"
folder = Path(st.sidebar.text_input("Thư mục artifacts", str(default_runs)))
runs = sorted([p.parent for p in folder.glob("*/state.json")], reverse=True)
if not runs:
    st.info("Chưa có run. Chạy: python -m mas_demo.cli run --config configs/high_lr.yaml")
    st.stop()
selected = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name)
st.sidebar.caption("Dashboard chỉ đọc artifacts. Training chạy bằng CLI ở terminal riêng.")


def read_events(path):
    result = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # writer may still be appending the last line
    return result


@st.fragment(run_every="2s")
def monitor():
    try:
        state = json.loads((selected/"state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        st.info("Đang chờ state...")
        return
    st.caption(f"Dataset: {state['origin']} · Agents: {state['analysis_mode']} · Resource: {state['resource_type']}")
    if state["analysis_mode"] == "heuristic":
        st.info("Run lịch sử dùng heuristic. Các run mới chỉ dùng API LLM; run này chỉ dùng để xem lại.")
    else:
        st.caption("Analysis, Planner và Evaluation đều gọi API LLM khi đến bước tương ứng.")
    if state["origin"] == "synthetic":
        st.warning("Synthetic smoke data — không phải kết quả CIFAR-10.")
    a, b, c, d = st.columns(4)
    a.metric("Trạng thái", state["status"])
    b.metric("Epoch đã commit", state["epoch"])
    c.metric("Budget đã dùng (giây)", f"{state['budget']['consumed']:.1f}")
    d.metric("Budget còn khả dụng (giây)", f"{state['budget']['remaining']:.1f}")
    st.caption(f"Phase: {state['phase']} · {state.get('reason', '')}")
    st.caption("Cost: cận trên theo thời gian worker, gồm startup/teardown; reservation không phải chi phí đã tiêu.")
    st.progress(min(1.0, state["budget"]["consumed"] / state["budget"]["cap"]))
    rows = []
    for path in [*selected.glob("segments/**/epochs.jsonl"), *selected.glob("rounds/**/epochs.jsonl")]:
        for item in read_events(path):
            event = item["event"]
            event["branch"] = path.parent.relative_to(selected).as_posix()
            rows.append(event)
    if rows:
        frame = pd.DataFrame(rows)
        left, right = st.columns(2)
        left.subheader("Validation loss")
        left.line_chart(frame.pivot_table(index="epoch", columns="branch", values="val_loss", aggfunc="last"))
        right.subheader("Validation accuracy")
        right.line_chart(frame.pivot_table(index="epoch", columns="branch", values="val_accuracy", aggfunc="last"))
        with st.expander("Metrics từng nhánh / VRAM / thông số huấn luyện"):
            st.caption("grad_norm đo trước clipping. Các cột momentum, weight_decay và grad_clip_norm là giá trị thực tế của từng nhánh.")
            st.dataframe(frame, hide_index=True, width="stretch")
    if state.get("history"):
        current = state["history"][-1]
        with st.expander("Thông số của nhánh đang tiếp tục", expanded=True):
            st.json({key: current.get(key) for key in
                     ("epoch", "learning_rate", "momentum", "weight_decay", "grad_clip_norm")})
    events = read_events(selected/"events.jsonl")
    diagnoses = [e for e in events if e["kind"] == "diagnosis"]
    plans = [e for e in events if e["kind"] == "trial_plan"]
    evaluations = [e for e in events if e["kind"] == "evaluation"]
    with st.expander("LLM: output và lần gọi của từng agent", expanded=state["analysis_mode"] == "openrouter"):
        st.caption("Chỉ hiển thị các bước đã chạy. Run cũ có thể chưa có log Planner/Evaluation.")
        for kind, name, output_key in (("diagnosis", "Analysis", "diagnosis"),
                                       ("planning", "Planner", "trials"),
                                       ("evaluation", "Evaluation", "verdict")):
            records = [e for e in events if e["kind"] == kind]
            st.write(name)
            if records:
                latest_agent = records[-1]
                st.json({"round": latest_agent.get("round"),
                         "output": latest_agent.get(output_key),
                         "metadata": latest_agent.get("metadata", {})})
            else:
                st.write("Chưa có log cho agent này.")
    left, right = st.columns(2)
    left.subheader("Chẩn đoán và thử sửa")
    if diagnoses:
        left.json(diagnoses[-1]["diagnosis"], expanded=True)
    else:
        left.write("Chưa có chẩn đoán; Analysis được gọi khi cửa sổ quan sát đủ dữ liệu.")
    if plans:
        with left.expander("Trial plan và admission", expanded=True):
            st.json(plans[-1])
    right.subheader("Đánh giá và tiếp tục")
    if evaluations:
        latest = evaluations[-1]
        right.json(latest["verdict"], expanded=True)
        right.write(f"Mốc so sánh: epoch {latest['result']['comparison_epoch']}")
    right.write("Checkpoint hiện tại")
    right.code(state["checkpoint"]["path"] if state["checkpoint"] else "Chưa có checkpoint")
    if state.get("best_checkpoint"):
        with right.expander("Best checkpoint trên lineage đã chấp nhận"):
            st.json(state["best_checkpoint"])
    with st.expander("Audit events"):
        st.json(events[-20:])


monitor()
