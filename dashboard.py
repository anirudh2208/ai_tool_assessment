import streamlit as st, json, os, subprocess, sys
from datetime import datetime

st.set_page_config(page_title="AI Assessment Dashboard", layout="wide")
st.title(" AI Assessment — Evaluation Dashboard")

METRICS_FILE = os.path.join(os.path.dirname(__file__), "metrics", "metrics.jsonl")
os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)

def log_metric(task, metric_type, value):
    with open(METRICS_FILE, "a") as f:
        f.write(json.dumps({"ts": datetime.utcnow().isoformat(), "task": task, "type": metric_type, "value": value}) + "\n")

def load_metrics():
    if not os.path.exists(METRICS_FILE): return []
    return [json.loads(l) for l in open(METRICS_FILE) if l.strip()]

#  Sidebar actions 
if st.sidebar.button("▶ Run RAG Evaluation"):
    with st.spinner("Running..."):
        r = subprocess.run([sys.executable, "rag.py", "evaluate"], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        st.sidebar.code(r.stdout + r.stderr)
        for line in r.stdout.split("\n"):
            if "Accuracy" in line:
                try: log_metric("rag", "accuracy", float(line.split("(")[1].split("%")[0]))
                except: pass
            if "Median" in line:
                try: log_metric("rag", "retrieval_ms", float(line.split(":")[1].strip().split(" ")[0]))
                except: pass

if st.sidebar.button("▶ Run Unit Tests"):
    with st.spinner("Running..."):
        r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        st.sidebar.code(r.stdout + r.stderr)

#  Dashboard columns 
metrics = load_metrics()
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader(" Latency")
    lat = [m for m in metrics if m["type"] == "retrieval_ms"]
    st.line_chart({m["ts"][-8:]: m["value"] for m in lat}) if lat else st.info("No data yet.")

with c2:
    st.subheader(" Retrieval Accuracy")
    acc = [m for m in metrics if m["type"] == "accuracy"]
    if acc:
        st.metric("Latest", f"{acc[-1]['value']:.1f}%")
        st.line_chart({m["ts"][-8:]: m["value"] for m in acc})
    else: st.info("No data yet.")

with c3:
    st.subheader(" Agent Runs")
    agent = [m for m in metrics if m["task"] == "agent"]
    st.metric("Total Runs", len(agent)) if agent else st.info("No data yet.")

st.subheader(" Raw Log")
st.dataframe(metrics) if metrics else st.info("Run tasks to see metrics.")
