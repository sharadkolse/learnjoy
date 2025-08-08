# AI Learning Lab — Streamlit MVP
# Free to run locally (streamlit) and free to deploy on Streamlit Community Cloud
# File: streamlit_app.py
# -----------------------------------------------------
# Quick start (locally):
#   pip install streamlit matplotlib graphviz
#   streamlit run streamlit_app.py
# -----------------------------------------------------

import math
from dataclasses import dataclass, field
from typing import List, Dict

import streamlit as st
from st_aggrid import AgGrid
import textwrap

# Optional: matplotlib for simple interactive demos
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Learning Lab", page_icon="🧠", layout="wide")

# -----------------------------
# Content Model
# -----------------------------
@dataclass
class QuizQ:
    prompt: str
    options: List[str]
    correct_index: int
    explanation: str

@dataclass
class Topic:
    key: str
    title: str
    tags: List[str]
    tl_dr: str
    eli5: str
    intuition: str
    formal: str
    analogies: List[str]
    code_snippet: str
    diagram_graphviz: str
    citations: List[str] = field(default_factory=list)


TOPICS: Dict[str, Topic] = {}
QUIZZES: Dict[str, List[QuizQ]] = {}

# -----------------------------
# Topic Seeds (3 examples to start)
# -----------------------------
TOPICS["gradient_descent"] = Topic(
    key="gradient_descent",
    title="Gradient Descent",
    tags=["ML", "Optimization", "Core"],
    tl_dr="Move downhill on the loss surface by stepping opposite the gradient until you reach (a) minimum.",
    eli5=(
        "Imagine you are on a foggy hill trying to get to the lowest point. You can't see far, "
        "but you can feel which way the ground slopes. You take small steps downhill again and again."
    ),
    intuition=(
        "The gradient tells you the direction of steepest increase. Going the negative direction reduces the loss the fastest locally. "
        "Step size (learning rate) balances speed vs. stability: too big and you overshoot, too small and you crawl."
    ),
    formal=(
        "For parameters w and differentiable loss L(w), the update is: w_{t+1} = w_t - η ∇L(w_t). "
        "Under convexity and appropriate η schedules, convergence to a global minimum is guaranteed; otherwise to a critical point."
    ),
    analogies=[
        "Hiking downhill with short careful steps (learning rate).",
        "Paying off debt by always attacking the steepest portion first.",
    ],
    code_snippet=textwrap.dedent(
        """
        # 1D gradient descent demo (f(x)=x^2)
        import numpy as np
        import matplotlib.pyplot as plt

        def f(x):
            return x**2

        def grad(x):
            return 2*x

        x = 4.0
        lr = 0.2
        xs = [x]
        for _ in range(15):
            x = x - lr*grad(x)
            xs.append(x)

        X = np.linspace(-5, 5, 400)
        Y = f(X)
        plt.figure()
        plt.plot(X, Y)
        plt.scatter(xs, [f(v) for v in xs])
        plt.title("Gradient Descent on f(x)=x^2")
        plt.xlabel("x"); plt.ylabel("f(x)")
        plt.show()
        """
    ),
    diagram_graphviz=textwrap.dedent(
        """
        digraph GD {
          rankdir=LR;
          node [shape=box, style=rounded];
          Loss -> "Compute gradient ∇L" -> "Update: w := w - η∇L" -> "Check stop?";
          "Check stop?" -> Loss [label="no", style=dashed];
          "Check stop?" -> "Return w*" [label="yes"];
        }
        """
    ),
    citations=[
        "Goodfellow, Bengio, Courville — Deep Learning (2016), Ch. 4",
        "Nocedal & Wright — Numerical Optimization (2e)"
    ],
)

QUIZZES["gradient_descent"] = [
    QuizQ(
        prompt="If the learning rate is too large, what’s the most likely behavior?",
        options=["Converges faster with no issues", "Diverges or overshoots the minimum", "Gets stuck at a local minimum always", "No effect"],
        correct_index=1,
        explanation="Large steps can jump over the valley and cause oscillation or divergence.",
    ),
    QuizQ(
        prompt="What does the gradient point to?",
        options=["Steepest decrease", "Steepest increase", "A saddle point", "A global minimum"],
        correct_index=1,
        explanation="The gradient points to the direction of steepest increase; we move against it to decrease.",
    ),
]

TOPICS["dropout"] = Topic(
    key="dropout",
    title="Dropout",
    tags=["DL", "Regularization"],
    tl_dr="Randomly deactivate neurons during training to prevent co-adaptation; scale at train-time (inverted dropout) so test-time uses full network without scaling.",
    eli5=(
        "In practice time, some teammates sit out randomly so everyone learns to play well independently. "
        "At match time, the full team plays together."
    ),
    intuition=(
        "By randomly zeroing activations, the network cannot rely on specific paths and must learn robust features. "
        "With inverted dropout, we scale activations during training by 1/(1-p) so that expected activation matches test-time."
    ),
    formal=(
        "For activation h, with dropout mask m~Bernoulli(1-p), inverted dropout uses: h' = (m ⊙ h) / (1-p). "
        "At inference, use the full network without masks or scaling."
    ),
    analogies=[
        "Scrimmage where random players sit out, improving team resilience.",
        "Redundant circuits: any one wire failing shouldn’t kill the system.",
    ],
    code_snippet=textwrap.dedent(
        """
        # PyTorch-style inverted dropout layer (conceptual)
        import torch
        import torch.nn.functional as F

        def inverted_dropout(x, p=0.5, training=True):
            if not training or p == 0.0:
                return x
            mask = (torch.rand_like(x) > p).float()
            return (x * mask) / (1 - p)
        """
    ),
    diagram_graphviz=textwrap.dedent(
        """
        digraph Dropout {
          rankdir=LR;
          node [shape=circle];
          subgraph cluster_train {
            label="Training (random mask)";
            a -> b; a -> c; a -> d;
            {rank=same; b c d}
            b [style=filled, fillcolor=lightgray, label="0"];
            c [label="h"];
            d [style=filled, fillcolor=lightgray, label="0"];
          }
          subgraph cluster_test {
            label="Inference (no mask)";
            x -> y; x -> z; x -> w;
          }
        }
        """
    ),
    citations=[
        "Srivastava et al. — Dropout: A Simple Way to Prevent Neural Networks from Overfitting (JMLR, 2014)"
    ],
)

QUIZZES["dropout"] = [
    QuizQ(
        prompt="In inverted dropout, why do we divide by (1-p) during training?",
        options=["To slow training", "To match expected activation at test-time", "To reduce gradient noise", "No specific reason"],
        correct_index=1,
        explanation="Scaling keeps the expected value of activations consistent between train and test.",
    ),
]

TOPICS["transformer"] = Topic(
    key="transformer",
    title="Transformers & Self-Attention",
    tags=["DL", "NLP", "GenAI"],
    tl_dr="Self-attention lets each token weigh other tokens to build rich context; stacks of attention + feed-forward blocks with residuals form Transformers.",
    eli5=(
        "Reading a sentence, you look back and forth to see which words matter most to the current word. "
        "You then write a new sentence where each word is influenced by the important ones."
    ),
    intuition=(
        "Attention creates weighted summaries of other tokens (via Q·K^T). Multi-heads view relationships from different angles. "
        "Residual connections and layer norm stabilize deep stacks."
    ),
    formal=(
        "Given Q,K,V, attention(Q,K,V)=softmax(QK^T/√d_k) V. Encoder/decoder blocks repeat: MHA → Add&Norm → FFN → Add&Norm."
    ),
    analogies=[
        "Roundtable discussion: each speaker listens to others and summarizes key points.",
        "Spotlight in a theater focusing on the most relevant actors for each line.",
    ],
    code_snippet=textwrap.dedent(
        """
        # Toy self-attention (single head) for tiny sequence
        import numpy as np

        def softmax(x):
            x = x - x.max(axis=-1, keepdims=True)
            e = np.exp(x)
            return e / e.sum(axis=-1, keepdims=True)

        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 3 tokens, d=2
        Wq = np.eye(2); Wk = np.eye(2); Wv = np.eye(2)
        Q = X @ Wq; K = X @ Wk; V = X @ Wv
        scores = Q @ K.T / np.sqrt(2)
        weights = softmax(scores)
        Y = weights @ V
        print("attention weights:\n", weights)
        print("output:\n", Y)
        """
    ),
    diagram_graphviz=textwrap.dedent(
        """
        digraph Transformer {
          rankdir=LR;
          node [shape=box, style=rounded];
          Input -> Embedding -> "Multi-Head Attention" -> "Add & Norm" -> "Feed-Forward" -> "Add & Norm" -> Output;
        }
        """
    ),
    citations=[
        "Vaswani et al. — Attention Is All You Need (NeurIPS, 2017)",
        "Press et al. — Train Short, Test Long (2021)"
    ],
)

QUIZZES["transformer"] = [
    QuizQ(
        prompt="In scaled dot-product attention, why divide by √d_k?",
        options=["For numerical stability and gradient flow", "To make outputs smaller", "To speed up training", "No reason"],
        correct_index=0,
        explanation="Scaling prevents large dot products from saturating softmax, improving stability.",
    ),
]

# -----------------------------
# Helpers
# -----------------------------
TAG_PALETTE = {
    "ML": "🎯",
    "Optimization": "📉",
    "Core": "🧩",
    "DL": "🧠",
    "Regularization": "🛡️",
    "NLP": "🗣️",
    "GenAI": "✨",
}


def tag_pill(tag: str) -> str:
    emoji = TAG_PALETTE.get(tag, "🔹")
    return f"{emoji} {tag}"


# -----------------------------
# UI — Sidebar
# -----------------------------
st.sidebar.title("AI Learning Lab")
st.sidebar.caption("Engaging, accurate, and fun learning for AI/ML/DL/GenAI.")

all_tags = sorted({t for topic in TOPICS.values() for t in topic.tags})
sel_tags = st.sidebar.multiselect("Filter by tags", options=all_tags, default=[])
search = st.sidebar.text_input("Search topics", placeholder="e.g. dropout, attention, gradient…")

# Topic selection
filtered = [t for t in TOPICS.values() if (not sel_tags or any(tag in t.tags for tag in sel_tags))
            and (not search or (search.lower() in t.title.lower() or search.lower() in t.tl_dr.lower()))]

sel_key = st.sidebar.selectbox(
    "Choose a topic",
    options=[t.key for t in filtered] if filtered else ["gradient_descent"],
    format_func=lambda k: TOPICS[k].title,
)

st.sidebar.markdown("---")
st.sidebar.write("**Download notes** for your selected topic as Markdown.")

# -----------------------------
# UI — Main
# -----------------------------
sel_topic = TOPICS.get(sel_key, TOPICS["gradient_descent"])  # fallback
st.title("🧠 AI Learning Lab")
st.subheader(sel_topic.title)

# Tag row
st.write(" ".join([tag_pill(t) for t in sel_topic.tags]))

st.info(sel_topic.tl_dr)

# Tabs
about_tab, eli5_tab, intuition_tab, formal_tab, diagram_tab, code_tab, quiz_tab, cite_tab = st.tabs([
    "About", "ELI5", "Intuition", "Formal", "Diagram", "Code Demo", "Quiz", "Citations"
])

with about_tab:
    st.markdown("### What you'll learn")
    st.write("- Big-picture idea\n- Why it matters\n- Common pitfalls\n- How to explain it to a stakeholder")
    st.markdown("---")
    st.markdown("#### Analogies")
    for a in sel_topic.analogies:
        st.markdown(f"- {a}")

with eli5_tab:
    st.write(sel_topic.eli5)

with intuition_tab:
    st.write(sel_topic.intuition)

with formal_tab:
    st.write(sel_topic.formal)

with diagram_tab:
    st.graphviz_chart(sel_topic.diagram_graphviz)

with code_tab:
    st.markdown("##### Reference snippet")
    st.code(sel_topic.code_snippet, language="python")

    # Optional interactive demo for specific topics
    if sel_topic.key == "gradient_descent":
        st.markdown("---")
        st.markdown("### Interactive: 1D Gradient Descent")
        lr = st.slider("Learning rate (η)", 0.01, 1.0, 0.2, 0.01)
        steps = st.slider("Steps", 1, 50, 15, 1)
        x0 = st.slider("Initial x", -8.0, 8.0, 4.0, 0.1)

        def f(x):
            return x**2
        def grad(x):
            return 2*x

        x = float(x0)
        xs = [x]
        for _ in range(int(steps)):
            x -= lr*grad(x)
            xs.append(x)

        X = np.linspace(-8, 8, 400)
        Y = f(X)
        fig = plt.figure()
        plt.plot(X, Y)
        plt.scatter(xs, [f(v) for v in xs])
        plt.title("Gradient Descent on f(x)=x^2")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        st.pyplot(fig)

with quiz_tab:
    qs = QUIZZES.get(sel_topic.key, [])
    if not qs:
        st.info("Quiz coming soon for this topic.")
    else:
        for i, q in enumerate(qs, 1):
            st.markdown(f"**Q{i}. {q.prompt}**")
            choice = st.radio("Select one:", q.options, key=f"q_{sel_topic.key}_{i}")
            if st.button(f"Check answer {i}"):
                idx = q.options.index(choice)
                if idx == q.correct_index:
                    st.success("Correct! " + q.explanation)
                else:
                    st.error("Not quite. " + q.explanation)
            st.markdown("---")

with cite_tab:
    for c in sel_topic.citations:
        st.markdown(f"- {c}")

# -----------------------------
# Notes download
# -----------------------------
notes_md = f"""
# {sel_topic.title}

**Tags:** {', '.join(sel_topic.tags)}

**TL;DR**  
{sel_topic.tl_dr}

## ELI5
{sel_topic.eli5}

## Intuition
{sel_topic.intuition}

## Formal
{sel_topic.formal}

## Analogies
- """ + "\n- ".join(sel_topic.analogies) + "\n\n## Citations\n- " + "\n- ".join(sel_topic.citations) + "\n"

st.download_button("⬇️ Download notes (Markdown)", data=notes_md, file_name=f"{sel_topic.key}_notes.md", mime="text/markdown")

st.caption("Built with ❤️ to make AI learning engaging, accurate, and fun. Add more topics by extending TOPICS and QUIZZES.")
