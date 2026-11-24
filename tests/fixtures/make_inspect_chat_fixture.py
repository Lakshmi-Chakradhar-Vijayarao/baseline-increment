"""Generate real Inspect logs (the shipped fixture was made on Python 3.14, whose zstd members were then repacked
with deflate, byte-identical content, so that the fixture reads on Python 3.10-3.13 as Inspect writes there).
Generate real Inspect logs (mockllm, no keys): chat-style + long inputs, metadata, 3 epochs, three scorers
(C/I match, numeric detector, dict-valued), one run in .eval and one in .json format."""
import random, sys
from inspect_ai import Task, eval as inspect_eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ModelOutput, get_model
from inspect_ai.scorer import Score, Target, mean, match, scorer
from inspect_ai.solver import TaskState, generate

LONG = " ".join(["context"] * 400)          # long text: may be stored as an attachment


def samples():
    out = []
    for i in range(30):
        a, b = i % 7, (i * 3) % 5
        msgs = [ChatMessageSystem(content="You are a careful calculator. " + (LONG if i % 3 == 0 else "")),
                ChatMessageUser(content=f"What is {a} + {b}? Answer with the number only.")]
        out.append(Sample(id=f"q{i}", input=msgs if i % 2 == 0 else f"What is {a} + {b}? " + (LONG if i % 5 == 0 else ""),
                          target=str(a + b), metadata={"difficulty": i % 4, "topic": "add" if i % 2 else "sum"}))
    return out


@scorer(metrics=[mean()])
def risk():
    async def score(state: TaskState, target: Target):
        return Score(value=float(len(state.output.completion)) / 10.0)
    return score


@scorer(metrics={"*": [mean()]})
def multi():
    async def score(state: TaskState, target: Target):
        ok = state.output.completion.strip() == target.text
        return Score(value={"exact": 1.0 if ok else 0.0, "confidence": 0.3 + 0.1 * (len(state.output.completion) % 5)})
    return score


@task
def arith():
    return Task(dataset=samples(), solver=generate(), scorer=[match(), risk(), multi()], epochs=3)


if __name__ == "__main__":
    fmt, seed = sys.argv[1], int(sys.argv[2])
    rng = random.Random(seed)
    outs = []
    for s in samples():
        for _ in range(3):
            right = rng.random() < 0.6
            outs.append(ModelOutput.from_content(model="mockllm", content=s.target if right else str(rng.randint(10, 99)) + (" extra words" * rng.randint(0, 3))))
    model = get_model("mockllm/model", custom_outputs=outs)
    inspect_eval(arith(), model=model, log_format=fmt, log_dir=f"logs_{fmt}_{seed}", display="none", max_connections=1, max_samples=1)
