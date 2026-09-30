"""Local web interface for the Aventro Motors RAG project."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rag import generate_answer, retrieve, settings


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aventro Motors 文档问答</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f4f6f8; color: #17212c; }
    main { max-width: 820px; margin: 0 auto; padding: 48px 20px 80px; }
    h1 { margin: 0 0 8px; font-size: clamp(28px, 5vw, 40px); }
    .subtitle { margin: 0 0 32px; color: #526171; line-height: 1.6; }
    .panel { background: white; border: 1px solid #dce3e9; border-radius: 16px; padding: 24px; box-shadow: 0 8px 30px #1825320a; }
    label { display: block; margin-bottom: 10px; font-weight: 650; }
    textarea { display: block; width: 100%; min-height: 110px; padding: 14px; border: 1px solid #adbbc8; border-radius: 10px; font: inherit; line-height: 1.5; resize: vertical; }
    textarea:focus { outline: 3px solid #b9d8fa; border-color: #2868ad; }
    button { margin-top: 16px; border: 0; border-radius: 10px; padding: 12px 22px; background: #1765a7; color: white; font: inherit; font-weight: 650; cursor: pointer; }
    button:hover { background: #10538c; }
    button:disabled { opacity: .6; cursor: wait; }
    #status { min-height: 24px; margin: 18px 0 0; color: #526171; }
    #result { display: none; margin-top: 24px; }
    #answer { white-space: pre-wrap; line-height: 1.7; }
    #sources { padding-left: 22px; line-height: 1.7; overflow-wrap: anywhere; }
    #sources li { margin-bottom: 8px; }
    .error { color: #ae1d2e !important; }
  </style>
</head>
<body>
  <main>
    <h1>Aventro Motors 文档问答</h1>
    <p class="subtitle">基于本地 Markdown 文档检索相关段落，并使用已配置的模型 API 生成带来源的回答。</p>
    <div class="panel">
      <form id="question-form">
        <label for="question">你的问题</label>
        <textarea id="question" required maxlength="2000" placeholder="例如：如何打开 Aventro 汽车的引擎盖？"></textarea>
        <button id="ask-button" type="submit">提问</button>
      </form>
      <p id="status" role="status"></p>
    </div>
    <section id="result" class="panel" aria-live="polite">
      <h2>回答</h2>
      <div id="answer"></div>
      <h2>检索来源</h2>
      <ol id="sources"></ol>
    </section>
  </main>
  <script>
    const form = document.getElementById('question-form');
    const button = document.getElementById('ask-button');
    const status = document.getElementById('status');
    const result = document.getElementById('result');
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      button.disabled = true;
      status.className = '';
      status.textContent = '正在检索文档并生成回答…';
      result.style.display = 'none';
      try {
        const response = await fetch('/api/ask', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({question: document.getElementById('question').value.trim()})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || '请求失败');
        document.getElementById('answer').textContent = data.answer;
        const sources = document.getElementById('sources');
        sources.replaceChildren();
        for (const item of data.sources) {
          const li = document.createElement('li');
          li.textContent = `${item.file} · ${item.section}`;
          sources.appendChild(li);
        }
        result.style.display = 'block';
        status.textContent = '';
      } catch (error) {
        status.className = 'error';
        status.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status: int, payload: dict) -> None:
        self.send_bytes(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_bytes(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self.send_json(404, {"error": "页面不存在"})

    def do_POST(self) -> None:
        if self.path != "/api/ask":
            self.send_json(404, {"error": "接口不存在"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 10000:
                raise ValueError("请求内容长度无效")
            data = json.loads(self.rfile.read(length))
            question = data.get("question", "")
            if not isinstance(question, str) or not question.strip() or len(question) > 2000:
                raise ValueError("请输入 1 到 2000 字的问题")
            settings()
            hits = retrieve(question.strip(), top_k=5, device="auto")
            answer = generate_answer(question.strip(), hits)
            self.send_json(200, {
                "answer": answer,
                "sources": [
                    {"file": hit["metadata"]["source"], "section": hit["metadata"]["section"]}
                    for hit in hits
                ],
            })
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
        except (RuntimeError, FileNotFoundError) as exc:
            self.send_json(500, {"error": str(exc)})
        except Exception:
            self.send_json(502, {"error": "模型 API 调用失败，请检查终端日志和 .env 配置。"})
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Aventro Motors local RAG web app")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"网页已启动：http://127.0.0.1:{args.port}")
    print("按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
