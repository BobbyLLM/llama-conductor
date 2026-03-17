from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_front_matter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    m = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n?", raw, flags=re.S)
    if not m:
        return {}, raw
    meta = yaml.safe_load(m.group(1)) or {}
    body = raw[m.end() :]
    return meta, body


def load_defaults(config_path: Path) -> dict[str, dict]:
    cfg = yaml.safe_load(read_text(config_path)) or {}
    out: dict[str, dict] = {}
    for item in cfg.get("defaults", []) or []:
        scope = item.get("scope", {}) or {}
        path = scope.get("path")
        vals = item.get("values", {}) or {}
        if path and vals:
            out[path.replace("\\", "/")] = vals
    return out


def strip_liquid_relative_url(md: str, base_path: str) -> str:
    pat = re.compile(r"\{\{\s*'([^']+)'\s*\|\s*relative_url\s*\}\}")
    # Keep root-relative paths here; base prefixing is handled in HTML phase.
    return pat.sub(lambda m: f"{m.group(1)}", md)


def ensure_h1(body: str, title: str) -> str:
    if re.search(r"(?m)^\s*#\s+", body):
        return body
    return f"# {title}\n\n{body}"


def page_shell(title: str, content_html: str, base_path: str) -> str:
    css = f"{base_path}/assets/css/site.css?v={int(dt.datetime.now().timestamp())}"
    about_href = f"{base_path}/about.html"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} | llama-conductor</title>
  <link rel="stylesheet" href="{css}">
</head>
<body>
  <div class="bg-grid"></div>
  <header class="site-header">
    <a class="brand" href="{base_path}/">llama-conductor</a>
    <nav>
      <a href="{base_path}/">Home</a>
      <a href="{about_href}">About</a>
    </nav>
  </header>
  <main class="wrap">
    <article class="card prose">
      {content_html}
    </article>
  </main>
</body>
</html>
"""


def _pretty_to_html(path: str) -> str:
    if path == "/":
        return path
    if path.endswith("/"):
        return path.rstrip("/") + ".html"
    return path


def prefix_root_links(html: str, base_path: str) -> str:
    base = base_path.rstrip("/")

    def repl_href(m: re.Match[str]) -> str:
        path = m.group(1)
        new_path = _pretty_to_html(path)
        return f'href="{base}{new_path}"'

    html = re.sub(r'href="(/(?!/)[^"]*)"', repl_href, html)
    html = re.sub(r'src="/(?!/)', f'src="{base}/', html)
    return html


def render_markdown(body: str) -> str:
    # Preferred parser path: markdown-it-py
    try:
        from markdown_it import MarkdownIt  # type: ignore

        md = MarkdownIt("commonmark", {"linkify": True, "typographer": True}).enable("table")
        return md.render(body)
    except Exception:
        pass

    # Prefer pandoc if available; otherwise use a minimal fallback.
    p = shutil.which("pandoc")
    if p:
        proc = subprocess.run(
            [p, "-f", "gfm", "-t", "html5"],
            input=body,
            text=True,
            capture_output=True,
            check=True,
        )
        return proc.stdout
    # Minimal fallback for this repo's content.
    html_lines: list[str] = []
    in_code = False
    in_ul = False

    def close_ul() -> None:
        nonlocal in_ul
        if in_ul:
            html_lines.append("</ul>")
            in_ul = False

    def linkify_plain_urls(s: str) -> str:
        # Only linkify outside existing HTML tags.
        parts = re.split(r"(<[^>]+>)", s)
        out: list[str] = []

        def repl(m: re.Match[str]) -> str:
            url = m.group(1)
            # Trim common trailing punctuation that should remain outside the link.
            tail = ""
            while url and url[-1] in "),.!?;:":
                tail = url[-1] + tail
                url = url[:-1]
            return f'<a href="{url}">{url}</a>{tail}'

        url_re = re.compile(r"(https?://[^\s<]+)")
        for part in parts:
            if part.startswith("<") and part.endswith(">"):
                out.append(part)
            else:
                out.append(url_re.sub(repl, part))
        return "".join(out)

    for line in body.splitlines():
        if line.strip().startswith("```"):
            close_ul()
            if not in_code:
                html_lines.append("<pre><code>")
            else:
                html_lines.append("</code></pre>")
            in_code = not in_code
            continue
        if in_code:
            html_lines.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            continue
        if line.startswith("# "):
            close_ul()
            html_lines.append(f"<h1>{line[2:].strip()}</h1>")
            continue
        if line.startswith("## "):
            close_ul()
            html_lines.append(f"<h2>{line[3:].strip()}</h2>")
            continue
        if line.startswith("- "):
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            item = line[2:].strip()
            item = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', item)
            html_lines.append(f"<li>{item}</li>")
            continue
        if line.strip() == "":
            close_ul()
            html_lines.append("")
            continue
        close_ul()
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', line)
        text = linkify_plain_urls(text)
        html_lines.append(f"<p>{text}</p>")
    close_ul()
    return "\n".join(html_lines)


def dest_from_permalink(out_root: Path, permalink: str) -> Path:
    p = permalink.strip()
    if not p.startswith("/"):
        p = "/" + p
    rel = p.strip("/")
    if rel == "":
        return out_root / "index.html"
    if p.endswith("/"):
        return out_root / f"{rel}.html"
    return out_root / f"{rel}.html"


def build_site(repo_root: Path, out_root: Path, base_path: str) -> None:
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    defaults = load_defaults(repo_root / "_config.yml")
    # static assets
    for rel in ("assets", "logo"):
        src = repo_root / rel
        if src.exists():
            shutil.copytree(src, out_root / rel, dirs_exist_ok=True)

    pages = [
        "index.md",
        "about.md",
        "blog/index.md",
        "blog/meme-test.md",
        "blog/SCP.md",
    ]
    for rel in pages:
        src = repo_root / rel
        raw = read_text(src)
        meta_fm, body = split_front_matter(raw)
        meta = {}
        meta.update(defaults.get(rel, {}))
        meta.update(meta_fm)

        title = meta.get("title") or src.stem
        permalink = meta.get("permalink")
        if not permalink:
            if rel == "index.md":
                permalink = "/"
            elif rel.endswith("/index.md"):
                permalink = "/" + rel[: -len("index.md")]
            else:
                permalink = "/" + rel.replace(".md", "/")

        body = strip_liquid_relative_url(body, base_path)
        body = ensure_h1(body, str(title))
        html = render_markdown(body)
        html = prefix_root_links(html, base_path)
        doc = page_shell(str(title), html, base_path)

        dst = dest_from_permalink(out_root, str(permalink))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(doc, encoding="utf-8")

    (out_root / ".nojekyll").write_text("", encoding="utf-8")


def publish_pages(repo_root: Path, out_root: Path, branch: str, remote: str, site_subdir: str) -> None:
    run(["git", "fetch", remote], repo_root)
    tmp = Path(tempfile.mkdtemp(prefix="codeberg-pages-"))
    try:
        run(["git", "worktree", "add", "--force", str(tmp), f"{remote}/{branch}"], repo_root)
        for item in tmp.iterdir():
            if item.name == ".git":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        for item in out_root.iterdir():
            dst = tmp / item.name
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)
        (tmp / ".nojekyll").write_text("", encoding="utf-8")
        run(["git", "add", "-A"], tmp)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(tmp))
        if diff.returncode != 0:
            run(["git", "commit", "-m", "Publish Codeberg Pages static site"], tmp)
            run(["git", "push", remote, f"HEAD:{branch}"], tmp)
    finally:
        try:
            run(["git", "worktree", "remove", "--force", str(tmp)], repo_root)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out-dir", default=".codeberg_site")
    ap.add_argument("--base-path", default="/llama-conductor")
    ap.add_argument("--branch", default="pages")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--site-subdir", default="")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_root = repo_root / args.out_dir
    build_site(repo_root, out_root, args.base_path.rstrip("/"))
    if args.publish:
        publish_pages(repo_root, out_root, args.branch, args.remote, args.site_subdir)


if __name__ == "__main__":
    main()
