#!/usr/bin/env python3
"""
简单的 GUI，用于从已训练的 run 中选择 checkpoint 或运行随机策略。
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List

import tkinter as tk
from tkinter import ttk, messagebox

BASE_DIR = Path(__file__).resolve().parent
LOGS_ROOT = BASE_DIR / "logs"
VISUALIZER_SCRIPT = BASE_DIR / "visualize_checkpoint2.py"
RANDOM_VIEWER_SCRIPT = BASE_DIR / "random_viewer.py"

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.prey_test_config import (
    prey_test_config,
)


def load_run_metadata() -> List[Dict]:
    runs: List[Dict] = []
    if not LOGS_ROOT.exists():
        return runs
    for run_dir in sorted(LOGS_ROOT.glob("run_*")):
        meta_path = run_dir / "run_metadata.json"
        env_path = run_dir / "env_config.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta["run_dir"] = str(run_dir.resolve())
        if not meta.get("env_config_file") and env_path.is_file():
            meta["env_config_file"] = str(env_path.resolve())

        checkpoints: List[str] = []
        ray_dir = meta.get("directories", {}).get("ray_results_dir")
        if ray_dir:
            ray_path = Path(ray_dir)
            if ray_path.exists():
                for ckpt in sorted(ray_path.rglob("checkpoint_*")):
                    if ckpt.is_dir():
                        checkpoints.append(str(ckpt))
        meta["checkpoints"] = checkpoints
        runs.append(meta)
    return runs


class Dashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PredPreyGrass Dashboard (env129)")
        self.geometry("800x600")
        self.runs: List[Dict] = []
        self.selected_run: Dict | None = None

        self.override_config_path = BASE_DIR / "env_config_override.json"

        self.create_widgets()
        self.load_config_text(prey_test_config)
        self.refresh_runs()

    def create_widgets(self) -> None:
        top_frame = ttk.Frame(self)
        top_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(top_frame, text="选择运行 (run_<timestamp>):").pack(anchor="w")
        self.run_var = tk.StringVar()
        self.run_combo = ttk.Combobox(
            top_frame, textvariable=self.run_var, state="readonly", width=60
        )
        self.run_combo.pack(fill="x", pady=5)
        self.run_combo.bind("<<ComboboxSelected>>", lambda *_: self.on_run_selected())

        ttk.Button(top_frame, text="刷新列表", command=self.refresh_runs).pack(anchor="e")

        checkpoint_frame = ttk.LabelFrame(self, text="Checkpoint 与参数")
        checkpoint_frame.pack(fill="x", padx=10, pady=10)
        ttk.Label(checkpoint_frame, text="可用的 checkpoint:").pack(anchor="w")
        self.ckpt_var = tk.StringVar()
        self.ckpt_combo = ttk.Combobox(
            checkpoint_frame, textvariable=self.ckpt_var, state="readonly", width=80
        )
        self.ckpt_combo.pack(fill="x", pady=5)
        button_row = ttk.Frame(checkpoint_frame)
        button_row.pack(fill="x", pady=5)
        ttk.Button(
            button_row,
            text="重置为该 run 参数",
            command=self.reset_to_run_config,
        ).pack(side="left", padx=5)
        ttk.Button(
            button_row,
            text="重置为默认参数",
            command=self.reset_to_default_config,
        ).pack(side="left", padx=5)

        config_frame = ttk.LabelFrame(self, text="环境参数（可编辑 JSON）")
        config_frame.pack(fill="both", padx=10, pady=10, expand=True)
        self.config_text = tk.Text(config_frame, height=12)
        self.config_text.pack(fill="both", expand=True, padx=5, pady=5)

        button_frame = ttk.Frame(self)
        button_frame.pack(fill="x", padx=10, pady=10)
        ttk.Button(
            button_frame, text="启动随机策略 (Random Viewer)", command=self.launch_random
        ).pack(side="left", padx=5)
        ttk.Button(
            button_frame,
            text="启动已训练策略",
            command=self.launch_checkpoint,
        ).pack(side="left", padx=5)

        self.status_var = tk.StringVar(value="准备就绪")
        ttk.Label(self, textvariable=self.status_var).pack(anchor="w", padx=10, pady=5)

        info_frame = ttk.LabelFrame(self, text="Run 信息")
        info_frame.pack(fill="both", padx=10, pady=10, expand=True)
        self.info_text = tk.Text(info_frame, height=10)
        self.info_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.info_text.configure(state="disabled")

    def refresh_runs(self) -> None:
        self.runs = load_run_metadata()
        values = [run["run_id"] for run in self.runs]
        self.run_combo["values"] = values
        self.run_var.set("")
        self.ckpt_combo["values"] = []
        self.ckpt_var.set("")
        self.selected_run = None
        self.set_status(f"已加载 {len(values)} 个 run。")
        self.display_info("")

    def on_run_selected(self) -> None:
        run_id = self.run_var.get()
        self.selected_run = next((r for r in self.runs if r["run_id"] == run_id), None)
        checkpoints = self.selected_run.get("checkpoints", []) if self.selected_run else []
        self.ckpt_combo["values"] = checkpoints
        self.ckpt_var.set(checkpoints[-1] if checkpoints else "")
        info = json.dumps(self.selected_run or {}, ensure_ascii=False, indent=2)
        self.display_info(info)
        self.reset_to_run_config(silent=True)

    def load_config_text(self, config: Dict[str, Any]) -> None:
        pretty = json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True)
        self.config_text.delete("1.0", tk.END)
        self.config_text.insert(tk.END, pretty)

    def display_info(self, text: str) -> None:
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, text)
        self.info_text.configure(state="disabled")

    def parse_config_text(self) -> Dict[str, Any] | None:
        raw = self.config_text.get("1.0", tk.END).strip()
        if not raw:
            messagebox.showwarning("提示", "请先在文本框中填写 JSON 配置。")
            return None
        try:
            data = json.loads(raw)
        except Exception as exc:
            messagebox.showerror("JSON 错误", f"无法解析配置：{exc}")
            return None
        if not isinstance(data, dict):
            messagebox.showerror("JSON 错误", "配置必须是一个 JSON 对象。")
            return None
        return data

    def persist_config_to_file(self) -> str | None:
        data = self.parse_config_text()
        if data is None:
            return None
        try:
            self.override_config_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            messagebox.showerror("写入失败", f"无法写入配置文件：{exc}")
            return None
        return str(self.override_config_path)

    def reset_to_run_config(self, silent: bool = False) -> None:
        if not self.selected_run:
            if not silent:
                messagebox.showinfo("提示", "未选择 run，已恢复默认配置。")
            self.load_config_text(prey_test_config)
            return
        env_file = self.selected_run.get("env_config_file")
        if not env_file:
            if not silent:
                messagebox.showwarning("提示", "该 run 无 env_config.json，已恢复默认配置。")
            self.load_config_text(prey_test_config)
            return
        path = Path(env_file)
        if not path.is_file():
            if not silent:
                messagebox.showwarning("提示", f"找不到配置文件：{env_file}，已恢复默认配置。")
            self.load_config_text(prey_test_config)
            return
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            if not silent:
                messagebox.showerror("读取失败", f"无法读取 {env_file}：{exc}")
            self.load_config_text(prey_test_config)
            return
        self.load_config_text(config)
        if not silent:
            self.set_status("已载入该 run 的配置。")

    def reset_to_default_config(self) -> None:
        self.load_config_text(prey_test_config)
        self.set_status("已恢复默认配置。")

    def launch_random(self) -> None:
        config_path = self.persist_config_to_file()
        if config_path is None:
            return
        cmd = [sys.executable, str(RANDOM_VIEWER_SCRIPT)]
        cmd += ["--env-config-file", config_path]
        self.run_subprocess(cmd, "Random viewer 已启动。")

    def launch_checkpoint(self) -> None:
        if not self.selected_run:
            messagebox.showwarning("提示", "请先选择一个 run。")
            return
        checkpoint = self.ckpt_var.get()
        if not checkpoint:
            messagebox.showwarning("提示", "当前 run 没有可用的 checkpoint。")
            return
        config_path = self.persist_config_to_file()
        if config_path is None:
            return
        cmd = [sys.executable, str(VISUALIZER_SCRIPT), "--checkpoint", checkpoint]
        cmd += ["--env-config-file", config_path]
        self.run_subprocess(cmd, "可视化已启动。")

    def run_subprocess(self, cmd: List[str], success_message: str) -> None:
        def _target():
            try:
                subprocess.Popen(cmd)
                self.set_status(success_message)
            except Exception as exc:
                self.set_status(f"启动失败: {exc}")
                messagebox.showerror("错误", str(exc))

        self.set_status(f"执行命令: {' '.join(cmd)}")
        threading.Thread(target=_target, daemon=True).start()

    def set_status(self, message: str) -> None:
        self.status_var.set(message)


def main() -> None:
    app = Dashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
