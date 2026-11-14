"""Запуск: ``python -m nullgraft_demo [--data PATH] [--ui qt|mpl]``."""

from __future__ import annotations

import argparse
import logging
import sys

from nullgraft_demo.config import cache_dir, load_config
from nullgraft_demo.datasource import open_source
from nullgraft_demo.workers import ResultCache


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="nullgraft_demo", description="Интерактивное демо алгоритмов nullgraft")
    ap.add_argument("--data", help="папка Visible Human (выход register_vhp.py) или файл среза; без него — синтетика")
    ap.add_argument("--ui", choices=("qt", "mpl"), help="интерфейс (по умолчанию из конфига)")
    ap.add_argument("--config", help="YAML поверх настроек по умолчанию")
    ap.add_argument("--cache-dir", help="каталог кэша (по умолчанию из конфига)")
    ap.add_argument("--screenshot", help="выполнить демо-сценарий, сохранить снимок окна и выйти")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.cache_dir:
        cfg["cache_dir"] = args.cache_dir
    source = open_source(args.data, cfg)
    cache = ResultCache(cache_dir(cfg))
    ui = args.ui or cfg.get("ui", "qt")
    if ui == "qt":
        try:
            from nullgraft_demo.app_qt import run
        except ImportError as exc:
            logging.warning("PyQt6 недоступен (%s): запускаю интерфейс на matplotlib", exc)
            ui = "mpl"
    if ui == "mpl":
        from nullgraft_demo.app_mpl import run
    return run(cfg, source, cache, screenshot=args.screenshot)


if __name__ == "__main__":
    sys.exit(main())
