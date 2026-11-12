"""
Console output helper
=====================
Trên Windows, khi stdout bị chuyển hướng ra file hoặc pipe (ví dụ
``python scripts/train_neat.py > log.txt``), Python ghi bằng mã hóa mặc định
cp1252 và chữ tiếng Việt có dấu (hay λ, ★) làm chương trình dừng với
UnicodeEncodeError. Gọi utf8_stdout() ở đầu mỗi script để ghi bằng UTF-8.
Khi chạy trong cửa sổ console, Python vốn đã in được Unicode nên không đổi gì.
"""

import sys


def utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
