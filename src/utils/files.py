"""訓練輸出用的檔案操作。"""

import shutil
from pathlib import Path


def replace_dir(src: Path, dst: Path) -> None:
    """把 src 的內容複製過去，整個取代 dst（dst 原本的檔案都會被刪掉）。

    先完整複製到 dst 旁邊的 <名稱>.new，再用 rename 換上去：
    複製途中出錯時 dst 還是舊的那份，不會留下新舊混雜的內容。
    """
    dst = Path(dst).resolve()
    new_dir = dst.with_name(dst.name + ".new")
    old_dir = dst.with_name(dst.name + ".old")
    for d in (new_dir, old_dir):
        shutil.rmtree(d, ignore_errors=True)

    shutil.copytree(src, new_dir)
    if dst.exists():
        dst.rename(old_dir)
    new_dir.rename(dst)
    shutil.rmtree(old_dir, ignore_errors=True)
