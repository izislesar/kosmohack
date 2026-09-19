#!/usr/bin/env python3
"""Validate submission.csv vs template (case section 6). Usage: validate.py <sub> <template>"""
import csv
import sys

sub_path, tpl_path = sys.argv[1], sys.argv[2]
sub = list(csv.DictReader(open(sub_path)))
tpl = list(csv.DictReader(open(tpl_path)))
print("rows", len(sub), "tpl", len(tpl))
sp = sorted((r["chip_id"], int(r["class_id"])) for r in sub)
tp = sorted((r["chip_id"], int(r["class_id"])) for r in tpl)
print("pairs_match", sp == tp)
bad_nan = [r for r in sub if r["rle"] is None or "nan" in str(r["rle"]).lower()]
print("nan_rows", len(bad_nan))


def check(s, H, W):
    if not s.strip():
        return True
    nums = list(map(int, s.split()))
    if len(nums) % 2:
        return False
    st, ln = nums[0::2], nums[1::2]
    if any(x < 1 for x in st):
        return False
    if any(a + b - 1 > H * W for a, b in zip(st, ln)):
        return False
    if st != sorted(st):
        return False
    for (a, la), (b, lb) in zip(zip(st, ln), zip(st[1:], ln[1:])):
        if b <= a + la:
            return False
    return True


sizes = {"AF_te": (256, 256), "BS_te": (512, 512)}
ok = all(check(r["rle"], *sizes[r["chip_id"][:5]]) for r in sub)
print("rle_ok", ok)
# BS non-overlap per chip
from collections import defaultdict
cov = defaultdict(set)
overlap = 0
for r in sub:
    if not r["chip_id"].startswith("BS") or not r["rle"].strip():
        continue
    nums = list(map(int, r["rle"].split()))
    px = set()
    for a, b in zip(nums[0::2], nums[1::2]):
        px.update(range(a, a + b))
    before = len(cov[r["chip_id"]])
    cov[r["chip_id"]] |= px
    overlap += before + len(px) - len(cov[r["chip_id"]])
print("bs_overlap_px", overlap)
tot = 0
for r in sub:
    if r["rle"].strip():
        nums = list(map(int, r["rle"].split()))
        tot += sum(nums[1::2])
print("total_pred_px", tot)
