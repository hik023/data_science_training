from typing import TypedDict

class CategoryProgress(TypedDict):
    completed: int
    not_started: int

categories = {}

with open("Readme.md", "r") as f:
    current_category = None
    for line in f.readlines():
        if line[:3] == "## ":
            current_category = line[2:].strip()
            categories[current_category] = CategoryProgress(completed=0, not_started=0)
        elif line[:6] == "- [x] ":
            categories[current_category]["completed"] += 1
        elif line[:6] == "- [ ] ":
            categories[current_category]["not_started"] += 1

for name, data in categories.items():
    all_themes = data["completed"] + data["not_started"]
    if all_themes != 0:
        percent = data["completed"] / (all_themes / 100)
        print(f"{name} url:\n![](https://geps.dev/progress/{percent:.0f})")