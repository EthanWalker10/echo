---
name: review-pr
description: Review current branch changes and provide actionable code suggestions.
---

## Changed Files
!`git diff --name-only`

## Detailed Diff
!`git diff`

Please review the changes above and output your feedback grouped by file:
1. Potential bugs or edge cases (especially in async flows or audio I/O).
2. Maintainability and readability improvements.
3. Deviation from the MVP architecture (e.g., accidental inclusion of complex frameworks).
Keep suggestions concise and actionable.