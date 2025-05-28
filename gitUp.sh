#!/bin/bash

branch_name="main"
datetime=$(date '+%Y-%m-%d %H:%M:%S')
commit_msg="[$datetime] Newest Version Updated"

# === Add & Commit nếu có thay đổi ===
echo "🔄 Committing to '$branch_name'..."
git add -A

if git diff --cached --quiet; then
    echo "⚠️  Nothing to commit."
else
    git commit -m "$commit_msg"
fi

echo "🚀 Pushing to remote..."
if git push origin "$branch_name"; then
    echo "✅ Push thành công lên '$branch_name'"
fi