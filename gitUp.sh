#!/bin/bash

branch_name="main"
datetime=$(date '+%Y-%m-%d %H:%M:%S')
commit_msg="[$datetime] Newest Version Updated"

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
else
    echo "⚠️ Push bị từ chối. Thử tự động pull --rebase..."
    git pull origin "$branch_name" --rebase

    echo "🔁 Push lại sau khi rebase..."
    if git push origin "$branch_name"; then
        echo "✅ Push thành công sau khi rebase"
    else
        echo "❌ Push vẫn thất bại. Hãy kiểm tra conflict hoặc dùng:"
        echo "   git push origin $branch_name --force"
    fi
fi
