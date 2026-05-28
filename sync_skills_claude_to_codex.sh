#!/bin/bash
# sync_skills.sh - 把 ~/.claude/skills 同步到 ~/.codex/skills

CLAUDE_SKILLS="$HOME/.claude/skills"
CODEX_SKILLS="$HOME/.codex/skills"

if [ ! -d "$CLAUDE_SKILLS" ]; then
    echo "FAIL: $CLAUDE_SKILLS 不存在"
    exit 1
fi

mkdir -p "$CODEX_SKILLS"

for skill_dir in "$CLAUDE_SKILLS"/*/; do
    skill_name=$(basename "$skill_dir")
    src="$skill_dir/SKILL.md"
    dst="$CODEX_SKILLS/$skill_name"

    if [ ! -f "$src" ]; then
        echo "SKIP [$skill_name] 没有 SKILL.md"
        continue
    fi

    mkdir -p "$dst"
    cp "$src" "$dst/SKILL.md"
    echo "OK   [$skill_name] -> $dst"
done

echo -e "\nDone."
