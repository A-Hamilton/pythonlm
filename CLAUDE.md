# CLAUDE.md

Once you recieve a system warning about context do the following;

1. Save current state to `.claude/continuation.md`:
   - What we accomplished
   - Current task status  
   - Key decisions
   - Next steps

2. Update CLAUDE.md with any new learnings

3. Output exactly:
```
   CONTEXT - Almost Full, Restart needed
   State saved to .claude/continuation.md
```

On session start: If `.claude/continuation.md` exists, read it immediately.

CRITICAL: Update this file every prompt. Add new information you learn about the project, preferences, decisions, and solutions. Remove or condense anything outdated or no longer relevant. Keep it short and current. 
ALWAYS KEEP EVERY ABOVE THIS LINE IN THE FILE. 

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a new Python project. Update this section as the project develops.

## Commands

```bash
# Install dependencies (update when requirements.txt or pyproject.toml is added)
pip install -r requirements.txt

# Run tests (update based on test framework used)
pytest

# Run the application (update with actual entry point)
python main.py
```

## Architecture

Document the high-level architecture here as the project grows.
