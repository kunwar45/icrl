#!/usr/bin/env bash
# Configure upstream + fork remotes for third-party dependency repos.
#
# Prerequisite: fork these repos on GitHub first (one-time, in the browser):
#   https://github.com/ServiceNow/BrowserGym        → github.com/YOUR_USER/BrowserGym
#   https://github.com/segev-shlomov/ST-WebAgentBench → github.com/YOUR_USER/ST-WebAgentBench
#
# Usage:
#   export GITHUB_USER=kunwar45          # your GitHub username
#   export REPOS_ROOT=$HOME                # parent dir for BrowserGym + ST-WebAgentBench
#   bash scripts/setup_fork_remotes.sh
#
# After patching a dependency, push to your fork:
#   cd $REPOS_ROOT/ST-WebAgentBench
#   git checkout -b icrl/patch-description
#   git add -A && git commit -m "fix: ..."
#   git push -u fork HEAD
set -euo pipefail

GITHUB_USER="${GITHUB_USER:-}"
REPOS_ROOT="${REPOS_ROOT:-$HOME}"

if [ -z "$GITHUB_USER" ]; then
    echo "ERROR: Set GITHUB_USER to your GitHub username."
    echo "  export GITHUB_USER=kunwar45"
    exit 1
fi

configure_repo() {
    local name="$1"
    local upstream_url="$2"
    local fork_url="git@github.com:${GITHUB_USER}/${name}.git"
    local path="${REPOS_ROOT}/${name}"

    echo ""
    echo "=== ${name} ==="

    if [ ! -d "$path/.git" ]; then
        echo "Cloning your fork → $path"
        git clone "$fork_url" "$path"
    else
        echo "Repo exists: $path"
    fi

    cd "$path"

    if ! git remote get-url upstream &>/dev/null; then
        git remote add upstream "$upstream_url"
        echo "Added upstream: $upstream_url"
    else
        git remote set-url upstream "$upstream_url"
        echo "Updated upstream: $upstream_url"
    fi

    if ! git remote get-url fork &>/dev/null; then
        git remote add fork "$fork_url"
        echo "Added fork: $fork_url"
    else
        git remote set-url fork "$fork_url"
        echo "Updated fork: $fork_url"
    fi

    # Keep origin pointing at your fork (standard fork workflow)
    git remote set-url origin "$fork_url"
    echo "origin → $fork_url"
    git remote -v
}

configure_repo "BrowserGym" "https://github.com/ServiceNow/BrowserGym.git"
configure_repo "ST-WebAgentBench" "https://github.com/segev-shlomov/ST-WebAgentBench.git"

echo ""
echo "Done. To sync with upstream later:"
echo "  cd \$REPOS_ROOT/BrowserGym && git fetch upstream && git checkout main && git merge upstream/main"
echo "  cd \$REPOS_ROOT/ST-WebAgentBench && git fetch upstream && git checkout main && git merge upstream/main"
