#!/bin/bash
# ============================================================================
# cluster_probe.sh — report what THIS cluster actually offers
# ============================================================================
#
# Run this first, on a login node, before submitting anything. Every value the
# submission needs (account, GPU type, partition, filesystem, module versions)
# differs between Alliance clusters, and guessing wrong costs a queue cycle
# each time.
#
# Read-only: it submits nothing and writes nothing outside a single optional
# summary file.
#
#   bash scripts/infra/cluster_probe.sh
#   bash scripts/infra/cluster_probe.sh --save     # also write cluster_probe.txt
#
# Feed the results into scripts/infra/submit_experiment.sh via ICRL_ACCOUNT /
# ICRL_GPU / ICRL_PARTITION, or export them in ~/.bashrc.
# ============================================================================

set -uo pipefail

SAVE=0
[ "${1:-}" = "--save" ] && SAVE=1

out() { printf '%s\n' "$*"; }
hdr() { out ""; out "── $* ──────────────────────────────────────────" ; }

probe() {
out "============================================================"
out " cluster probe   host=$(hostname)   user=${USER}   $(date -u +%FT%TZ)"
out "============================================================"

hdr "Cluster identity"
out "CC_CLUSTER      : ${CC_CLUSTER:-<unset>}"
out "hostname        : $(hostname -f 2>/dev/null || hostname)"
[ -r /etc/os-release ] && out "os              : $(. /etc/os-release; echo "$PRETTY_NAME")"

hdr "Accounts you can charge (use one of these for --account)"
if command -v sshare >/dev/null 2>&1; then
    sshare -U -u "$USER" --format=Account,User,RawShares -P 2>/dev/null \
        | awk -F'|' 'NR>1 && $1 !~ /^root/ {print "  " $1}' | sort -u
elif command -v sacctmgr >/dev/null 2>&1; then
    sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | sort -u | sed 's/^/  /'
else
    out "  (no slurm accounting tools found)"
fi
out "  NOTE: GPU work usually needs the 'aip-*' (AI) or 'def-*' allocation."

hdr "Partitions"
if command -v sinfo >/dev/null 2>&1; then
    sinfo -o "%20P %10a %10l %6D %N" 2>/dev/null | head -25
else
    out "  (sinfo unavailable)"
fi

hdr "GPU types available (use for --gpus-per-node=<type>:<n>)"
if command -v sinfo >/dev/null 2>&1; then
    sinfo -o "%20P %40G %6D" 2>/dev/null | grep -i gpu | head -20
    out ""
    out "  distinct types:"
    sinfo -h -o "%G" 2>/dev/null | tr ',' '\n' | grep -oE 'gpu:[a-z0-9_]+' \
        | sort -u | sed 's/^/    /'
else
    out "  (sinfo unavailable)"
fi

hdr "Filesystems and quota"
for var in HOME SCRATCH PROJECT; do
    path="${!var:-}"
    out "\$$var = ${path:-<unset>}"
    [ -n "${path}" ] && [ -d "${path}" ] && out "    writable: $([ -w "${path}" ] && echo yes || echo NO)"
done
if command -v diskusage_report >/dev/null 2>&1; then
    out ""
    diskusage_report 2>/dev/null | head -15
fi

hdr "Modules (what the pipeline wants)"
for mod in python apptainer cuda gcc arrow; do
    out "${mod}:"
    (module -t spider "$mod" 2>&1 || module -t avail "$mod" 2>&1) \
        | grep -iE "^${mod}/" | sort -V | tail -6 | sed 's/^/    /'
done

hdr "Internet reachability from THIS node"
if command -v curl >/dev/null 2>&1; then
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 12 https://huggingface.co 2>/dev/null)
    out "  huggingface.co : HTTP ${code:-<no response>}"
    out "  (compute nodes are usually offline — prefetch models here, on the login node)"
else
    out "  (curl unavailable)"
fi

hdr "Suggested settings"
acct=$( { sshare -U -u "$USER" --format=Account -P 2>/dev/null | awk -F'|' 'NR>1 && $1 !~ /^root/ {print $1}' \
          || sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null; } \
        | sort -u | grep -E '^(aip|def|rrg)-' | head -1)
# Prefer the GPU type with the most nodes: queue time dominates, and the biggest
# card is usually the scarcest. Anything in this pipeline fits a 48GB card.
gputype=$(sinfo -h -o "%G|%D" 2>/dev/null \
          | awk -F'|' '{n=$2; while (match($1, /gpu:[a-z0-9_]+/)) {
                          t=substr($1, RSTART+4, RLENGTH-4); tot[t]+=n;
                          $1=substr($1, RSTART+RLENGTH) } }
                       END { for (t in tot) if (t !~ /^[0-9]+$/) print tot[t], t }' \
          | sort -rn | head -1 | awk '{print $2}')
out "  export ICRL_ACCOUNT=${acct:-<pick from the account list above>}"
out "  export ICRL_GPU=${gputype:-<pick from the GPU types above>}:1"
[ -n "${gputype}" ] && out "     (most available type — swap for another if you need more VRAM)"
out "  export SCRATCH=${SCRATCH:-/scratch/\$USER}"
out ""
out "submit_experiment.sh picks the partition automatically from the GPU type"
out "and --time, so you only set ICRL_PARTITION to override it."
out ""
out "Then: bash scripts/infra/submit_experiment.sh --stages preflight,splits,encode,constraint,gate,plots"
out "============================================================"
}

if [ "$SAVE" = "1" ]; then
    probe | tee cluster_probe.txt
    echo
    echo "Saved: $(pwd)/cluster_probe.txt"
else
    probe
fi
