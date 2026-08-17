#!/usr/bin/env bash
# ABOUTME: Resets SuiteCRM's demo records to a known state so generated trajectories are attributable
# ABOUTME: Run on a login node: bash scripts/reseed_suitecrm_demo_data.sh  (needs .env sourced; then re-run the wizard script)
#
# WHY this is mandatory between generation passes: a trajectory is only kept when
# the database proves the change persisted AND was not already true beforehand
# (see src/trajectory_collection/stwebagentbench_state_verifier.py). Successful
# episodes mutate shared state — deletes remove their target, meetings pile up
# duplicates — so a second pass over a dirty database can neither succeed
# honestly nor fail honestly.
#
# It also repairs a defect in the benchmark's own seed: demo_data.sql inserts
# staff users with `status=1` and no user_hash, which SuiteCRM rejects, so those
# users never existed. Every seeded record therefore had a NULL assignee and
# task 245 ("reassign to asmith") was impossible. Users are created FIRST here so
# demo_data.sql's `(SELECT id FROM users WHERE user_name=...)` lookups resolve.
#
# Usage:
#   set -a; source .env; set +a
#   bash scripts/reseed_suitecrm_demo_data.sh                   # everything
#   RESEED_TABLES=leads,opportunities bash scripts/reseed_suitecrm_demo_data.sh
#   PYTHONPATH=. python scratch/complete_wizard_via_ui.py   # login wizard, if it returns
#
# RESEED_TABLES restores only the records a cycle actually consumed. A full
# reseed truncates fifteen tables and replays the whole demo seed; a cycle whose
# tasks only touch leads and opportunities does not need the other thirteen, and
# at a few minutes a cycle that cost real time once the episodes themselves got
# faster. Each name is EXPANDED to its related tables (an email lives in
# `emails` plus three relationship tables), because a partial restore that
# leaves relationship rows behind is worse than no optimisation at all — the
# drift is invisible until a check starts failing for no visible reason.
# Leave RESEED_TABLES unset for the full, always-correct reseed.
set -euo pipefail

DB_HOST="${ICRL_SUITECRM_DB_HOST:-klogin01}"
DB_PORT="${ICRL_SUITECRM_DB_PORT:-3306}"
DB_NAME="${ICRL_SUITECRM_DB_NAME:-bitnami_suitecrm}"
DB_USER="${ICRL_SUITECRM_DB_USER:-bn_suitecrm}"
DB_PASSWORD="${ICRL_SUITECRM_DB_PASSWORD:?set ICRL_SUITECRM_DB_PASSWORD in .env}"
MARIADB_SIF="${MARIADB_SIF:-/project/aip-s2ganapa/kunwar/apptainer/mariadb.sif}"
SEED_SQL="${SEED_SQL:-${STWEBAGENT_ROOT:-/project/aip-s2ganapa/kunwar/ST-WebAgentBench}/suitecrm_setup/init-db/demo_data.sql}"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

command -v apptainer >/dev/null 2>&1 || module load apptainer/1.4.5 >/dev/null 2>&1 || true

run_sql() {  # run_sql <file> <label>
    echo "  applying $2..."
    apptainer exec "${MARIADB_SIF}" mysql \
        -h "${DB_HOST}" -P "${DB_PORT}" -u "${DB_USER}" -p"${DB_PASSWORD}" \
        "${DB_NAME}" < "$1"
}

# ── 1. Decide which tables to restore ─────────────────────────────────────────
ALL_TABLES="accounts accounts_contacts accounts_opportunities calls cases
contacts email_addr_bean_rel email_addresses emails emails_beans leads meetings
meetings_contacts meetings_users opportunities"

# A record is never alone: deleting an email leaves rows in emails_beans and
# email_addr_bean_rel, and restoring only `emails` would accumulate orphans
# forever. Naming any member of a group restores the whole group.
_expand_table_group() {
    case "$1" in
        leads)                          echo "leads" ;;
        opportunities|accounts_opportunities)
                                        echo "opportunities accounts_opportunities" ;;
        cases)                          echo "cases" ;;
        emails|emails_beans)            echo "emails emails_beans email_addresses email_addr_bean_rel" ;;
        meetings|meetings_contacts|meetings_users)
                                        echo "meetings meetings_contacts meetings_users" ;;
        contacts|accounts|accounts_contacts)
                                        echo "accounts contacts accounts_contacts email_addresses email_addr_bean_rel" ;;
        calls)                          echo "calls" ;;
        *)                              echo "" ;;
    esac
}

if [ -n "${RESEED_TABLES:-}" ]; then
    TABLES=""
    for _requested in ${RESEED_TABLES//,/ }; do
        _group="$(_expand_table_group "${_requested}")"
        if [ -z "${_group}" ]; then
            echo "ERROR: RESEED_TABLES names '${_requested}', which is not a demo table." >&2
            echo "       Known: ${ALL_TABLES}" >&2
            exit 2
        fi
        TABLES="${TABLES} ${_group}"
    done
    # Deduplicate while keeping the set stable for the log line below.
    TABLES="$(echo "${TABLES}" | tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' ')"
    echo "Targeted reseed — restoring only: ${TABLES}"
else
    TABLES="$(echo "${ALL_TABLES}" | tr -s ' \n' ' ')"
fi

# ── 2. Wipe the demo records in scope, and anything the agent runs created ────
# `users` is preserved (admin + the benchmark login live there); only demo
# business records and the relationship rows pointing at them are cleared.
{
    echo "SET FOREIGN_KEY_CHECKS=0;"
    for _table in ${TABLES}; do echo "TRUNCATE TABLE ${_table};"; done
    echo "SET FOREIGN_KEY_CHECKS=1;"
} > "${WORK}/01_truncate.sql"

# ── 3. Staff users the seed references ────────────────────────────────────────
# The benchmark's own users INSERT omits `deleted`, so its rows land with
# deleted=NULL and are invisible to every `deleted=0` query — SuiteCRM's
# included. Rather than guess at repairing them, delete every row for these
# demo names and write exactly one clean row each, so the seed's
# `(SELECT id FROM users WHERE user_name=...)` lookups resolve to one row.
# `admin` and the benchmark login (`user`) are never touched.
cat > "${WORK}/02_users.sql" <<'SQL'
DELETE FROM users WHERE user_name IN
    ('jdoe','asmith','bjones','cjames','dwilson','emiller','fgarcia','fharris','gharris','hlee');

INSERT INTO users (id, user_name, user_hash, first_name, last_name,
                   status, is_admin, employee_status, deleted,
                   date_entered, date_modified)
VALUES
    ('demo-user-jdoe',   'jdoe',   '', 'John',  'Doe',    'Active', 0, 'Active', 0, NOW(), NOW()),
    ('demo-user-asmith', 'asmith', '', 'Alice', 'Smith',  'Active', 0, 'Active', 0, NOW(), NOW()),
    ('demo-user-bjones', 'bjones', '', 'Bob',   'Jones',  'Active', 0, 'Active', 0, NOW(), NOW()),
    ('demo-user-cjames', 'cjames', '', 'Carol', 'James',  'Active', 0, 'Active', 0, NOW(), NOW()),
    ('demo-user-dwilson','dwilson','', 'David', 'Wilson', 'Active', 0, 'Active', 0, NOW(), NOW()),
    ('demo-user-emiller','emiller','', 'Eve',   'Miller', 'Active', 0, 'Active', 0, NOW(), NOW()),
    ('demo-user-fgarcia','fgarcia','', 'Frank', 'Garcia', 'Active', 0, 'Active', 0, NOW(), NOW()),
    -- 'fharris' is referenced by the seed's Stark Industries row but appears in
    -- no users INSERT; created so that lookup resolves instead of yielding NULL.
    ('demo-user-fharris','fharris','', 'Fiona', 'Harris', 'Active', 0, 'Active', 0, NOW(), NOW()),
    ('demo-user-gharris','gharris','', 'Grace', 'Harris', 'Active', 0, 'Active', 0, NOW(), NOW()),
    ('demo-user-hlee',   'hlee',   '', 'Henry', 'Lee',    'Active', 0, 'Active', 0, NOW(), NOW());
SQL

# ── 4. The benchmark's own demo records, minus its broken users statement ─────
# Statements are skipped whole (from `INSERT INTO x` to the line ending in `;`),
# because the seed's inserts span many lines. `users` is always skipped — step 3
# above replaces it — and on a targeted reseed every table outside the scope is
# skipped too, so untouched records are left exactly as they are rather than
# being duplicated by a second insert.
awk -v allowed="${TABLES}" '
  BEGIN { n = split(allowed, names, " "); for (i = 1; i <= n; i++) in_scope[names[i]] = 1 }
  /^INSERT INTO[ \t]+`?[A-Za-z0-9_]+/ {
      table = $0
      sub(/^INSERT INTO[ \t]+`?/, "", table)
      sub(/`?[ \t(].*$/, "", table)
      skipping = (table == "users") || !(table in in_scope)
  }
  skipping { if (/;[[:space:]]*$/) skipping = 0; next }
  { print }
' "${SEED_SQL}" > "${WORK}/03_demo_records.sql"

if ! grep -qE "^INSERT INTO" "${WORK}/03_demo_records.sql"; then
    echo "ERROR: stripped seed lost every record insert — refusing to reseed" >&2
    echo "       (scope was: ${TABLES})" >&2
    exit 1
fi

# ── 5. Verify the resulting state is the one the state checks expect ──────────
# Counts cover every demo table, not just the ones in scope: on a targeted
# reseed the out-of-scope numbers are exactly how drift becomes visible.
cat > "${WORK}/04_verify.sql" <<'SQL'
SELECT 'contacts' t, COUNT(*) n FROM contacts WHERE deleted=0
UNION ALL SELECT 'leads', COUNT(*) FROM leads WHERE deleted=0
UNION ALL SELECT 'opportunities', COUNT(*) FROM opportunities WHERE deleted=0
UNION ALL SELECT 'meetings', COUNT(*) FROM meetings WHERE deleted=0
UNION ALL SELECT 'cases', COUNT(*) FROM cases WHERE deleted=0
UNION ALL SELECT 'accounts', COUNT(*) FROM accounts WHERE deleted=0
UNION ALL SELECT 'emails', COUNT(*) FROM emails WHERE deleted=0
UNION ALL SELECT 'accounts_contacts', COUNT(*) FROM accounts_contacts WHERE deleted=0
UNION ALL SELECT 'meetings_contacts', COUNT(*) FROM meetings_contacts WHERE deleted=0
UNION ALL SELECT 'users', COUNT(*) FROM users WHERE deleted=0
UNION ALL SELECT 'records_with_assignee', COUNT(*) FROM contacts
          WHERE deleted=0 AND assigned_user_id IS NOT NULL;
-- Targets the state checks require to START unsatisfied:
SELECT 'michael_scott_present' probe, COUNT(*) n FROM contacts
  WHERE first_name='Michael' AND last_name='Scott' AND deleted=0
UNION ALL SELECT 'bruce_wayne_present', COUNT(*) FROM leads
  WHERE first_name='Bruce' AND last_name='Wayne' AND deleted=0
UNION ALL SELECT 'analytics_opp_present', COUNT(*) FROM opportunities
  WHERE name='Data Analytics Implementation' AND deleted=0
UNION ALL SELECT 'closed_lost_opps', COUNT(*) FROM opportunities
  WHERE sales_stage='Closed Lost' AND deleted=0
UNION ALL SELECT 'leads_status_new', COUNT(*) FROM leads
  WHERE status='New' AND deleted=0
UNION ALL SELECT 'asmith_exists', COUNT(*) FROM users
  WHERE user_name='asmith' AND deleted=0;
SQL

echo "Reseeding SuiteCRM demo data at ${DB_HOST}:${DB_PORT}/${DB_NAME}"
run_sql "${WORK}/01_truncate.sql"      "truncate of demo records"
run_sql "${WORK}/02_users.sql"         "demo staff users"
run_sql "${WORK}/03_demo_records.sql"  "benchmark demo records"
echo "Resulting state:"
run_sql "${WORK}/04_verify.sql"        "verification"
echo
echo "Reseed complete. SuiteCRM caches list views per user, so also run:"
echo "  PYTHONPATH=. python scratch/complete_wizard_via_ui.py"
