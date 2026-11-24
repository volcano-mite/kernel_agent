#!/bin/sh
set -eu

once=0
agent_only=0
usage='Usage: sh scripts/watch_run.sh [--once | --agent] RUN_DIR'
case "${1:-}" in
    --once) once=1; shift ;;
    --agent) agent_only=1; shift ;;
    -h|--help)
        printf '%s\n' "$usage"
        exit 0 ;;
esac

if [ "$#" -ne 1 ]; then
    printf '%s\n' "$usage" >&2
    exit 2
fi

run_dir=$1
if [ ! -f "$run_dir/run.json" ]; then
    printf 'Run directory missing or invalid: %s\n' "$run_dir" >&2
    exit 1
fi

events="$run_dir/agent/agent.stdout.jsonl"
filter="$(dirname "$0")/agent_activity.jq"
if [ "$agent_only" -eq 1 ]; then
    if [ -f "$run_dir/agent/result.json" ]; then
        exec jq -Rr --arg compact no -f "$filter" "$events"
    fi
    printf '%s\n' 'Following Agent events (Ctrl+C to exit)' >&2
    tail -n +1 -F "$events" | jq --unbuffered -Rr --arg compact no -f "$filter"
    exit
fi

if [ "$once" -eq 0 ]; then
    exec watch -n 1 -x sh "$0" --once "$run_dir"
fi

printf '%s\n' 'Recent Agent activity'
if [ -s "$events" ]; then
    tail -n 40 "$events" | jq -Rr --arg compact yes -f "$filter" | tail -n 6
else
    printf '%s\n' 'Waiting for Agent events'
fi
printf '\n'
printf '%s\n' 'Public evaluations'
printf 'Candidate\tStatus\tStage\tScore\tFeedback\n'
set -- "$run_dir"/evaluations/*.json
if [ -f "$1" ]; then
    jq -rs '
        sort_by(.started_at)[] |
        [.candidate_id[:8], .status, (.stage // "-"), (.score // "-"),
         ((.detail // "-")[:80])] | @tsv
    ' "$@"
else
    printf '%s\n' 'Waiting for the first evaluation'
fi

printf '\n%s\n' 'Agent summary'
if [ -f "$run_dir/agent/result.json" ]; then
    jq '{status, duration_seconds, usage}' "$run_dir/agent/result.json"
else
    printf '%s\n' 'Awaiting Agent summary'
fi

printf '\n%s\n' 'Acceptance'
if [ "$(jq -r .backend "$run_dir/run.json")" = mock ]; then
    printf '%s\n' 'CPU Mock run'
elif [ -f "$run_dir/acceptance/decision.json" ]; then
    jq '{status, candidate_id, speedup}' "$run_dir/acceptance/decision.json"
elif [ -f "$run_dir/acceptance/shortlist.json" ]; then
    printf '%s\n' 'Shortlist frozen, awaiting a decision'
else
    printf '%s\n' 'Awaiting acceptance'
fi
