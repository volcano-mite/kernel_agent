# Read each JSONL record separately so a partial final record can be retried
# on the next dashboard refresh.
def activity:
  .type as $event | (.item // {}) as $item |
  if $event == "error" or $event == "turn.failed" then
    ["Error", (.error.message // .message // "turn failed")]
  elif $event == "item.completed" and $item.type == "agent_message" then
    ["Codex", $item.text]
  elif $item.type == "command_execution" then
    if $event == "item.started" then ["Command", $item.command]
    elif $event == "item.completed" then
      ["Exit \($item.exit_code)", $item.aggregated_output]
    else empty end
  elif $item.type == "mcp_tool_call" then
    if $event == "item.started" then ["Tool started", $item.tool]
    elif $event == "item.completed" then
      ($item.result.structured_content // {}) as $r |
      ["Tool returned", ([$item.tool, ($r.status // $item.status), $r.stage,
        (if $r.score then "score=\($r.score)" else null end),
        $r.detail, $item.error.message] | map(select(. != null)) | join(" "))]
    else empty end
  elif $event == "assistant" then
    .message.content[]? |
    if .type == "text" then ["Claude", .text]
    elif .type == "tool_use" then ["Tool started", .name]
    else empty end
  elif $event == "user" then
    .message.content[]? | select(.type == "tool_result") |
    ["Tool returned", (if .is_error then "error" else "completed" end)]
  elif $event == "turn.started" or $event == "turn.completed" then
    ["Session", $event]
  else empty end;

fromjson? | objects | activity |
(.[1] // "" | tostring | if $compact == "yes" then gsub("[[:cntrl:]]+"; " ")
 else split("\n") | map(gsub("[[:cntrl:]]+"; " ")) | join("\n") end) as $text |
"\(.[0]) | \(if $compact == "yes" then $text[:140] else $text end)"
