#!/usr/bin/env bash
# Publish the prototype on a public HTTPS URL and print a QR code for phones.
#
#   ./phone_link.sh            start a tunnel, print URLs, stay attached
#   ./phone_link.sh watch      SUPERVISOR: restart + re-publish automatically when the
#                              tunnel goes away (run this during a demo)
#   ./phone_link.sh status     show the current URL and whether it answers
#   ./phone_link.sh stop       kill the tunnel
#
# Why `watch` exists: Cloudflare quick tunnels are ephemeral and have no uptime
# guarantee. The connection can drop while cloudflared keeps running - the process
# looks healthy, the re-register even succeeds, but the hostname is left serving
# "HTTP 530 / error code 1033". Nothing on the phone explains that; the page just
# fails to load. `watch` probes the public URL end to end and republishes.
set -uo pipefail
cd "$(dirname "$0")"                       # repo root is the script's directory

PORT=${PORT:-8000}
BIN=${CLOUDFLARED:-/tmp/cloudflared}
LOG=${TUNNEL_LOG:-/tmp/tunnel.log}
URLF=${TUNNEL_URL_FILE:-/tmp/tunnel_url}
PIDF=${TUNNEL_PID_FILE:-/tmp/cloudflared.pid}
PROBE_EVERY=${PROBE_EVERY:-25}              # seconds between health probes
DEAD_AFTER=${DEAD_AFTER:-2}                 # consecutive failures before republishing
GRACE_AFTER_REPUBLISH=${GRACE_AFTER_REPUBLISH:-4}  # extra misses tolerated right after a
                                                   # republish, to cover edge propagation                 # consecutive failures before republishing
MODE=${1:-start}

api_up()    { curl -sf -m 3 "http://localhost:$PORT/api/health" >/dev/null; }
url_now()   { cat "$URLF" 2>/dev/null || true; }

# Probe the PUBLIC url, not localhost - that is the path that breaks.
# -4 : this sandbox has no working IPv6 egress. trycloudflare.com returns AAAA
#      records, curl tries them first, each attempt burns the timeout, and a
#      perfectly healthy tunnel reads as dead. Symptom if you omit -4: the URL
#      republishes every probe interval while everything is actually fine.
# status code, not curl's exit code: when a quick tunnel loses its origin
#      connection Cloudflare answers HTTP 530 (browser shows "error 1033"), which
#      curl happily reports as a successful fetch of an error page.
probe_code() {
  local u c
  u=$(url_now)
  [ -z "$u" ] && { echo 000; return; }
  # curl prints 000 itself when the transfer fails, so do NOT also `|| echo 000` -
  # that produced the nonsense "HTTP 000000" in the supervisor log.
  c=$(curl -4 -s -o /dev/null -w '%{http_code}' -m "${1:-12}" --retry 1 --retry-delay 2 \
        "$u/api/health" 2>/dev/null)
  echo "${c:-000}"
}
tunnel_ok() { [ "$(probe_code "${1:-12}")" = "200" ]; }

ensure_binary() {
  [ -x "$BIN" ] && return 0
  echo "downloading cloudflared (~40 MB) ..."
  curl -sL -o "$BIN" \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    && chmod +x "$BIN"
}

regen_qr() {
  local u=$1
  [ -n "$u" ] || return 0
  # stdout must stay empty: start_tunnel returns the URL on stdout, so progress
  # text goes to stderr or it gets glued onto the URL
  python3 tools/make_qr.py "${u%/}/#capture" "web/phone-qr.png" "Scan with your phone" >&2
}

stop_tunnel() {
  if [ -f "$PIDF" ]; then
    local p; p=$(cat "$PIDF" 2>/dev/null)
    # never pkill by pattern: the pattern also matches the shell that runs it
    [ -n "${p:-}" ] && kill "$p" 2>/dev/null
    rm -f "$PIDF"
  fi
}

start_tunnel() {
  api_up || {
    echo "! API not running on :$PORT - start it first:" >&2
    echo "    python -m uvicorn vitalguard.server:app --host 0.0.0.0 --port $PORT &" >&2
    return 1
  }
  stop_tunnel; sleep 1
  : > "$LOG"
  echo "starting tunnel ..." >&2
  # http2 over TCP, not QUIC/UDP: the failure we hit was a QUIC idle timeout
  # ("timeout: no recent network activity") that left the hostname 530 forever.
  "$BIN" tunnel --no-autoupdate --protocol http2 --url "http://localhost:$PORT" > "$LOG" 2>&1 &
  echo $! > "$PIDF"

  local u="" i
  for i in $(seq 1 40); do
    u=$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | tail -1)
    [ -n "$u" ] && break
    sleep 1
  done
  [ -z "$u" ] && { echo "! no URL obtained; see $LOG" >&2; return 1; }
  echo "$u" > "$URLF"

  # the edge can answer 530 for a few seconds after the URL appears - wait for real
  for i in $(seq 1 30); do tunnel_ok 12 && break; sleep 4; done
  if tunnel_ok 10; then echo "  edge is serving 200" >&2
  else echo "  ! edge not answering yet (code $(probe_code 10)) - the supervisor will keep retrying" >&2; fi
  regen_qr "$u"
  echo "$u"
}

print_urls() {
  local u=$1
  echo
  echo "  phone  -> $u/#capture     (record + score on the phone)"
  echo "  landing-> $u/phone        (QR + instructions for the team)"
  echo "  doctor -> $u/#ward"
  echo "  api    -> $u/docs"
  echo
  echo "camera note: HTTPS is present, so getUserMedia + torch work on Android Chrome."
  echo "iOS Safari can capture but has no torch API - the flash button stays disabled there."
  echo "public + unauthenticated: anyone with the link can post readings. Kill it with"
  echo "  ./phone_link.sh stop"
}

case "$MODE" in
  start)
    ensure_binary || exit 1
    u=$(start_tunnel) || exit 1
    print_urls "$u"
    echo
    echo "  (tunnel pid $(cat "$PIDF")) - keep this terminal open, or use './phone_link.sh watch'"
    wait "$(cat "$PIDF")"
    ;;

  watch)
    ensure_binary || exit 1
    u=$(start_tunnel) || exit 1
    print_urls "$u"
    echo
    echo "supervising: probing $(url_now)/api/health every ${PROBE_EVERY}s, republishing after $DEAD_AFTER misses"
    fails=0
    grace=0
    # one restart per pass, and `continue` right after it. The earlier version let the
    # probe branch and the process-exited branch both fire in the same iteration: each
    # minted a fresh quick-tunnel hostname, so the QR ended up encoding the URL from
    # the first restart while the file held the second - a printed QR that points at
    # a dead host, which is precisely the bug this script exists to prevent.
    republish() {
      local nu
      nu=$(start_tunnel)
      if [ -n "${nu:-}" ]; then
        echo "  [$(date +%H:%M:%S)] NEW URL: $nu"
        print_urls "$nu"
        fails=0
        grace=$GRACE_AFTER_REPUBLISH
      else
        echo "  [$(date +%H:%M:%S)] republish failed; retrying in ${PROBE_EVERY}s"
      fi
    }
    while :; do
      sleep "$PROBE_EVERY"
      if ! kill -0 "$(cat "$PIDF" 2>/dev/null || echo 0)" 2>/dev/null; then
        echo "  [$(date +%H:%M:%S)] cloudflared exited - republishing"
        republish
        continue
      fi
      code=$(probe_code)
      if [ "$code" = "200" ]; then fails=0; grace=0; continue; fi
      fails=$((fails + 1))
      # A brand-new quick-tunnel hostname needs ~10-30 s at the Cloudflare edge before
      # it answers anything (curl sees a connection failure, code 000). Without this
      # grace window the supervisor reads that warm-up as a second failure and mints
      # yet another URL, so the link changes under the person holding the QR code.
      need=$((DEAD_AFTER + grace))
      echo "  [$(date +%H:%M:%S)] probe got HTTP $code ($fails/$need)"
      if [ "$fails" -ge "$need" ]; then
        echo "  tunnel is not serving - republishing"
        republish
      fi
    done
    ;;

  status)
    u=$(url_now)
    if [ -z "$u" ]; then echo "no tunnel recorded in $URLF"; exit 1; fi
    code=$(probe_code 12)
    echo "  url    $u"
    case "$code" in
      200)  verdict="serving" ;;
      530)  verdict="NOT serving - Cloudflare error 1033: tunnel registered, origin connection dead." ;;
      000)  verdict="NOT serving - no HTTP answer at all (network/DNS path from this machine)." ;;
      *)    verdict="NOT serving (HTTP $code)" ;;
    esac
    echo "  health HTTP $code  $verdict"
    pid=$(cat "$PIDF" 2>/dev/null || echo "")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "  pid    $pid alive"; else echo "  pid    not running"; fi
    ;;

  stop)
    stop_tunnel; echo "tunnel stopped"
    ;;

  *)
    echo "usage: $0 [start|watch|status|stop]"; exit 2
    ;;
esac
