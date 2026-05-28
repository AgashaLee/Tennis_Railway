"""
Tennis match predictor — surface-aware Elo, ATP + WTA, web dashboard.

Pure standard library (no numpy/pandas/flask).

Sibling of the football predictor: SAME infrastructure (dashboard, background
builder, caching, odds value-layer, prediction track record) but a tennis-
appropriate model (Elo blended overall + per-surface), since Poisson/goals
makes no sense for tennis.

Data:
  - Results/training:  ./data/<tour>_matches_<year>.csv  (run fetch_tennis.py)
                       Jeff Sackmann schema (winner_name/loser_name/surface/...)
  - Upcoming + market: The Odds API (tennis_atp / tennis_wta) — also doubles
                       as the upcoming-fixtures source (no free tennis schedule
                       API otherwise). Put the key in odds_api_key.txt.

Run:  python tennis_predictor.py   ->  http://localhost:8801

Honest note (unchanged from football): surface Elo is a decent tennis model
but does NOT reliably beat the betting market. Insight tool, not profit.
"""

import csv
import glob
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

CONFIG = {
    "data_dir":            "data",

    # Tours (switchable in the dashboard, like football leagues).
    # The Odds API has NO generic tennis_atp/tennis_wta key — tennis is
    # listed per active tournament (e.g. tennis_atp_french_open). Active
    # keys for each tour's prefix are discovered dynamically from /sports.
    "tours": [
        {"id": "atp", "name": "ATP (Men)",   "prefix": "atp"},
        {"id": "wta", "name": "WTA (Women)", "prefix": "wta"},
    ],

    # Surface-aware Elo
    "elo_start":           1500.0,
    "elo_k":               32.0,    # tennis commonly uses a higher K than team sport
    "surface_weight":      0.60,    # blend: 0.6*surface_elo + 0.4*overall_elo
    "year_regress":        0.80,    # pull toward 1500 between calendar years
    "recent_surface_n":    200,     # infer current surface from last N matches

    # The Odds API (upcoming fixtures + market). Key: env ODDS_API_KEY >
    # odds_api_key.txt > CONFIG. Only 2 sports here so quota use is tiny.
    "odds_api_key":        "",     # set ODDS_API_KEY env var on Railway
    "odds_base":           "https://api.the-odds-api.com/v4",
    "odds_regions":        "uk",
    # Tennis is per-tournament so a build can hit several sport keys. 24h
    # cache keeps the shared 500/mo Odds API quota safe alongside football.
    "odds_cache_minutes":  1440,    # 24h
    "odds_sports_cache_minutes": 720,   # cache the /sports discovery 12h
    # VALUE = model and market AGREE on the winner AND |edge| <= band.
    # i.e. both sides see the match the same way, within a tight margin.
    "value_edge_band":     0.05,

    # Refresh current-year results + rebuild predictions this often.
    "refresh_minutes":     60,
    "results_refresh_hours": 12,    # re-download current-year CSVs this often
    # Sackmann stamps every match with the TOURNAMENT START date, not the
    # match date, so a result can be dated up to ~2 weeks before we logged
    # the pick. Allow that window when matching (Grand Slams span 2 weeks).
    "resolve_grace_days":  16,

    "dashboard_port":      8801,
}


def _resolve_data_dir(cfg):
    name = cfg["data_dir"]
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (name, os.path.join(here, name),
                 os.path.join(here, "..", name)):
        if os.path.isdir(cand):
            cfg["data_dir"] = os.path.abspath(cand)
            return
    cfg["data_dir"] = os.path.join(here, name)


_resolve_data_dir(CONFIG)
_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "tennis_dashboard.html")


# ═══════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════

def _surface(s):
    s = (s or "").strip().capitalize()
    if s in ("Hard", "Clay", "Grass"):
        return s
    return "Hard"      # Carpet / blank -> treat as Hard


class TMatch:
    __slots__ = ("date", "year", "surface", "winner", "loser")

    def __init__(self, date, year, surface, winner, loser):
        self.date = date
        self.year = year
        self.surface = surface
        self.winner = winner
        self.loser = loser


def load_matches(cfg, prefix):
    pat = os.path.join(cfg["data_dir"], f"{prefix}_matches_*.csv")
    out = []
    for path in sorted(glob.glob(pat)):
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                w = (row.get("winner_name") or "").strip()
                l = (row.get("loser_name") or "").strip()
                if not w or not l:
                    continue
                raw = (row.get("tourney_date") or "").strip()
                try:
                    dt = datetime.strptime(raw, "%Y%m%d")
                except ValueError:
                    continue
                out.append(TMatch(dt, dt.year, _surface(row.get("surface")),
                                   w, l))
    out.sort(key=lambda m: m.date)
    return out


# ═══════════════════════════════════════════════════════════
# SURFACE-AWARE ELO
# ═══════════════════════════════════════════════════════════

class SurfaceElo:
    """
    Two rating tables: overall, and per-surface. Match win probability uses a
    blend (surface_weight). Both tables update after every match; ratings are
    regressed toward the mean at each new calendar year.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.ovr = {}
        self.surf = {}            # key: (player, surface)
        self._year = None

    def _ro(self, p):
        return self.ovr.get(p, self.cfg["elo_start"])

    def _rs(self, p, s):
        # New surface for a player starts from their overall rating.
        return self.surf.get((p, s), self._ro(p))

    def _blend(self, p, s):
        w = self.cfg["surface_weight"]
        return w * self._rs(p, s) + (1 - w) * self._ro(p)

    def prob(self, a, b, s):
        """P(a beats b) on surface s, from pre-match ratings."""
        ra, rb = self._blend(a, s), self._blend(b, s)
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def _regress(self, year):
        if self._year is not None and year != self._year:
            k, base = self.cfg["year_regress"], self.cfg["elo_start"]
            for d in (self.ovr, self.surf):
                for key in d:
                    d[key] = base + k * (d[key] - base)
        self._year = year

    def update(self, m: TMatch):
        self._regress(m.year)
        K = self.cfg["elo_k"]
        s = m.surface
        # Overall
        eo = 1.0 / (1.0 + 10 ** ((self._ro(m.loser) - self._ro(m.winner))
                                 / 400.0))
        do = K * (1.0 - eo)
        self.ovr[m.winner] = self._ro(m.winner) + do
        self.ovr[m.loser] = self._ro(m.loser) - do
        # Surface
        es = 1.0 / (1.0 + 10 ** ((self._rs(m.loser, s) - self._rs(m.winner, s))
                                 / 400.0))
        ds = K * (1.0 - es)
        self.surf[(m.winner, s)] = self._rs(m.winner, s) + ds
        self.surf[(m.loser, s)] = self._rs(m.loser, s) - ds

    def fit(self, matches):
        for m in matches:
            self.update(m)
        return self


def current_surface(cfg, matches):
    """Tennis runs in surface 'swings' — assume upcoming matches are on the
    surface most common among the most recent N results."""
    if not matches:
        return "Hard"
    recent = matches[-cfg["recent_surface_n"]:]
    counts = {}
    for m in recent:
        counts[m.surface] = counts.get(m.surface, 0) + 1
    return max(counts, key=counts.get)


# ═══════════════════════════════════════════════════════════
# BACKTEST (binary: did the model favour the actual winner?)
# ═══════════════════════════════════════════════════════════

def backtest(cfg, matches):
    if not matches:
        return {"n": 0}, None
    last_year = matches[-1].year
    elo = SurfaceElo(cfg)
    n = hits = 0
    ll = br = 0.0
    for m in matches:
        if m.year == last_year:
            p = elo.prob(m.winner, m.loser, m.surface)   # P(actual winner)
            n += 1
            if p >= 0.5:
                hits += 1
            p = min(max(p, 1e-12), 1.0)
            ll += -math.log(p)
            br += (1.0 - p) ** 2
        elo.update(m)
    if not n:
        return {"n": 0}, last_year
    return ({"n": n, "accuracy": round(hits / n, 4),
             "log_loss": round(ll / n, 4), "brier": round(br / n, 4)},
            last_year)


# ═══════════════════════════════════════════════════════════
# THE ODDS API  (upcoming fixtures + market)
# ═══════════════════════════════════════════════════════════

def _clean_key(s):
    """Strip whitespace, surrounding quotes and stray non-key chars."""
    s = s.strip().strip('"').strip("'").strip()
    # An Odds API key is hex-ish; drop anything that isn't a key char.
    return "".join(c for c in s if c.isalnum())


def _read_key_file(path):
    """Read a key file tolerant of encoding (UTF-8, UTF-8/16 BOM from
    PowerShell Set-Content). Decodes by BOM sniffing, not a fixed codec."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16", "ignore")
    elif raw[:3] == b"\xef\xbb\xbf":
        text = raw.decode("utf-8-sig", "ignore")
    else:
        text = raw.decode("utf-8", "ignore")
    return _clean_key(text)


def load_odds_key(cfg):
    env = os.environ.get("ODDS_API_KEY", "")
    if env.strip():
        return _clean_key(env)
    # Look next to the script too, not just the CWD.
    here = os.path.dirname(os.path.abspath(__file__))
    for path in ("odds_api_key.txt", os.path.join(here, "odds_api_key.txt")):
        try:
            k = _read_key_file(path)
            if k:
                return k
        except FileNotFoundError:
            continue
    return _clean_key(cfg.get("odds_api_key") or "")


_GENERIC = {"jr", "ii", "iii"}


def _norm(name):
    out = []
    for tok in str(name).lower().replace(".", " ").replace("-", " ").split():
        tok = "".join(c for c in tok if c.isalnum())
        if tok and tok not in _GENERIC:
            out.append(tok)
    return out


def _sim(a, b):
    ta, tb = _norm(a), _norm(b)
    if not ta or not tb:
        return 0.0
    short, lng = (ta, tb) if len(ta) <= len(tb) else (tb, ta)

    def hit(t):
        return any(t == u or u.startswith(t) or t.startswith(u)
                   for u in lng if min(len(t), len(u)) >= 2)
    return sum(1 for t in short if hit(t)) / len(short)


class OddsClient:
    def __init__(self, cfg, key, sport):
        self.cfg, self.key, self.sport = cfg, key, sport

    def _path(self):
        return os.path.join(self.cfg["data_dir"], f"odds_{self.sport}.json")

    def _fresh(self, p):
        try:
            return (datetime.now().timestamp() - os.path.getmtime(p)) / 60.0 \
                < self.cfg["odds_cache_minutes"]
        except OSError:
            return False

    def events(self):
        p = self._path()
        if self._fresh(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, ValueError):
                pass
        url = (f"{self.cfg['odds_base']}/sports/{self.sport}/odds/"
               f"?apiKey={self.key}&regions={self.cfg['odds_regions']}"
               f"&markets=h2h&oddsFormat=decimal")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tp"})
            with urllib.request.urlopen(req, timeout=30) as r:
                left = r.headers.get("x-requests-remaining")
                data = json.loads(r.read())
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f)
            if left is not None:
                print(f"  The Odds API ({self.sport}): {left} req left")
            return data
        except Exception as e:
            if os.path.exists(p):
                try:
                    with open(p, encoding="utf-8") as f:
                        print(f"  odds error ({self.sport}) — cached")
                        return json.load(f)
                except (OSError, ValueError):
                    pass
            print(f"  odds {self.sport}: {repr(e)[:120]}")
            return []


def active_sport_keys(cfg, key, prefix):
    """Discover currently-active Odds API sport keys for a tour, e.g.
    prefix 'atp' -> ['tennis_atp_french_open', 'tennis_atp_hamburg_open'].
    The /sports list is cached (it changes only as tournaments start/end)."""
    cache = os.path.join(cfg["data_dir"], "odds_sports.json")
    sports = None
    try:
        age = (datetime.now().timestamp() - os.path.getmtime(cache)) / 60.0
        if age < cfg["odds_sports_cache_minutes"]:
            with open(cache, encoding="utf-8") as f:
                sports = json.load(f)
    except (OSError, ValueError):
        sports = None
    if sports is None:
        url = f"{cfg['odds_base']}/sports/?apiKey={key}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tp"})
            with urllib.request.urlopen(req, timeout=30) as r:
                sports = json.loads(r.read())
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(sports, f)
        except Exception as e:
            if os.path.exists(cache):
                try:
                    with open(cache, encoding="utf-8") as f:
                        sports = json.load(f)
                except (OSError, ValueError):
                    sports = []
            else:
                print(f"  /sports discovery failed: {repr(e)[:100]}")
                sports = []
    pre = f"tennis_{prefix}_"
    return [s["key"] for s in sports
            if s.get("active") and str(s.get("key", "")).startswith(pre)]


def _market_two(event):
    """De-vigged P for the two players, averaged across bookmakers."""
    h, a = event.get("home_team"), event.get("away_team")
    accH = accA = 0.0
    books = 0
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk.get("key") != "h2h":
                continue
            pr = {}
            for oc in mk.get("outcomes", []):
                nm, price = oc.get("name"), oc.get("price")
                if not price or price <= 1.0:
                    continue
                if nm == h:
                    pr["H"] = 1.0 / price
                elif nm == a:
                    pr["A"] = 1.0 / price
            if {"H", "A"} <= pr.keys():
                s = pr["H"] + pr["A"]
                accH += pr["H"] / s
                accA += pr["A"] / s
                books += 1
    if not books:
        return None
    return {"home": accH / books, "away": accA / books, "books": books}


# ═══════════════════════════════════════════════════════════
# PREDICTION TRACK RECORD (forward test)
# ═══════════════════════════════════════════════════════════

# Bump when the meaning of the `value` flag changes — triggers a one-time
# re-evaluation of every existing log entry under the new rule on next start.
_VALUE_RULE_VERSION = 2


def _log_path(cfg):
    return os.path.join(cfg["data_dir"], "tennis_predictions_log.json")


def _load_log(cfg):
    try:
        with open(_log_path(cfg), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_log(cfg, log):
    try:
        with open(_log_path(cfg), "w", encoding="utf-8") as f:
            json.dump(log, f)
    except OSError as e:
        print(f"  log save failed: {repr(e)[:100]}")


def _pair_key(tour, a, b):
    x, y = sorted([a, b])
    return f"{tour}|{x}|{y}"


def _migrate_value_flag(cfg):
    """One-time re-evaluation of `value` on every existing log entry under
    the current rule (uses already-stored market/edge/pick). Idempotent — a
    version stamp in log['_meta'] gates it so it runs exactly once per rule
    change, then never again. Other code paths ignore '_meta' because it has
    no 'status' field."""
    log = _load_log(cfg)
    meta = log.get("_meta") or {}
    if meta.get("value_rule_version") == _VALUE_RULE_VERSION:
        return 0
    band = cfg["value_edge_band"]
    changed = 0
    for k, r in log.items():
        if k == "_meta" or not isinstance(r, dict):
            continue
        mk = r.get("market")
        edge = r.get("edge")
        if not mk or edge is None:
            new_val = False
        else:
            market_fav = r["a"] if mk["a"] >= mk["b"] else r["b"]
            new_val = (r.get("pick") == market_fav) and (abs(edge) <= band)
        if bool(r.get("value")) != new_val:
            r["value"] = new_val
            changed += 1
    log["_meta"] = {"value_rule_version": _VALUE_RULE_VERSION,
                    "migrated_at": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S")}
    _save_log(cfg, log)
    print(f"  value-rule migration: re-flagged {changed} entries "
          f"(rule v{_VALUE_RULE_VERSION})")
    return changed


def _log_predictions(cfg, tour, preds):
    log = _load_log(cfg)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    added = 0
    for p in preds:
        key = _pair_key(tour, p["player_a"], p["player_b"])
        if key in log:
            continue
        log[key] = {
            "tour": tour, "a": p["player_a"], "b": p["player_b"],
            "pick": p["pick"], "p_pick": p["p_pick"],
            "market": p.get("market"), "edge": p.get("edge"),
            "value": bool(p.get("value")), "surface": p["surface"],
            "start": p.get("start"),
            "logged_at": now,
            "logged_day": datetime.now().strftime("%Y-%m-%d"),
            "status": "pending",
        }
        added += 1
    if added:
        _save_log(cfg, log)
    return added


_NAME_TH = 0.6      # fuzzy name-match threshold (same as odds->Elo matching)


def _resolve(cfg, tour, matches):
    """Score pending predictions against the first FUTURE meeting.

    Predictions are logged under The Odds API's name spelling but results
    come from Sackmann's — the two differ constantly (accents, middle names,
    abbreviations). So we fuzzy-match the player pair (via _sim) instead of an
    exact key, in either orientation, and only accept a result dated on/after
    the day the prediction was logged (ignoring earlier meetings of the pair).
    """
    log = _load_log(cfg)
    pend = [(k, r) for k, r in log.items()
            if r.get("status") == "pending" and r.get("tour") == tour]
    if not pend:
        return 0

    grace = timedelta(days=cfg.get("resolve_grace_days", 16))

    # Prefilter: results no earlier than (earliest pending log day - grace),
    # because Sackmann dates a match by its TOURNAMENT START, which can be
    # well before the pick was logged.
    floor = None
    for _k, r in pend:
        try:
            d = datetime.strptime(r.get("logged_day", ""), "%Y-%m-%d")
            floor = d if floor is None else min(floor, d)
        except ValueError:
            pass
    if floor is not None:
        floor = floor - grace
    cand = [m for m in matches if floor is None or m.date >= floor]

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    changed = 0
    for _k, rec in pend:
        try:
            logged = datetime.strptime(rec["logged_day"], "%Y-%m-%d")
        except (ValueError, KeyError):
            continue
        # Anchor on the match's scheduled start if we have it, else log day.
        anchor = logged
        st = rec.get("start")
        if st:
            try:
                anchor = datetime.strptime(st[:10], "%Y-%m-%d")
            except ValueError:
                pass
        window_floor = logged - grace

        best, best_gap = None, None
        for m in cand:
            if m.date < window_floor:
                continue
            fwd = min(_sim(a := rec["a"], m.winner), _sim(b := rec["b"],
                                                          m.loser))
            rev = min(_sim(rec["a"], m.loser), _sim(rec["b"], m.winner))
            if max(fwd, rev) < _NAME_TH:
                continue
            gap = abs((m.date - anchor).days)     # closest tournament wins
            if best is None or gap < best_gap:
                best, best_gap, best_fwd = m, gap, (fwd >= rev)
        if best is None:
            continue
        winner_side = rec["a"] if best_fwd else rec["b"]
        rec["winner"] = winner_side
        rec["winner_src"] = best.winner
        rec["correct"] = (rec["pick"] == winner_side)
        rec["resolved_at"] = now
        rec["status"] = "resolved"
        changed += 1
    if changed:
        _save_log(cfg, log)
    return changed


def results_summary(cfg):
    log = _load_log(cfg)
    recs = list(log.values())
    res = [r for r in recs if r.get("status") == "resolved"]
    pend = sum(1 for r in recs if r.get("status") == "pending")

    def acc(rows):
        return round(sum(1 for r in rows if r.get("correct")) / len(rows), 4) \
            if rows else 0

    s = {"resolved": len(res), "pending": pend, "accuracy": acc(res)}
    if res:
        ll = 0.0
        for r in res:
            p = r.get("p_pick", 0.5) if r.get("correct") else 1 - r.get("p_pick", 0.5)
            ll += -math.log(min(max(p, 1e-9), 1.0))
        s["log_loss"] = round(ll / len(res), 4)
        wm = [r for r in res if r.get("market")]
        if wm:
            mk = 0
            for r in wm:
                m = r["market"]
                mpick = r["a"] if m["a"] >= m["b"] else r["b"]
                mk += (mpick == r["winner"])
            s["with_market"] = len(wm)
            s["model_acc_vs_market"] = acc(wm)
            s["market_acc"] = round(mk / len(wm), 4)
        vb = [r for r in res if r.get("value")]
        if vb:
            s["value_bets"] = len(vb)
            s["value_acc"] = acc(vb)
    recent = sorted(res, key=lambda r: r.get("resolved_at", ""),
                    reverse=True)[:80]
    return {"summary": s, "recent": recent}


# ═══════════════════════════════════════════════════════════
# OPTIONAL: refresh current-year results in-process
# ═══════════════════════════════════════════════════════════

def refresh_current_year(cfg):
    """Keep the track record working without manual fetches: re-download the
    current-year CSV per tour if older than results_refresh_hours."""
    yr = datetime.now().year
    repos = {"atp": "tennis_atp", "wta": "tennis_wta"}
    for t in cfg["tours"]:
        path = os.path.join(cfg["data_dir"],
                            f"{t['prefix']}_matches_{yr}.csv")
        try:
            age_h = (datetime.now().timestamp()
                     - os.path.getmtime(path)) / 3600.0
            if age_h < cfg["results_refresh_hours"]:
                continue
        except OSError:
            pass
        url = (f"https://raw.githubusercontent.com/JeffSackmann/"
               f"{repos[t['prefix']]}/master/{t['prefix']}_matches_{yr}.csv")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tp"})
            data = urllib.request.urlopen(req, timeout=30).read()
            if data.startswith(b"tourney_id"):
                with open(path, "wb") as f:
                    f.write(data)
                print(f"  refreshed {t['prefix']} {yr} results")
        except Exception as e:
            print(f"  refresh {t['prefix']} {yr}: {repr(e)[:90]}")


# ═══════════════════════════════════════════════════════════
# BUILD
# ═══════════════════════════════════════════════════════════

def _build_tour(cfg, tour):
    matches = load_matches(cfg, tour["prefix"])
    if not matches:
        return None
    surface = current_surface(cfg, matches)
    elo = SurfaceElo(cfg).fit(matches)

    bt, bt_year = backtest(cfg, matches)

    preds = []
    key = load_odds_key(cfg)
    odds_status = "no key - odds disabled"
    if key:
        sport_keys = active_sport_keys(cfg, key, tour["prefix"])
        events = []
        for sk in sport_keys:
            try:
                events.extend(OddsClient(cfg, key, sk).events())
            except Exception as e:
                print(f"  {sk}: {repr(e)[:80]}")
        tourn = ", ".join(s.replace(f"tennis_{tour['prefix']}_", "")
                          for s in sport_keys) or "none active"
        if events:
            band = cfg["value_edge_band"]
            known = set(elo.ovr) | {p for (p, _s) in elo.surf}
            matched = 0
            for ev in events:
                a = ev.get("home_team")
                b = ev.get("away_team")
                if not a or not b:
                    continue
                # Map odds names to our rating names (closest known player).
                ra = max(known, key=lambda k: _sim(a, k), default=None)
                rb = max(known, key=lambda k: _sim(b, k), default=None)
                if not ra or not rb or _sim(a, ra) < 0.6 or _sim(b, rb) < 0.6:
                    continue
                pa = elo.prob(ra, rb, surface)
                pick = a if pa >= 0.5 else b
                p_pick = pa if pa >= 0.5 else 1 - pa
                rec = {"player_a": a, "player_b": b, "surface": surface,
                       "p_a": round(pa, 3), "p_b": round(1 - pa, 3),
                       "pick": pick, "p_pick": round(p_pick, 3),
                       # ISO8601 UTC kickoff from The Odds API; the dashboard
                       # renders it in the viewer's local time.
                       "start": ev.get("commence_time")}
                mp = _market_two(ev)
                if mp:
                    matched += 1
                    mk_pick_p = mp["home"] if pick == a else mp["away"]
                    edge = p_pick - mk_pick_p
                    market_fav = a if mp["home"] >= mp["away"] else b
                    rec["market"] = {"a": round(mp["home"], 3),
                                     "b": round(mp["away"], 3),
                                     "books": mp["books"]}
                    rec["edge"] = round(edge, 3)
                    # Agreement pick: model & market favour the same player AND
                    # they're within band (default ±5%) of each other.
                    rec["value"] = (pick == market_fav) and (abs(edge) <= band)
                preds.append(rec)
            odds_status = (f"{tourn}: matched {matched}/{len(preds)}"
                           if preds else f"{tourn}: no priced matches")
        else:
            odds_status = f"active: {tourn} — no events returned"

    # Soonest match first (matches with no time go last).
    preds.sort(key=lambda r: r.get("start") or "9999")
    _log_predictions(cfg, tour["id"], preds)
    _resolve(cfg, tour["id"], matches)

    return {
        "id": tour["id"], "name": tour["name"],
        "meta": {
            "surface_assumed": surface,
            "trained_matches": len(matches),
            "years": sorted({m.year for m in matches}),
            "backtest_year": bt_year,
            "odds_status": odds_status,
        },
        "predictions": preds,
        "backtest": bt,
    }


def build_payload(cfg):
    refresh_current_year(cfg)
    tours = []
    for t in cfg["tours"]:
        try:
            blk = _build_tour(cfg, t)
        except Exception as e:
            print(f"  {t['id']}: build failed {repr(e)[:120]}")
            blk = None
        if blk:
            tours.append(blk)
            print(f"  built {t['id']}: {len(blk['predictions'])} matches, "
                  f"{blk['meta']['odds_status']}")
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tours": tours,
        "results": results_summary(cfg),
    }


# ═══════════════════════════════════════════════════════════
# SERVER (background builder + instant-serve cache)
# ═══════════════════════════════════════════════════════════

_CACHE = {"payload": None, "built_at": None}
_LOCK = threading.Lock()


def _builder_loop(cfg):
    while True:
        try:
            t0 = time.time()
            print(f"[builder] rebuilding {len(cfg['tours'])} tours...")
            payload = build_payload(cfg)
            with _LOCK:
                _CACHE["payload"] = payload
                _CACHE["built_at"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")
            print(f"[builder] done in {time.time() - t0:.0f}s")
        except Exception as e:
            print(f"[builder] error: {repr(e)[:160]}")
        time.sleep(max(cfg["refresh_minutes"], 5) * 60)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/data"):
            with _LOCK:
                payload, built = _CACHE["payload"], _CACHE["built_at"]
            if payload is None:
                body = json.dumps({"status": "warming up",
                                   "tours": []}).encode()
            else:
                body = json.dumps(dict(payload, built_at=built)).encode()
            self._send(200, body, "application/json")
            return
        try:
            with open(_HTML_PATH, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            body = b"<h1>tennis_dashboard.html missing</h1>"
        self._send(200, body, "text/html; charset=utf-8")


def main():
    if not glob.glob(os.path.join(CONFIG["data_dir"], "*_matches_*.csv")):
        raise SystemExit("No data. Run: python fetch_tennis.py")
    _migrate_value_flag(CONFIG)
    threading.Thread(target=_builder_loop, args=(CONFIG,),
                     daemon=True).start()
    # Railway sets $PORT; locally fall back to CONFIG. Bind 0.0.0.0 so the
    # platform's router can reach the container (127.0.0.1 is loopback-only).
    port = int(os.environ.get("PORT") or CONFIG["dashboard_port"])
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"Tennis predictor -> http://{host}:{port}")
    print("First build runs in the background (~10-20s). 'warming up' "
          "until ready. Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
