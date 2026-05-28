"""Download ATP + WTA match results from Jeff Sackmann's GitHub datasets.

Re-run any time to refresh (the current-year file updates through the season).
Files land in ./data/<tour>_matches_<year>.csv
Schema: tourney_date(YYYYMMDD), surface, winner_name, loser_name, score, ...
"""
import os
import urllib.request

RAW = "https://raw.githubusercontent.com/JeffSackmann/{repo}/master/{tour}_matches_{yr}.csv"
HDRS = {"User-Agent": "Mozilla/5.0"}

TOURS = [("atp", "tennis_atp"), ("wta", "tennis_wta")]
YEARS = [2022, 2023, 2024, 2025, 2026]   # 2026 may be partial/empty early on


def main():
    os.makedirs("data", exist_ok=True)
    ok = 0
    for tour, repo in TOURS:
        for yr in YEARS:
            url = RAW.format(repo=repo, tour=tour, yr=yr)
            out = f"data/{tour}_matches_{yr}.csv"
            try:
                req = urllib.request.Request(url, headers=HDRS)
                data = urllib.request.urlopen(req, timeout=30).read()
                # A missing year returns GitHub's 404 HTML, not CSV — skip it.
                if not data.startswith(b"tourney_id"):
                    print(f"skip {tour} {yr} (not available yet)")
                    continue
                with open(out, "wb") as f:
                    f.write(data)
                rows = data.count(b"\n")
                print(f"OK   {out}  {rows} matches")
                ok += 1
            except Exception as e:
                print(f"FAIL {url}  {repr(e)[:120]}")
    print(f"\n{ok} files downloaded into ./data/")


if __name__ == "__main__":
    main()
