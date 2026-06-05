$env:PYTHONPATH = "C:\Projects\Algo-Bot-Trader;C:\Projects\Algo-Bot-Trader\src;C:\Projects\Algo-Bot-Trader\src\data;C:\Projects\Algo-Bot-Trader\src\scanner;C:\Projects\Algo-Bot-Trader\src\analysis;C:\Projects\Algo-Bot-Trader\src\setups;C:\Projects\Algo-Bot-Trader\src\scoring;C:\Projects\Algo-Bot-Trader\src\risk;C:\Projects\Algo-Bot-Trader\src\execution;C:\Projects\Algo-Bot-Trader\src\learning;C:\Projects\Algo-Bot-Trader\src\frontend"

@'
from src.bot_runner import run_once

result = run_once()

print()
print("=" * 76)
print("ALGO BOT TRADER - ONE CYCLE SUMMARY")
print("=" * 76)

status = result.get("status")
errors = result.get("errors", []) or []
warnings = result.get("warnings", []) or []

scanner_candidates = result.get("scanner_candidates", []) or []
ranked_candidates = result.get("ranked_candidates", []) or []
decisions = result.get("decisions", []) or []
orders = result.get("orders", []) or []

print(f"Status:             {status}")
print(f"Scanner Candidates: {len(scanner_candidates)}")
print(f"Ranked Candidates:  {len(ranked_candidates)}")
print(f"Trade Decisions:    {len(decisions)}")
print(f"Orders:             {len(orders)}")
print(f"Warnings:           {len(warnings)}")
print(f"Errors:             {len(errors)}")

print()
print("-" * 76)
print("TOP SCANNER CANDIDATES")
print("-" * 76)

if not scanner_candidates:
    print("No scanner candidates.")
else:
    for i, c in enumerate(scanner_candidates[:20], start=1):
        ticker = getattr(c, "ticker", str(c))
        price = getattr(c, "price", "-")
        rvol = getattr(c, "relative_volume", "-")
        change = getattr(c, "day_change_pct", "-")
        spread = getattr(c, "spread_percent", "-")
        reason = getattr(c, "candidate_reason", "")

        print(
            f"{i:>2}. {ticker:<7} "
            f"price={price:<9} "
            f"rvol={rvol:<6} "
            f"change={change:<8}% "
            f"spread={spread:<8}% "
            f"{reason}"
        )

print()
print("-" * 76)
print("RANKED CANDIDATES")
print("-" * 76)

if not ranked_candidates:
    print("No ranked candidates.")
else:
    for i, item in enumerate(ranked_candidates[:20], start=1):
        candidate = getattr(item, "candidate", item)
        ticker = getattr(candidate, "ticker", str(candidate))
        score = getattr(item, "rank_score", "-")
        reasons = getattr(item, "rank_reasons", [])
        reason_text = ", ".join(reasons) if reasons else ""

        print(f"{i:>2}. {ticker:<7} score={score:<8} {reason_text}")

print()
print("-" * 76)
print("TRADE DECISIONS")
print("-" * 76)

if not decisions:
    print("No trade decisions.")
else:
    for i, d in enumerate(decisions, start=1):
        ticker = getattr(d, "ticker", "-")
        decision = getattr(d, "decision", "-")
        setup = getattr(d, "setup", "-")
        scores = getattr(d, "scores", None)

        final_score = getattr(scores, "final_trade_quality_score", "-") if scores else "-"
        setup_score = getattr(scores, "setup_score", "-") if scores else "-"
        prob_score = getattr(scores, "probability_score", "-") if scores else "-"
        rr_score = getattr(scores, "risk_reward_score", "-") if scores else "-"

        print(f"{i}. {ticker} - {decision}")
        print(f"   Setup: {setup}")
        print(f"   Scores: final={final_score}, setup={setup_score}, probability={prob_score}, risk/reward={rr_score}")

        reasons = getattr(d, "reasons", []) or []
        if reasons:
            print("   Reasons:")
            for reason in reasons[:5]:
                print(f"     - {reason}")
        print()

print("-" * 76)
print("ORDERS")
print("-" * 76)

if not orders:
    print("No orders submitted.")
else:
    for i, order in enumerate(orders, start=1):
        print(f"{i}. {order}")

if warnings:
    print()
    print("-" * 76)
    print("WARNINGS")
    print("-" * 76)
    for w in warnings:
        print(f"- {w}")

if errors:
    print()
    print("-" * 76)
    print("ERRORS")
    print("-" * 76)
    for e in errors:
        print(f"- {e}")

print()
print("=" * 76)
print("Cycle complete.")
print("=" * 76)
'@ | python -

