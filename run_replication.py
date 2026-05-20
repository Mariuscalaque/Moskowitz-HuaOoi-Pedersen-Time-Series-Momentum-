#!/usr/bin/env python3
"""
run_replication.py — Lance la réplication COMPLÈTE (6 tables + 7 figures) de
Moskowitz, Hua Ooi & Pedersen (2012), « Time Series Momentum ».

    python run_replication.py                 # échantillon papier 1985-2009
    python run_replication.py --no-external   # sans téléchargements externes
    python run_replication.py --end 2025-12-31  # échantillon étendu

Les sorties vont dans outputs/tables/ (CSV + Markdown) et outputs/figures/ (PNG).
Les données externes (AQR, Pástor-Stambaugh, Baker-Wurgler, CFTC) sont
téléchargées puis mises en cache dans data/external/ (voir data/external/_manifest.json).
"""
import argparse
from src.pipeline import run
from src.config import PAPER_START, PAPER_END

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=PAPER_START)
    ap.add_argument("--end", default=PAPER_END)
    ap.add_argument("--no-external", action="store_true",
                    help="n'essaie pas de télécharger les données externes")
    args = ap.parse_args()
    run(start=args.start, end=args.end, use_external=not args.no_external)
