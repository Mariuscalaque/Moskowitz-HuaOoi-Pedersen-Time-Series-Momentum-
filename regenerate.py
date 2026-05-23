#!/usr/bin/env python3
"""
regenerate.py — Régénération PROPRE et COHÉRENTE de tous les outputs.

Pourquoi ce script ? `performance_summary.csv` et `diversified_tsmom_series.csv`
étaient autrefois produits par le notebook, séparément des tables produites par
le pipeline -> deux sources de vérité, donc des CSV potentiellement périmés et
incohérents entre eux (ex. Sharpe 0.90 dans la série sauvegardée vs 1.13 dans
le code actuel).

Ce script :
  1) SUPPRIME entièrement le dossier outputs/ (table + figures + perf),
  2) relance le pipeline complet en UNE passe,
de sorte que TOUS les fichiers proviennent du même run et du même code.

Usage :
    python regenerate.py                  # 1985-2009 (échantillon papier), avec externes
    python regenerate.py --no-external     # sans téléchargements (caches/coeur seulement)
    python regenerate.py --end 2025-12-31  # échantillon étendu
"""
import argparse
import shutil
from src.config import OUTPUT_DIR, PAPER_START, PAPER_END
from src.pipeline import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=PAPER_START)
    ap.add_argument("--end", default=PAPER_END)
    ap.add_argument("--no-external", action="store_true")
    args = ap.parse_args()

    # 1) Table rase : on efface tout outputs/ pour éliminer tout fichier périmé.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        print(f"[clean] {OUTPUT_DIR} supprimé.")

    # 2) Un seul run -> une seule source de vérité (tables, figures, perf, série).
    run(start=args.start, end=args.end, use_external=not args.no_external)
    print("\n[done] Tous les outputs régénérés depuis le code actuel, en une passe.")


if __name__ == "__main__":
    main()
