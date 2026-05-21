"""
validate_replication.py — Vérifie la fidélité de la réplication.

1) Reconstruit le facteur TSMOM (toutes classes + par classe) ;
2) le compare au facteur TSMOM OFFICIEL d'AQR (aqr_tsmom_factors.csv) ;
3) vérifie l'identité comptable de la décomposition Lo-MacKinlay (Table 5B) ;
4) vérifie l'identité Futures = Spot + Roll (Table 6).

Usage :
    TSMOM_DATA=data/data.xlsx TSMOM_DATA_DIR=data python validate_replication.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

from src.data_loader import load_raw
from src.returns import build_daily_excess_returns, daily_to_monthly_returns
from src.volatility import ewma_ex_ante_vol, vol_for_signal
from src.strategy import tsmom_instrument_returns, diversified_tsmom, tsmom_by_asset_class
from src.crosssectional import decomposition_by_asset_class
from src.rollyield import spot_roll_monthly
from src.config import LOOKBACK_MONTHS

S, E = "1985-01-31", "2009-12-31"
prices = load_raw()
daily = build_daily_excess_returns(prices)
monthly = daily_to_monthly_returns(daily)
mvol = vol_for_signal(ewma_ex_ante_vol(daily), monthly.index)
inst = tsmom_instrument_returns(monthly, mvol, k=LOOKBACK_MONTHS)
mine_all = diversified_tsmom(inst)
mine_ac = tsmom_by_asset_class(inst)

m = (mine_all.index >= S) & (mine_all.index <= E)
tp = mine_all.loc[m].dropna()
print("="*64)
print(f"1) TSMOM diversifié 1985-2009 : Sharpe={tp.mean()/tp.std()*np.sqrt(12):.2f}  "
      f"vol={tp.std()*np.sqrt(12):.1%}  (papier : SR>1, vol~12%)")

aqr_path = Path("data/external/aqr_tsmom_factors.csv")
if not aqr_path.exists(): aqr_path = Path("aqr_tsmom_factors.csv")
if aqr_path.exists():
    aqr = pd.read_csv(aqr_path); aqr.columns = ["date","ALL","CM","EQ","FI","FX"]
    aqr["date"] = pd.to_datetime(aqr["date"]); aqr = aqr.set_index("date")
    aqr.index = aqr.index + pd.offsets.MonthEnd(0)
    def corr(a, b):
        df = pd.concat([a.rename("m"), b.rename("a")], axis=1).dropna()
        df = df.loc[(df.index>=S)&(df.index<=E)]
        return df["m"].corr(df["a"])
    print("\n2) Corrélation avec le facteur TSMOM officiel AQR :")
    print(f"   ALL={corr(mine_all,aqr['ALL']):.3f}  CM={corr(mine_ac['Commodity'],aqr['CM']):.3f}  "
          f"EQ={corr(mine_ac['Equity'],aqr['EQ']):.3f}  "
          f"FI={corr(mine_ac['Bond'],aqr['FI']):.3f}  FX={corr(mine_ac['Currency'],aqr['FX']):.3f}")
else:
    print("\n2) aqr_tsmom_factors.csv introuvable — comparaison AQR sautée.")

dec = decomposition_by_asset_class(monthly.loc[m])
err = (dec["XS_Total(sum)"] - dec["XS_Total(emp)"]).abs().max()
print(f"\n3) Décomposition Lo-MacKinlay 12->1 : identité Auto+Cross+Mean=empirique, "
      f"écart max={err*1e4:.2f} bp (doit être ~0)")

comp = spot_roll_monthly(prices)
idn = (comp["total"] - comp["spot"] - comp["roll"]).abs().max().max()
print(f"4) Décomposition Futures=Spot+Roll : écart max={idn:.1e} (doit être 0)")
print("="*64)
