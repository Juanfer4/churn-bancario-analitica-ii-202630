"""Cifras citadas en el texto del EDA (Fase 2, 2.2 a 2.9, del notebook).

Recalcula cada número con pandas, scipy y sklearn directamente (sin churn.exploracion)
y lo compara con el valor escrito en el markdown, a la precisión con la que se escribió.
Si los datos o el código cambian y alguna cifra del texto deja de ser cierta, esta prueba falla.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from churn import config, datos


def _info(_mensaje):
    pass


def test_cifras_citadas_en_el_eda(cifras):
    df, p, _ = datos.preparar_datos(guardar=False)
    X, y = p.X_train, p.y_train
    chk = cifras.chk

    def auc(col):
        a = roc_auc_score(y, X[col]); return max(a, 1 - a)

    def tasa(mask):
        return 100 * y[mask].mean()

    def wilson(k, n, z=1.959963984540054):
        ph = k / n; d = 1 + z*z/n; c = (ph + z*z/(2*n))/d; m = z*np.sqrt(ph*(1-ph)/n + z*z/(4*n*n))/d
        return 100*(c-m), 100*(c+m)

    # Intro
    chk("train n", len(X), 8101, 0)
    # 2.2
    for col, v in {"total_trans_ct": .796, "total_ct_chng_q4_q1": .748, "avg_utilization_ratio": .694,
                   "total_revolving_bal": .690, "total_trans_amt": .679, "contacts_count_12_mon": .644,
                   "months_inactive_12_mon": .635, "total_relationship_count": .621, "months_on_book": .508,
                   "customer_age": .512, "dependent_count": .525, "credit_limit": .540}.items():
        chk(f"auc {col}", auc(col), v, 0.0005)
    orden = sorted([c for c in X.select_dtypes("number")], key=auc, reverse=True)[:5]
    assert orden == ["total_trans_ct", "total_ct_chng_q4_q1", "avg_utilization_ratio", "total_revolving_bal", "total_trans_amt"], orden
    chk("mediana ct churn", X.loc[y == 1, "total_trans_ct"].median(), 43, 0)
    chk("mediana ct queda", X.loc[y == 0, "total_trans_ct"].median(), 71, 0)
    chk("mediana util churn", X.loc[y == 1, "avg_utilization_ratio"].median(), 0, 0)
    for col in ["total_trans_ct", "total_ct_chng_q4_q1", "avg_utilization_ratio", "total_revolving_bal", "total_trans_amt", "total_relationship_count"]:
        assert roc_auc_score(y, X[col]) < 0.5, col  # más alto, menos churn
    for col in ["contacts_count_12_mon", "months_inactive_12_mon"]:
        assert roc_auc_score(y, X[col]) > 0.5, col
    # 2.3
    for col, v in {"total_ct_chng_q4_q1": 2.25, "total_trans_amt": 2.05, "total_amt_chng_q4_q1": 1.80,
                   "credit_limit": 1.68, "avg_open_to_buy": 1.68}.items():
        chk(f"skew {col}", X[col].skew(), v, 0.005)
    top_skew = X.select_dtypes("number").skew().abs().sort_values(ascending=False).index[:5].tolist()
    assert set(top_skew) == {"total_ct_chng_q4_q1", "total_trans_amt", "total_amt_chng_q4_q1", "credit_limit", "avg_open_to_buy"}, top_skew
    def pct_out(col):
        q1, q3 = X[col].quantile([.25, .75]); i = q3 - q1
        return 100 * ((X[col] < q1 - 1.5*i) | (X[col] > q3 + 1.5*i)).mean(), q3 + 1.5*i
    chk("atip credit", pct_out("credit_limit")[0], 9.68, 0.005)
    chk("atip amt", pct_out("total_trans_amt")[0], 8.69, 0.005)
    cerca_sup_amt = pct_out("total_trans_amt")[1]
    assert cerca_sup_amt < 11000, cerca_sup_amt   # los atípicos altos incluyen el segmento > 11.000
    entre_30_50 = X.loc[y == 1, "total_trans_ct"].between(30, 50).mean()
    _info(f"[info] churners con 30 a 50 transacciones: {entre_30_50:.1%}; clientes que se quedan: {X.loc[y == 0, 'total_trans_ct'].between(30, 50).mean():.1%}")
    assert entre_30_50 > 0.5
    # 2.4
    monto = X["total_trans_amt"]
    for (a, b), v in {(0, 3000): 33.06, (3000, 6000): 2.31, (6000, 11000): 31.23, (11000, 1e9): 0.0}.items():
        m = (monto > a) & (monto <= b) if a > 0 else (monto <= b)
        chk(f"tramo {a}-{b}", tasa(m), v, 0.005)
    m11 = monto > 11000
    chk("n >11000", m11.sum(), 596, 0); chk("churn >11000", y[m11].sum(), 0, 0)
    chk("wilson sup >11000", wilson(0, 596)[1], 0.64, 0.005)
    for (a, b), (vc, vq) in {(3000, 6000): (54, 75), (6000, 11000): (69, 90)}.items():
        m = (monto > a) & (monto <= b)
        chk(f"med churn {a}", X.loc[m & (y == 1), "total_trans_ct"].median(), vc, 0)
        chk(f"med queda {a}", X.loc[m & (y == 0), "total_trans_ct"].median(), vq, 0)
    tk = X["total_trans_amt"] / X["total_trans_ct"]; a = roc_auc_score(y, tk)
    chk("auc ticket", max(a, 1 - a), .509, 0.0005)
    # 2.5
    s = X["total_revolving_bal"]; mx = s.max()
    chk("saldo max", mx, 2517, 0)
    chk("pct saldo 0", 100*(s == 0).mean(), 24.29, 0.005); chk("tasa saldo 0", tasa(s == 0), 36.53, 0.005)
    chk("pct saldo max", 100*(s == mx).mean(), 4.94, 0.005); chk("tasa saldo max", tasa(s == mx), 32.50, 0.005)
    chk("tasa saldo medio", tasa((s > 0) & (s < mx)), 7.90, 0.005); chk("n saldo max", (s == mx).sum(), 400, 0)
    # 2.6
    r = X["total_relationship_count"]
    chk("prod1", tasa(r == 1), 26.78, .005); chk("prod2", tasa(r == 2), 28.62, .005); chk("prod3", tasa(r == 3), 16.83, .005)
    t46 = [tasa(r == k) for k in (4, 5, 6)]; chk("prod4-6 min", min(t46), 10.53, .005); chk("prod4-6 max", max(t46), 11.82, .005)
    lo1, hi1 = wilson(y[r == 1].sum(), (r == 1).sum()); lo2, hi2 = wilson(y[r == 2].sum(), (r == 2).sum())
    assert lo2 < hi1, "los intervalos de 1 y 2 productos deberían solaparse"
    i = X["months_inactive_12_mon"]
    chk("inact1", tasa(i == 1), 4.01, .005); chk("inact4", tasa(i == 4), 31.18, .005)
    chk("inact5", tasa(i == 5), 16.78, .005); chk("inact6", tasa(i == 6), 12.87, .005)
    chk("inact0", tasa(i == 0), 52.38, .005); chk("n inact0", (i == 0).sum(), 21, 0)
    lo, hi = wilson(y[i == 0].sum(), 21); chk("ic inact0 inf", lo, 32.37, .005); chk("ic inact0 sup", hi, 71.66, .005)
    assert min(tasa(i == k) for k in range(7)) == tasa(i == 1)
    c = X["contacts_count_12_mon"]
    tasas_c = [tasa(c == k) for k in range(7)]; assert all(np.diff(tasas_c) > 0), tasas_c
    chk("cont0", tasas_c[0], 1.52, .005); chk("cont5", tasas_c[5], 32.65, .005)
    chk("n cont6", (c == 6).sum(), 42, 0); chk("churn cont6", y[c == 6].sum(), 42, 0)
    chk("ic cont6 inf", wilson(42, 42)[0], 91.62, .005)
    d = X["dependent_count"]; td = [tasa(d == k) for k in range(6)]
    chk("dep min", min(td), 14.01, .005); chk("dep max", max(td), 17.64, .005)
    # 2.7
    def cramer(col):
        ct = pd.crosstab(X[col], y); chi2, pv, _, _ = stats.chi2_contingency(ct)
        return np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))), pv
    vs = {col: cramer(col) for col in datos.COLUMNAS_CATEGORICAS}
    assert max(vs, key=lambda k: vs[k][0]) == "gender"
    chk("V gender", vs["gender"][0], 0.042, .0005)
    assert all(v < 0.1 for v, _ in vs.values())
    chk("p gender", vs["gender"][1], 0.0002, .00005); chk("p income", vs["income_category"][1], 0.039, .0005)
    chk("p edu", vs["education_level"][1], 0.20, .005); chk("p marital", vs["marital_status"][1], 0.13, .005)
    chk("p card", vs["card_category"][1], 0.52, .005)
    assert vs["gender"][1] < .05 and vs["income_category"][1] < .05 and all(vs[k][1] > .05 for k in ["education_level", "marital_status", "card_category"])
    g = X["gender"]; chk("F", tasa(g == "F"), 17.53, .005); chk("M", tasa(g == "M"), 14.44, .005)
    e = X["education_level"]; te = {k: tasa(e == k) for k in e.unique()}
    assert max(te, key=te.get) == "Doctorate" and min(te, key=te.get) == "College"
    chk("doct", te["Doctorate"], 20.51, .005); chk("college", te["College"], 14.82, .005)
    inc = X["income_category"]; chk("<40k", tasa(inc == "Less than $40K"), 17.23, .005); chk("120k", tasa(inc == "$120K +"), 17.86, .005)
    medios = [tasa(inc == k) for k in ["$40K - $60K", "$60K - $80K", "$80K - $120K"]]
    assert max(medios) < min(tasa(inc == "Less than $40K"), tasa(inc == "$120K +"))
    chk("unk edu", te["Unknown"], 16.98, .005); chk("unk mar", tasa(X["marital_status"] == "Unknown"), 17.85, .005)
    chk("unk inc", tasa(inc == "Unknown"), 17.27, .005)
    card = X["card_category"]; chk("n gold", (card == "Gold").sum(), 94, 0); chk("n plat", (card == "Platinum").sum(), 17, 0)
    lo, hi = wilson(y[card == "Platinum"].sum(), 17); chk("plat inf", lo, 9.56, .005); chk("plat sup", hi, 47.26, .005)
    # Regla de 100 clientes: en cada pliegue de CV y en train completo, solo Gold y Platinum quedan por debajo
    skf = StratifiedKFold(config.N_PLIEGUES, shuffle=True, random_state=config.SEMILLA)
    for k, (itr, _) in enumerate(skf.split(X, y)):
        Xf = X.iloc[itr]
        for col in datos.COLUMNAS_CATEGORICAS:
            raras = set(Xf[col].value_counts()[lambda s: s < 100].index)
            esperado = {"Gold", "Platinum"} if col == "card_category" else set()
            assert raras == esperado, (k, col, raras)
    # 2.8
    sp = X.select_dtypes("number").corr("spearman")
    for (a_, b_), v in {("credit_limit", "avg_open_to_buy"): .931, ("total_trans_amt", "total_trans_ct"): .882,
                        ("customer_age", "months_on_book"): .770, ("total_revolving_bal", "avg_utilization_ratio"): .709,
                        ("avg_open_to_buy", "avg_utilization_ratio"): -.684}.items():
        chk(f"sp {a_}-{b_}", sp.loc[a_, b_], v, .0005)
    tri = sp.where(np.triu(np.ones(sp.shape, dtype=bool), 1)).stack()
    assert (tri.abs() >= .5).sum() == 5
    # 2.9
    mob = X["months_on_book"]; chk("n36", (mob == 36).sum(), 1979, 0); chk("pct36", 100*(mob == 36).mean(), 24.43, .005)
    vecinos = [(mob == k).sum() for k in (33, 34, 35, 37, 38, 39)]
    _info(f"[info] clientes con 33-35 y 37-39 meses: {vecinos}")
    assert all(200 <= v <= 320 for v in vecinos), vecinos
    chk("tasa36", tasa(mob == 36), 17.48, .005); chk("resto36", tasa(mob != 36), 15.62, .005)
    ct = pd.crosstab(X["gender"], X["income_category"])
    chk("F>=60k", ct.loc["F", ["$60K - $80K", "$80K - $120K", "$120K +"]].sum(), 0, 0)
    chk("F unknown", 100 * ct.loc["F", "Unknown"] / ct["Unknown"].sum(), 95.03, .005)
    # Identidad citada en 2.8 y 2.10
    assert (X["credit_limit"] - X["total_revolving_bal"] - X["avg_open_to_buy"]).abs().max() < 1e-6

    cifras.verificar()
