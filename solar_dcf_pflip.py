"""
Utility-Scale Solar DCF — Partnership Flip Model (v2)
Streamlit interactive app

Install dependencies:
    pip install streamlit pandas numpy numpy-financial plotly

Run:
    streamlit run solar_dcf_pflip.py
"""

import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Constants ────────────────────────────────────────────────────────────────
YEARS = 20
YRS = list(range(1, YEARS + 1))
MACRS = [0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576]
TAX_RATE = 0.21

# ── Math helpers ─────────────────────────────────────────────────────────────
def calc_npv(rate, cashflows):
    return sum(cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cashflows))

def calc_irr(cashflows):
    try:
        return npf.irr(cashflows)
    except Exception:
        return float("nan")

def fmt_m(v):
    if v is None or np.isnan(v):
        return "—"
    return f"${v/1e6:.1f}M"

def fmt_pct(v, d=1):
    if v is None or np.isnan(v):
        return "—"
    return f"{v*100:.{d}f}%"

# ── Debt sizing (sculpted) ────────────────────────────────────────────────────
def size_debt(ebitda_arr, dscr_target, rate, term, capex):
    """
    Sculpted debt: DS[i] = EBITDA[i] / dscr_target for each year in the debt term.
    Loan = PV of that payment stream at the debt rate.
    DSCR = dscr_target in every year by construction.
    Capped at 70% LTV; if binding, DS payments are scaled down proportionally.
    """
    r = rate / 100
    target_ds = [ebitda_arr[i] / dscr_target for i in range(term)]

    # Loan = PV of sculpted DS stream
    loan = sum(ds / (1 + r) ** (i + 1) for i, ds in enumerate(target_ds))

    # Apply 70% LTV cap — scale DS proportionally if needed
    max_ltv = capex * 0.70
    binding = f"Sculpted DSCR {dscr_target}x in every year"
    if loan > max_ltv:
        scale = max_ltv / loan
        target_ds = [ds * scale for ds in target_ds]
        loan = max_ltv
        binding = "LTV Cap (70%)"

    # Build amortization schedule
    bal = loan
    rows = []
    min_dscr = float("inf")

    for i in range(YEARS):
        if i >= term:
            rows.append(dict(open_bal=0, interest=0, principal=0,
                             debt_service=0, close_bal=0, dscr=None))
            continue
        interest = bal * r
        ds = target_ds[i]
        principal = max(ds - interest, 0)  # floor at 0 to avoid negative amortization
        close_bal = bal - principal
        dscr = ebitda_arr[i] / ds if ds > 0 else None
        if dscr is not None and dscr < min_dscr:
            min_dscr = dscr
        rows.append(dict(open_bal=bal, interest=interest, principal=principal,
                         debt_service=ds, close_bal=close_bal, dscr=dscr))
        bal = close_bal

    return loan, pd.DataFrame(rows), min_dscr, binding


# ── Tax equity sizing ─────────────────────────────────────────────────────────
def size_tax_equity(capex, itc_rate, te_yield, cfads_arr, te_pre_flip, te_post_flip, flip_year=None):
    """
    Back-solve TE contribution from NPV of ITC + MACRS tax shields + cash
    allocation at the TE investor's target yield.
    flip_year: if provided, cash benefits after this year use te_post_flip allocation.
    """
    itc_amt = capex * (itc_rate / 100)
    dep_base = capex * (1 - itc_rate / 100 / 2)  # ITC basis haircut

    tax_benefits = []
    for i in range(YEARS):
        dep = (MACRS[i] if i < len(MACRS) else 0) * dep_base
        itc = itc_amt if i == 0 else 0
        tax_benefits.append(itc + dep * TAX_RATE)

    cash_benefits = []
    for i in range(YEARS):
        yr = i + 1
        alloc = (te_pre_flip / 100) if (flip_year is None or yr <= flip_year) else (te_post_flip / 100)
        cash_benefits.append(cfads_arr[i] * alloc)

    combined = [tax_benefits[i] + cash_benefits[i] for i in range(YEARS)]
    te_contribution = calc_npv(te_yield / 100, combined)
    # Hard cap: TE rarely exceeds ~45% of capex in practice
    te_contribution = max(0, min(te_contribution, capex * 0.45))
    return te_contribution, itc_amt, dep_base, tax_benefits


def estimate_flip_year(te_contrib, cfads_arr, itc_amt, dep_base, te_pre_flip, te_post_flip, te_yield_target):
    """Quick pass to find the year TE running IRR first hits the target yield."""
    te_cfs = [-te_contrib]
    flip_year = None
    for i in range(YEARS):
        is_pre_flip = flip_year is None
        te_alloc = (te_pre_flip / 100) if is_pre_flip else (te_post_flip / 100)
        dep = (MACRS[i] if i < len(MACRS) else 0) * dep_base
        macrs_shield = dep * TAX_RATE
        itc = itc_amt if i == 0 else 0
        te_cf = cfads_arr[i] * te_alloc + itc + macrs_shield
        te_cfs.append(te_cf)
        if flip_year is None and len(te_cfs) > 2:
            irr = calc_irr(te_cfs)
            if irr is not None and not np.isnan(irr) and irr >= te_yield_target / 100:
                flip_year = i + 1
    return flip_year


# ── Core model ────────────────────────────────────────────────────────────────
def run_model(p):
    """
    p: dict of all input parameters
    Returns: dict of computed results
    """
    capex = p["size_mw"] * 1e6 * p["capex_per_w"]

    # Step 1 — operating cash flows (pre-debt)
    op_rows = []
    for i, yr in enumerate(YRS):
        cf_factor = (1 - p["degradation"] / 100) ** i
        energy_mwh = p["size_mw"] * (p["capacity_factor"] / 100) * 8760 * cf_factor
        ppa = p["ppa_mwh"] * (1 + p["ppa_escalator"] / 100) ** i
        revenue = energy_mwh * ppa
        opex_cur = p["size_mw"] * p["opex_per_mw"] * (1 + p["opex_escalator"] / 100) ** i
        ebitda = revenue - opex_cur
        op_rows.append(dict(yr=yr, revenue=revenue, opex=opex_cur, ebitda=ebitda))
    op_df = pd.DataFrame(op_rows)
    ebitda_arr = op_df["ebitda"].tolist()

    # Step 2 — size debt (min DSCR across all years)
    loan, debt_sched, min_dscr, binding = size_debt(
        ebitda_arr, p["dscr_target"], p["debt_rate"], p["debt_term"], capex
    )

    # Step 3 — CFADS
    cfads_arr = [ebitda_arr[i] - debt_sched.loc[i, "debt_service"] for i in range(YEARS)]

    # Step 4 — size tax equity (iterate to convergence with flip year)
    flip_year_est = None
    for _ in range(10):
        te_contrib, itc_amt, dep_base, tax_benefits = size_tax_equity(
            capex, p["itc_rate"], p["te_yield"], cfads_arr,
            p["te_pre_flip"], p["te_post_flip"], flip_year=flip_year_est
        )
        new_flip = estimate_flip_year(
            te_contrib, cfads_arr, itc_amt, dep_base,
            p["te_pre_flip"], p["te_post_flip"], p["te_yield"]
        )
        if new_flip == flip_year_est:
            break
        flip_year_est = new_flip
    equity_contrib = max(capex - loan - te_contrib, 0)

    # Step 5 — build annual rows with flip logic
    te_cfs = [-te_contrib]
    sp_cfs = [-equity_contrib]
    flip_year = None
    rows = []

    for i, yr in enumerate(YRS):
        ds = debt_sched.iloc[i]
        cfads = cfads_arr[i]
        dep = (MACRS[i] if i < len(MACRS) else 0) * dep_base
        taxable_income = op_df.loc[i, "revenue"] - op_df.loc[i, "opex"] - ds["interest"] - dep
        macrs_shield = dep * TAX_RATE
        itc_alloc = itc_amt if yr == 1 else 0

        is_pre_flip = flip_year is None
        te_alloc = p["te_pre_flip"] / 100 if is_pre_flip else p["te_post_flip"] / 100
        sp_alloc = 1 - te_alloc

        te_cf = cfads * te_alloc + itc_alloc + macrs_shield
        sp_cf = cfads * sp_alloc

        te_cfs.append(te_cf)
        sp_cfs.append(sp_cf)

        te_running_irr = calc_irr(te_cfs) if len(te_cfs) > 2 else None

        if flip_year is None and te_running_irr is not None \
                and not np.isnan(te_running_irr) \
                and te_running_irr >= p["te_yield"] / 100 \
                and i > 0:
            flip_year = yr

        rows.append(dict(
            yr=yr,
            revenue=op_df.loc[i, "revenue"],
            opex=op_df.loc[i, "opex"],
            ebitda=op_df.loc[i, "ebitda"],
            open_bal=ds["open_bal"],
            interest=ds["interest"],
            principal=ds["principal"],
            debt_service=ds["debt_service"],
            close_bal=ds["close_bal"],
            dscr=ds["dscr"],
            cfads=cfads,
            depreciation=dep,
            taxable_income=taxable_income,
            macrs_shield=macrs_shield,
            itc_alloc=itc_alloc,
            is_pre_flip=is_pre_flip,
            te_alloc_pct=te_alloc,
            sp_alloc_pct=sp_alloc,
            te_cf=te_cf,
            sp_cf=sp_cf,
            te_running_irr=te_running_irr,
        ))

    results_df = pd.DataFrame(rows)
    sp_irr = calc_irr(sp_cfs)
    te_irr = calc_irr(te_cfs)
    project_npv = calc_npv(p["discount_rate"] / 100, cfads_arr)

    return dict(
        capex=capex,
        loan=loan,
        te_contrib=te_contrib,
        equity_contrib=equity_contrib,
        itc_amt=itc_amt,
        dep_base=dep_base,
        min_dscr=min_dscr,
        binding=binding,
        flip_year=flip_year,
        sp_irr=sp_irr,
        te_irr=te_irr,
        project_npv=project_npv,
        df=results_df,
        debt_sched=debt_sched,
        sp_cfs=sp_cfs,
        te_cfs=te_cfs,
    )


# ── Streamlit UI ──────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Solar DCF — Partnership Flip",
        page_icon="☀️",
        layout="wide",
    )

    st.title("☀️ Utility-Scale Solar DCF — Partnership Flip Model")
    st.caption(
        "Debt sized via **sculpted DS = EBITDA ÷ DSCR target** each year; loan = PV of DS stream. "
        "Tax equity sized from **NPV of ITC + MACRS + cash** at TE target yield."
    )

    # ── Sidebar inputs ────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Inputs")

        st.subheader("📐 Project")
        size_mw = st.number_input("Size (MW-dc)", value=100, step=5, min_value=1)
        capex_per_w = st.number_input("Capex ($/Wdc)", value=1.50, step=0.01, format="%.2f")
        capacity_factor = st.number_input("Capacity Factor (%)", value=25.0, step=0.5, min_value=10.0, max_value=40.0)
        degradation = st.number_input("Panel Degradation (% /yr)", value=0.5, step=0.1, format="%.1f")

        st.subheader("💰 Revenue")
        ppa_mwh = st.number_input("PPA Price Yr 1 ($/MWh)", value=45.0, step=1.0)
        ppa_escalator = st.number_input("PPA Escalator (% /yr)", value=0.0, step=0.25, format="%.2f")

        st.subheader("🔧 Operating Costs")
        opex_per_mw = st.number_input("O&M + Insur + Mgmt ($/MW/yr)", value=9500, step=100)
        opex_escalator = st.number_input("Opex Escalator (% /yr)", value=2.0, step=0.25, format="%.2f")

        st.subheader("🏦 Debt")
        dscr_target = st.number_input("Min DSCR Target (all years)", value=1.35, step=0.05, format="%.2f",
                                       help="Debt sized so every year of the debt term clears this DSCR")
        debt_term = st.number_input("Loan Term (yrs)", value=18, step=1, min_value=5, max_value=25)
        debt_rate = st.number_input("Interest Rate (%)", value=6.5, step=0.25, format="%.2f")

        st.subheader("🌿 Tax Equity (Pflip)")
        itc_rate = st.number_input("ITC Rate (%)", value=30, step=1, min_value=0, max_value=50)
        te_yield = st.number_input("TE Target Yield (% IRR)", value=7.5, step=0.25, format="%.2f",
                                    help="TE contribution back-solved so TE hits this IRR on ITC + MACRS + cash")
        te_pre_flip = st.number_input("TE Pre-Flip Allocation (%)", value=99, step=1, min_value=50, max_value=99)
        te_post_flip = st.number_input("TE Post-Flip Allocation (%)", value=5, step=1, min_value=1, max_value=30)

        st.subheader("💼 Sponsor")
        discount_rate = st.number_input("Discount Rate for NPV (%)", value=10.0, step=0.5, format="%.1f")

    params = dict(
        size_mw=size_mw, capex_per_w=capex_per_w,
        capacity_factor=capacity_factor, degradation=degradation,
        ppa_mwh=ppa_mwh, ppa_escalator=ppa_escalator,
        opex_per_mw=opex_per_mw, opex_escalator=opex_escalator,
        dscr_target=dscr_target, debt_term=debt_term, debt_rate=debt_rate,
        itc_rate=itc_rate, te_yield=te_yield,
        te_pre_flip=te_pre_flip, te_post_flip=te_post_flip,
        discount_rate=discount_rate,
    )

    m = run_model(params)
    df = m["df"]

    # ── Header metrics ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Capex", fmt_m(m["capex"]))
    c2.metric("Senior Debt", f"{fmt_m(m['loan'])} ({m['loan']/m['capex']*100:.0f}%)")
    c3.metric("Tax Equity", f"{fmt_m(m['te_contrib'])} ({m['te_contrib']/m['capex']*100:.0f}%)")
    c4.metric("Sponsor Equity", f"{fmt_m(m['equity_contrib'])} ({m['equity_contrib']/m['capex']*100:.0f}%)")
    c5.metric("ITC Generated", fmt_m(m["itc_amt"]))

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Returns & Sources",
        "📋 Cash Flows",
        "📐 DSCR Profile",
        "🔀 Flip Analysis",
        "📈 Charts",
    ])

    # ── Tab 1: Returns & Sources ──────────────────────────────────────────────
    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Sponsor Equity IRR", fmt_pct(m["sp_irr"]),
                    help=f"On {fmt_m(m['equity_contrib'])} invested")
        col2.metric("Tax Equity IRR", fmt_pct(m["te_irr"]),
                    help="Includes ITC + MACRS shields + pre-flip cash")
        col3.metric("Project NPV (CFADS)", fmt_m(m["project_npv"]),
                    help=f"At {discount_rate}% discount rate")

        st.subheader("Sources & Uses")
        su = pd.DataFrame([
            {"Item": "Total Project Capex", "Amount ($M)": m["capex"]/1e6, "% Capex": m["capex"]/m["capex"]*100, "Sizing Method": "MW × $/W"},
            {"Item": "Senior Debt", "Amount ($M)": m["loan"]/1e6, "% Capex": m["loan"]/m["capex"]*100, "Sizing Method": f"Min DSCR {dscr_target}x across all {debt_term} years"},
            {"Item": "Tax Equity", "Amount ($M)": m["te_contrib"]/1e6, "% Capex": m["te_contrib"]/m["capex"]*100, "Sizing Method": f"NPV(ITC + MACRS + cash) @ {te_yield}% yield"},
            {"Item": "Sponsor Equity", "Amount ($M)": m["equity_contrib"]/1e6, "% Capex": m["equity_contrib"]/m["capex"]*100, "Sizing Method": "Residual (capex − debt − TE)"},
        ])
        st.dataframe(
            su.style.format({"Amount ($M)": "${:.1f}M", "% Capex": "{:.1f}%"}),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            f"**Binding debt constraint:** {m['binding']} | "
            f"**Min DSCR achieved:** {m['min_dscr']:.2f}x | "
            f"**Flip year:** Year {m['flip_year'] or 'Not reached'} | "
            f"**ITC haircut (dep base):** {fmt_m(m['dep_base'])}"
        )

    # ── Tab 2: Cash Flows ─────────────────────────────────────────────────────
    with tab2:
        st.subheader("Project-Level Annual Cash Flows")
        cf_display = df[[
            "yr", "revenue", "opex", "ebitda",
            "interest", "principal", "debt_service", "cfads",
            "depreciation", "macrs_shield", "taxable_income"
        ]].copy()
        cf_display.columns = [
            "Year", "Revenue", "Opex", "EBITDA",
            "Interest", "Principal", "Debt Service", "CFADS",
            "Depreciation", "MACRS Shield", "Taxable Income"
        ]
        for col in cf_display.columns[1:]:
            cf_display[col] = cf_display[col] / 1000  # convert to $000s

        st.dataframe(
            cf_display.style.format({c: "{:,.0f}" for c in cf_display.columns[1:]})
                .applymap(lambda v: "color: red" if isinstance(v, (int, float)) and v < 0 else "", subset=cf_display.columns[1:]),
            use_container_width=True, hide_index=True,
        )
        st.caption("All figures in $000s")

        st.divider()
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.markdown("**How CFADS is calculated**")
            st.markdown(
                "CFADS (Cash Flow Available for Debt Service) is the cash remaining "
                "after operating costs but before debt service is paid:\n\n"
                "> **Revenue** *(PPA price × energy generated)*  \n"
                "> − **Opex** *(O&M, insurance, land, mgmt fees)*  \n"
                "> = **EBITDA**  \n"
                "> − **Debt Service** *(interest + scheduled principal)*  \n"
                "> = **CFADS**\n\n"
                "CFADS is the pool of cash split between the tax equity investor "
                "and sponsor according to the pre/post-flip allocation percentages. "
                "It is also the numerator in the DSCR calculation."
            )
        with col_exp2:
            st.markdown("**How Taxable Income is calculated**")
            st.markdown(
                "Taxable income is the partnership's book income allocated to investors "
                "for tax purposes. It differs from CFADS because it uses depreciation "
                "instead of principal repayment:\n\n"
                "> **Revenue**  \n"
                "> − **Opex**  \n"
                "> − **Interest** *(deductible)*  \n"
                "> − **Depreciation** *(5-yr MACRS on ITC-haircut basis)*  \n"
                "> = **Taxable Income**\n\n"
                "Taxable income is typically large and negative in early years due to "
                "MACRS front-loading, generating losses that the tax equity investor "
                "absorbs (at 99% pre-flip) to offset income elsewhere. The MACRS Shield "
                "column shows the tax cash value of those losses (Depreciation × 21%)."
            )

    # ── Tab 3: DSCR Profile ───────────────────────────────────────────────────
    with tab3:
        st.subheader("DSCR by Year — Full Debt Term")
        st.info(
            f"**Binding constraint:** {m['binding']} | "
            f"**Min DSCR:** {m['min_dscr']:.2f}x | "
            f"**Target:** {dscr_target}x | "
            f"**Loan:** {fmt_m(m['loan'])}"
        )
        dscr_df = df[df["yr"] <= debt_term][[
            "yr", "open_bal", "revenue", "opex", "ebitda", "debt_service", "cfads", "dscr"
        ]].copy()
        dscr_df.columns = ["Year", "Open Balance ($M)", "Revenue", "Opex", "EBITDA", "Debt Service", "CFADS", "DSCR"]
        dscr_df["Open Balance ($M)"] = dscr_df["Open Balance ($M)"] / 1e6
        for col in ["Revenue", "Opex", "EBITDA", "Debt Service", "CFADS"]:
            dscr_df[col] = dscr_df[col] / 1000
        dscr_df["Pass/Fail"] = dscr_df["DSCR"].apply(lambda x: "✅ Pass" if x >= dscr_target else "❌ Fail")
        dscr_df["Min Year"] = dscr_df["DSCR"].apply(
            lambda x: "← binding" if abs(x - m["min_dscr"]) < 0.001 else ""
        )

        def highlight_min(row):
            if row["Min Year"] == "← binding":
                return ["background-color: #fef3c7"] * len(row)
            return [""] * len(row)

        st.dataframe(
            dscr_df.style
                .format({"Open Balance ($M)": "${:.1f}M", "Revenue": "{:,.0f}", "Opex": "{:,.0f}",
                         "EBITDA": "{:,.0f}", "Debt Service": "{:,.0f}", "CFADS": "{:,.0f}", "DSCR": "{:.2f}x"})
                .apply(highlight_min, axis=1),
            use_container_width=True, hide_index=True,
        )
        st.caption("Revenue, Opex, EBITDA, Debt Service, CFADS in $000s. Highlighted row = binding (minimum DSCR) year.")

        st.divider()
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**How DSCR is calculated**")
            st.markdown(
                "DSCR (Debt Service Coverage Ratio) measures how many times the project's "
                "cash flow covers its annual debt obligation:\n\n"
                "> **DSCR = EBITDA ÷ Annual Debt Service**\n\n"
                "where Annual Debt Service = Interest + Principal for that year. "
                "A DSCR of 1.0x means the project exactly covers its debt payments with nothing left over. "
                f"This model targets **{dscr_target:.2f}x** in every year of the debt term."
            )
        with col_d2:
            st.markdown("**How sculpted debt works**")
            st.markdown(
                "This model uses **sculpted debt service**: principal payments are shaped each year so that "
                "DSCR equals the target exactly in every period:\n\n"
                "> **DS[yr] = EBITDA[yr] ÷ DSCR target**\n\n"
                "The loan amount is then the **PV of the full DS stream** discounted at the debt rate — "
                "no binary search needed. This maximises the loan relative to the cash flow profile. "
                "Because EBITDA declines slightly each year from panel degradation, debt service also steps "
                "down over time, with more principal paid early and less later."
            )

    # ── Tab 4: Flip Analysis ──────────────────────────────────────────────────
    with tab4:
        st.subheader("Partnership Flip — Year-by-Year Allocation")
        st.info(
            f"**Flip year:** Year {m['flip_year'] or 'Not reached in 20yr'} | "
            f"**TE target:** {te_yield}% IRR | "
            f"**Pre-flip TE:** {te_pre_flip}% → **Post-flip:** {te_post_flip}%"
        )
        flip_display = df[[
            "yr", "cfads", "itc_alloc", "macrs_shield",
            "is_pre_flip", "te_alloc_pct", "te_cf", "te_running_irr", "sp_cf"
        ]].copy()
        flip_display["phase"] = flip_display["is_pre_flip"].map({True: "Pre-flip", False: "Post-flip ⚡"})
        flip_display = flip_display.drop(columns=["is_pre_flip"])
        flip_display.columns = [
            "Year", "CFADS", "ITC Alloc", "MACRS Shield",
            "TE Alloc %", "TE Cash Flow", "TE Running IRR", "Sponsor CF", "Phase"
        ]
        for col in ["CFADS", "ITC Alloc", "MACRS Shield", "TE Cash Flow", "Sponsor CF"]:
            flip_display[col] = flip_display[col] / 1000
        flip_display["TE Alloc %"] = flip_display["TE Alloc %"] * 100
        flip_display["TE Running IRR"] = flip_display["TE Running IRR"].apply(
            lambda x: f"{x*100:.1f}%" if x is not None and not np.isnan(x) else "—"
        )

        def highlight_flip(row):
            if row["Phase"] == "Post-flip ⚡":
                return ["background-color: #dbeafe"] * len(row)
            return [""] * len(row)

        st.dataframe(
            flip_display[["Year", "Phase", "CFADS", "ITC Alloc", "MACRS Shield",
                           "TE Alloc %", "TE Cash Flow", "TE Running IRR", "Sponsor CF"]]
            .style
            .format({
                "CFADS": "{:,.0f}", "ITC Alloc": "{:,.0f}", "MACRS Shield": "{:,.0f}",
                "TE Alloc %": "{:.0f}%", "TE Cash Flow": "{:,.0f}", "Sponsor CF": "{:,.0f}",
            })
            .apply(highlight_flip, axis=1),
            use_container_width=True, hide_index=True,
        )
        st.caption("All cash flows in $000s. TE Cash Flow = (CFADS × TE%) + ITC (Yr 1) + MACRS shield. Blue rows = post-flip.")

    # ── Tab 5: Charts ─────────────────────────────────────────────────────────
    with tab5:
        st.subheader("Visual Summary")

        col_a, col_b = st.columns(2)

        # DSCR bar chart
        with col_a:
            dscr_data = df[df["yr"] <= debt_term].copy()
            colors = ["#ef4444" if d < dscr_target else "#22c55e" for d in dscr_data["dscr"]]
            fig_dscr = go.Figure()
            fig_dscr.add_bar(x=dscr_data["yr"], y=dscr_data["dscr"], marker_color=colors, name="DSCR")
            fig_dscr.add_hline(y=dscr_target, line_dash="dash", line_color="#f59e0b",
                               annotation_text=f"Target {dscr_target}x")
            fig_dscr.update_layout(title="DSCR by Year", xaxis_title="Year", yaxis_title="DSCR (x)",
                                   height=350, margin=dict(t=40, b=40))
            st.plotly_chart(fig_dscr, use_container_width=True)

        # Cash flow waterfall
        with col_b:
            fig_cf = go.Figure()
            fig_cf.add_bar(x=df["yr"], y=df["ebitda"] / 1e6, name="EBITDA", marker_color="#93c5fd")
            fig_cf.add_bar(x=df["yr"], y=-df["debt_service"] / 1e6, name="Debt Service", marker_color="#fca5a5")
            fig_cf.add_scatter(x=df["yr"], y=df["cfads"] / 1e6, name="CFADS", mode="lines+markers",
                               line=dict(color="#16a34a", width=2))
            fig_cf.update_layout(title="EBITDA vs Debt Service vs CFADS ($M)",
                                 xaxis_title="Year", yaxis_title="$M",
                                 barmode="relative", height=350, margin=dict(t=40, b=40))
            st.plotly_chart(fig_cf, use_container_width=True)

        col_c, col_d = st.columns(2)

        # TE running IRR vs target
        with col_c:
            irr_data = df[df["te_running_irr"].notna()].copy()
            irr_vals = [x * 100 if x is not None and not np.isnan(x) else None
                        for x in irr_data["te_running_irr"]]
            fig_irr = go.Figure()
            fig_irr.add_scatter(x=irr_data["yr"], y=irr_vals, mode="lines+markers",
                                name="TE Running IRR", line=dict(color="#8b5cf6", width=2))
            fig_irr.add_hline(y=te_yield, line_dash="dash", line_color="#f59e0b",
                              annotation_text=f"Target {te_yield}%")
            if m["flip_year"]:
                fig_irr.add_vline(x=m["flip_year"], line_dash="dot", line_color="#ef4444",
                                  annotation_text=f"Flip Yr {m['flip_year']}")
            fig_irr.update_layout(title="Tax Equity Running IRR vs Target",
                                  xaxis_title="Year", yaxis_title="IRR (%)",
                                  height=350, margin=dict(t=40, b=40))
            st.plotly_chart(fig_irr, use_container_width=True)

        # Capital stack donut
        with col_d:
            labels = ["Senior Debt", "Tax Equity", "Sponsor Equity"]
            values = [m["loan"] / 1e6, m["te_contrib"] / 1e6, m["equity_contrib"] / 1e6]
            colors_donut = ["#3b82f6", "#22c55e", "#f59e0b"]
            fig_pie = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.5,
                marker_colors=colors_donut,
                textinfo="label+percent",
            ))
            fig_pie.update_layout(title=f"Capital Stack — {fmt_m(m['capex'])} Total",
                                  height=350, margin=dict(t=40, b=40))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Sponsor vs TE annual cash flows
        st.subheader("Annual Cash Flow Split: Sponsor vs Tax Equity ($M)")
        fig_split = go.Figure()
        fig_split.add_bar(x=df["yr"], y=df["sp_cf"] / 1e6, name="Sponsor CF", marker_color="#f59e0b")
        fig_split.add_bar(x=df["yr"], y=df["te_cf"] / 1e6, name="TE CF", marker_color="#22c55e")
        if m["flip_year"]:
            fig_split.add_vline(x=m["flip_year"], line_dash="dot", line_color="#ef4444",
                                annotation_text=f"Flip Year {m['flip_year']}")
        fig_split.update_layout(barmode="group", xaxis_title="Year", yaxis_title="$M",
                                height=380, margin=dict(t=20, b=40))
        st.plotly_chart(fig_split, use_container_width=True)


if __name__ == "__main__":
    main()