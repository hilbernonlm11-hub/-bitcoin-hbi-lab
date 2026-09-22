from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        derivatives_report = state.get("derivatives_report", "")
        catalyst_report = state.get("catalyst_report", "")
        regime_report = state.get("regime_report", "")
        instrument_context = get_instrument_context_from_state(state)
        trader_decision = state.get("trader_investment_plan", "")
        asset_type = state.get("asset_type", "stock")

        if asset_type == "crypto":
            prompt = f"""You are the Neutral Risk Analyst for a crypto perpetual-futures desk.
Evaluate the trader proposal from a neutral risk perspective. Use the
actual crypto reports below and do not invent market data.

Trader proposal:
{trader_decision}

Evidence:
{instrument_context}
Market Structure Report: {market_report}
Derivatives Report: {derivatives_report}
Sentiment Report: {sentiment_report}
News / Macro Report: {news_report}
Catalyst Report: {catalyst_report}
Market Regime Report: {regime_report}

Risk debate history:
{history}
Aggressive view: {current_aggressive_response}
Conservative view: {current_conservative_response}
Neutral view: {current_neutral_response}

Assess:
- Whether LONG, SHORT, or NO_TRADE is justified by current structure.
- Funding, open interest, premium, liquidation and squeeze risk.
- Volatility, liquidity, spread, regime, and stale/missing-data risk.
- Entry/stop/take-profit coherence and asymmetry of reward versus risk.
- Macro/news/catalyst risks that could invalidate the trade.
- Whether size or leverage should be reduced, or the trade rejected.

Balance opportunity and capital preservation. Challenge both overconfidence and excessive caution, and prefer NO_TRADE when evidence is mixed.
Do not use company-fundamental concepts. Be explicit when evidence is weak or missing.
The deterministic risk gate remains the final authority; your job is to surface the
strongest relevant risk arguments for it and the Portfolio Manager."""
        else:
            prompt = f"""As the Neutral Risk Analyst, evaluate the trader's decision from a
neutral risk perspective.

Trader decision:
{trader_decision}

Context:
{instrument_context}
Market Research Report: {market_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Risk debate history: {history}
Aggressive view: {current_aggressive_response}
Conservative view: {current_conservative_response}
Neutral view: {current_neutral_response}

Balance opportunity and capital preservation. Challenge both overconfidence and excessive caution, and prefer NO_TRADE when evidence is mixed."""

        response = llm.invoke(prompt + get_language_instruction())
        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": (
                risk_debate_state.get("aggressive_history", "")
            ),
            "conservative_history": (
                risk_debate_state.get("conservative_history", "")
            ),
            "neutral_history": (
                neutral_history + "\n" + argument
            ),
            "latest_speaker": "Neutral",
            "current_aggressive_response": (
                risk_debate_state.get("current_aggressive_response", "")
            ),
            "current_conservative_response": (
                risk_debate_state.get("current_conservative_response", "")
            ),
            "current_neutral_response": (
                argument
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
