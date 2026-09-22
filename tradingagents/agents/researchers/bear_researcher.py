from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)


def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")
        current_response = investment_debate_state.get("current_response", "")

        market_research_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        derivatives_report = state.get("derivatives_report", "")
        catalyst_report = state.get("catalyst_report", "")
        regime_report = state.get("regime_report", "")
        instrument_context = get_instrument_context_from_state(state)
        asset_type = state.get("asset_type", "stock")

        if asset_type == "crypto":
            prompt = f"""You are the Bear Researcher for a crypto perpetual-futures research desk.
Build the strongest evidence-based bearish case for the asset while remaining
faithful to the supplied data. Do not invent metrics, events, or market facts.

Evidence available:
{instrument_context}
Market Structure Report: {market_research_report}
Derivatives Report: {derivatives_report}
Sentiment Report: {sentiment_report}
News / Macro Report: {news_report}
Catalyst Report: {catalyst_report}
Market Regime Report: {regime_report}
Debate History: {history}
Last Bull Argument: {current_response}

Focus on:
- Downside structure, failed momentum, volatility expansion, liquidity weakness,
  resistance, and poor entry quality.
- Excessive funding, open-interest risk, premium distortion, liquidation pressure,
  crowded positioning, and long-squeeze conditions.
- Market regime and whether it argues for risk-off or no-trade conditions.
- Confirmed negative catalysts, macro tightening, regulatory or geopolitical risk.
- Sentiment extremes that may increase downside or reversal risk.
- Directly rebut the bull case using specific evidence from the reports.
- State what evidence would invalidate the bearish thesis.

Do not use company-analysis concepts such as revenue, earnings, margins, branding,
competition, or market share. If a data source is missing or stale, explicitly
acknowledge the limitation rather than filling the gap with assumptions.

Deliver a concise but substantive bearish argument for the next decision stage."""
        else:
            prompt = f"""You are a Bear Analyst making the case against investing in the stock.
Your goal is to present a well-reasoned argument emphasizing risks, challenges,
and negative indicators. Use the supplied research to counter bullish arguments.

Key points to focus on:
- Risks and Challenges: Highlight financial instability, macroeconomic threats,
  or adverse industry conditions.
- Competitive Weaknesses: Identify vulnerabilities in market positioning,
  innovation, or competitive dynamics.
- Negative Indicators: Use financial data, market trends, and adverse news.
- Bull Counterpoints: Critically test the bull argument with specific evidence.
- Engagement: Address the bull analyst's points directly.

Resources available:
{instrument_context}
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}

Use this information to deliver a compelling bear argument and refute the bull's claims."""

        response = llm.invoke(prompt + get_language_instruction())
        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
