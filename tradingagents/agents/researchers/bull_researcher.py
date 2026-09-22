from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")
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
            prompt = f"""You are the Bull Researcher for a crypto perpetual-futures research desk.
Build the strongest evidence-based bullish case for the asset while remaining
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
Last Bear Argument: {current_response}

Focus on:
- Trend, momentum, volatility, liquidity, support/resistance, and entry quality.
- Funding, open interest, premium, liquidations, crowded positioning, and squeeze risk.
- Market regime and whether conditions support directional exposure.
- Confirmed catalysts, macro conditions, and material news.
- Sentiment only as supporting evidence, not as a substitute for market data.
- Directly rebut the bear case using specific evidence from the reports.
- State the conditions that would invalidate the bullish thesis.

Do not use company-analysis concepts such as revenue, earnings, margins, branding,
or competitive moat. If a data source is missing or stale, explicitly acknowledge
the limitation rather than filling the gap with assumptions.

Deliver a concise but substantive bullish argument for the next decision stage."""
        else:
            prompt = f"""You are a Bull Analyst advocating for investing in the stock.
Your task is to build a strong, evidence-based case emphasizing growth potential,
competitive advantages, and positive market indicators. Leverage the provided
research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections,
  and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding,
  or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive
  news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and
  sound reasoning.
- Engagement: Address the bear analyst's points directly rather than just listing data.

Resources available:
{instrument_context}
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}

Use this information to deliver a compelling bull argument and refute the bear's concerns."""

        response = llm.invoke(prompt + get_language_instruction())
        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
