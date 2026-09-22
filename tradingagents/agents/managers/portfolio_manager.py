"""Portfolio Manager: synthesises the risk-analyst debate into the final decision."""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]
        asset_type = state.get("asset_type", "stock")

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        if asset_type == "crypto":
            prompt = f"""As the Portfolio Manager for a crypto perpetual-futures desk,
synthesize the risk debate and provide the final portfolio view before the
deterministic risk gate.

{instrument_context}

For the structured rating, use:
- Buy: supports LONG
- Overweight: mildly supports LONG / smaller size
- Hold: NO_TRADE or wait
- Underweight: mildly supports SHORT / materially reduced long exposure
- Sell: supports SHORT / avoid long exposure

Context:
- Research Manager plan: {research_plan}
- Trader proposal: {trader_plan}
{lessons_line}
Risk Analysts Debate:
{history}

Evaluate whether the proposal is justified by current evidence, whether size and
leverage are appropriate, and whether data quality, volatility, liquidity,
derivatives positioning, or macro/catalyst risk should force NO_TRADE. Do not
introduce company-fundamental concepts. The deterministic risk gate after you is
the final authority for approval, sizing and leverage.
"""
        else:
            prompt = f"""As the Portfolio Manager, synthesize the risk analysts'
debate and deliver the final trading decision.

{instrument_context}

Rating Scale:
- Buy: Strong conviction to enter or add to position
- Overweight: Favorable outlook, gradually increase exposure
- Hold: Maintain current position, no action needed
- Underweight: Reduce exposure, take partial profits
- Sell: Exit position or avoid entry

Context:
- Research Manager's investment plan: {research_plan}
- Trader's transaction proposal: {trader_plan}
{lessons_line}
Risk Analysts Debate History:
{history}

Be decisive and ground every conclusion in specific evidence from the analysts.
"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt + get_language_instruction(),
            render_pm_decision,
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
