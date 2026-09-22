"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")
        investment_debate_state = state["investment_debate_state"]
        asset_type = state.get("asset_type", "stock")

        if asset_type == "crypto":
            prompt = f"""As the Research Manager for a crypto perpetual-futures desk,
critically evaluate the bull/bear debate and deliver a clear directional plan for
the trader.

{instrument_context}

For the structured rating, use the scale below with these crypto meanings:
- Buy: strong LONG bias
- Overweight: constructive LONG bias, but size conservatively
- Hold: NO_TRADE / wait for clearer evidence
- Underweight: cautious SHORT bias or materially reduced long exposure
- Sell: strong SHORT / avoid-long bias

Reserve Hold for genuinely mixed evidence or poor data quality. Ground the plan
in the debate evidence, especially market structure, derivatives, macro/news,
catalysts, regime, sentiment, and invalidation conditions. Do not introduce
company-fundamental concepts.

Debate History:
{history}
"""
        else:
            prompt = f"""As the Research Manager and debate facilitator, your role is to
critically evaluate this round of debate and deliver a clear, actionable
investment plan for the trader.

{instrument_context}

Rating Scale:
- Buy: Strong conviction in the bull thesis; recommend taking or growing the position
- Overweight: Constructive view; recommend gradually increasing exposure
- Hold: Balanced view; recommend maintaining the current position
- Underweight: Cautious view; recommend trimming exposure
- Sell: Strong conviction in the bear thesis; recommend exiting or avoiding the position

Commit to a clear stance whenever the debate's strongest arguments warrant one;
reserve Hold for situations where the evidence on both sides is genuinely balanced.

Debate History:
{history}
"""

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt + get_language_instruction(),
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
