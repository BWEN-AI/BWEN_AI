Holographic consensus was a voting mechanism created by DAOstack that combines futarchy, or prediction markets, with a token-based quorum voting system. In the holographic consensus model, predictors can stake funds for or against a proposal that they believe will succeed or fail, and if they predict correctly, they earn a financial incentive. If a proposal is expected to be adopted, it is "boosted," meaning it can be approved by a simple relative majority rather than requiring a quorum to be reached.

Holographic consensus is a unique solution to address the trade-off between resilience and scalability. The most straightforward response to increasing decision-making power is to approve proposals by a relative majority rather than having a quorum threshold. A relative majority only requires a majority of those who voted to approve or reject a proposal, rather than a threshold of the total number of votes. However, this could allow a malicious actor to potentially make proposals if they could slip them under the radar of the broader community. To combat this, holographic consensus implements "boosting" through prediction markets.

Decisions made under the holographic consensus model must be approved by an absolute majority by default. However, a proposal can be "boosted" from an absolute majority to a relative majority if it meets the boosting conditions. Boosting a proposal effectively gives DAOs their most scarce resource back: attention. The prediction market creates an attention economy, leading to uncontroversial proposals being adopted by a simple relative majority, while the most controversial votes remain subject to a stricter absolute majority with a voting quorum.

It is important to note that a fixed boosting rate will not solve much by itself. A fixed rate is, in fact, just another quorum mechanism and presents similar flaws: too high and quorum difficulties prevail; too low and an attacker can easily boost the proposal to attempt to attack the protocol. Implementing an exponential boosting threshold adds a layer of resilience to governance: it slows down the number of proposals to be reviewed, ensuring the DAO's attention remains highly focused. However, the cost could quickly become prohibitive, reducing scalability.

The prediction market for proposals should theoretically be managed by a token distinct from the DAO’s governance token implementing holographic consensus. This will ensure impartial predictions while freeing the DAO's attention in line with the model's stated goal. Predictors should be incentivized through a payment to accurately stake their predictions. If a predictor bets incorrectly, their stake should be partially reduced. This makes governance attacks through the boosting mechanism economically disadvantageous while improving the overall accuracy of the market.

According to DAOstack, predictors are incentivized to:

1. **Search for “good proposals”**: Predictors filter through a long list of proposals for voters to focus on. The generosity of predictors creates an economic incentive to discover these proposals, and an open market will seize this opportunity.

2. **Signal and balance the spread of “bad proposals”**: Once a proposal is boosted, predictors are even more incentivized to counter it if they believe it will ultimately not be approved by voters. Again, an economic market should saturate this arbitrage.

3. **Finally, once predictors are staked on both sides of the outcome, they are incentivized to maintain the voting process and its alignment with global opinion**. Those who observe a misalignment would draw collective attention, particularly from voters who believe they can correct it.

[DXdao](https://dxdao.eth.link/#/) is the most notable example of a DAO using the holographic consensus mechanism.

#Governance 