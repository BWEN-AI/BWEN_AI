Voting methods primarily attempt to resolve the resilience/scalability dilemma in the design of DAO governance. Each has its own set of challenges, such as the difficulty of educating a voter about the holographic consensus mechanism or overcoming Sybil resistance for quadratic voting, but the exploration itself is the most important outcome. The goal should be to constantly innovate, developing 2,000 years of political philosophy through technological applications.

DAOs and the crypto-economy allow for exploring the design space of voting mechanisms at an unprecedented scale. They are a new organism, and the game-theoretic mechanisms on which they are built will define their long-term survival. Their rapid pace of innovation and relative ease of formation lead to a rapid succession of experiments, helping us understand how to best coordinate to tackle humanity's greatest challenges.

Holographic Consensus—Part 1

Decentralized governance is the domain of protocols that coordinate a large number of agents in collective action, implemented with smart contracts on the blockchain. It is also the basis of the decentralized autonomous organization (DAO). With DAOs, we envision thousands or even millions of people spontaneously cooperating on common goals, generalizing and unifying the notion of enterprise, digital networks, and collective intelligence. The DAO vision soon becomes possible with the maturation of the necessary technology, particularly the DAO stack and its first DApp interface, Alchemy, both of which are live on the Ethereum blockchain mainnet.

The Scalability Problem

Decentralized governance presents numerous challenges, which is why we have not yet seen DAOs in action. In particular, I previously described what I believe to be the biggest of all, the scalability problem and its inherent tension with resilience. Every decision makes a decentralized governance system evidently unscalable; while demanding too little, it risks failing to withstand bad decisions, collusion, or simply a misrepresentation of collective opinion. Ultimately, naive consensus systems (and particularly governance systems) are simply not scalable.

Almost by definition, an appropriate solution to this problem should allow for decision-making in a DAO locally, meaning with limited attention and voting power, as long as these decisions are assured to conform to the overall opinion of the DAO. We invented this specific holographic consensus, reminiscent of a hologram where each small piece of the image actually contains the information of the entire image. In this article, I present the holographic consensus (HC) solution for scalable, resilient, and decentralized governance, describing its foundations and how it is activated. In the next article in this series, I will continue with a more detailed prescription of an HC protocol and discuss its scalability and resilience properties.

It is worth noting that holographic consensus as a scaling solution for decentralized governance systems is closely analogous to TrueBit as a scaling solution for off-chain computations for the blockchain. We will describe this analogy in more detail elsewhere.

Holographic consensus resolves the tension between resilience and scalability of governance systems. To make the discussion below as precise as possible (but not too rigorous), we will start with our definition of these two terms.

Resilience

Let us first define the overall opinion of the DAO, or simply the DAO's opinion on a proposal under consideration. The DAO's opinion on a proposal would be its ultimate decision, or outcome, given its decision-making protocol, when all agents of the DAO have sufficient attention to adequately consider this proposal and express their opinion to the DAO. (Having no opinion is also a legitimate opinion.) Sufficient bandwidth of attention is exactly what we would like to relax later for scalability reasons, but we need the DAO's opinion as a theoretical reference point. The DAO's approximate opinion on a proposal would then be obtained when most opinionated agents have approximately sufficient bandwidth to consider the proposal and express their opinion.

It should be noted that a general DAO is like a living creature, with its own subjective mind that detects, perceives, thinks, makes sense, and makes decisions. In particular, it has its own subjective good or bad, right and wrong, rather than objective. We will not discuss here what drives a DAO to make "good decisions" or what good means overall. Instead, for now, we only wish for a "good DAO" to act in accordance with its own overall opinion.

In this context, a decentralized decision-making system will be deemed resilient if it ensures that all decisions made within the DAO conform to its approximate opinion (or are approximately aligned with its overall opinion).

Scalability

The DAO has a mission, and its governance objective is to make decisions that further this mission. In particular, this could involve decisions on fund allocation to incentivize agents to act in favor of its mission. More contributing agents in the DAO means more possible contributions and the need for more resource allocation decisions regarding those contributions. Thus, to grow effectively, a DAO decision-making system must handle an increasing number of decisions within a fixed period as the DAO increases in the number of agents. When we talk about DAO scalability, we refer to its ability to increase the number of decisions it can effectively make over a period of time.

It is important to note that the only thing common to all known economic organizations today is that they become less and less efficient in decision-making as they grow. Thus, we propose that DAOs are a new generation of scalable organizations in that they can effectively scale their operation and decision-making by increasing the number of their agents.

As stated earlier, the fact that most voters review each proposal is clearly not scalable. At the same time, a resilient DAO governance system means that decisions are made in accordance with its overall opinion. Therefore, scalable and resilient governance systems must, by definition, allow for decisions to be made with relatively low influence or voting power (which we also refer to as reputation) on behalf of the entire DAO, as long as these decisions are still guaranteed to be well-aligned with the DAO's overall opinion. This delicate situation is the very definition of holographic consensus.

Relative Majority

The simplest solution to increase decision-making power is to approve decisions by relative majority rather than absolute majority. Absolute majority is the majority of all voting rights within the DAO. For example, among one hundred equal agents, a decision by absolute majority requires the approval of fifty-one agents, and implicitly the active participation of at least fifty-one of them. Very resilient and non-scalable situation.

Approval by relative majority requires only a majority of those who have voted on a given proposal within a certain voting period. A proposal is open, let’s say, for a week, at the end of which a decision is made. If only nine equal agents out of one hundred have voted on this proposal within that week, it is enough for five of them to support the proposal for it to be approved. A relative majority voting system is indefinitely scalable, able to process and make a decision on any number of proposals within a given time frame, but it is also potentially not resilient. If there is no limit to the number of proposals and no other conditions to the relative majority, anyone can easily attack the system by spamming it with millions of proposals, diluting voters' attention where most proposals will remain unnoticed. The attacker could be the only voter on a malicious proposal to transfer all funds from the DAO. But beyond the overwhelming attack on collective attention, decisions will not reflect the DAO's overall opinion, increasingly so with the increase in the scale of decisions made within a fixed timeframe (hence the aforementioned tension). In our previous terminology, the system will not be resilient, which means it would be both manipulable and would not act with or generate coherence.

It is worth mentioning that the standard use of relative majority-based voting systems includes a quorum: a minimum threshold of voters necessary to render a vote valid. The problem with quorums is that they encounter the scalability problem of an absolute majority system when set too high, the resilience problem of a relative majority system when set too low, and often both at the same time. This conflict increases with the volume of decisions made, and therefore quorums are bad for scalable governance.

Boosted Proposals

Decisions must be approved by absolute majority by default. However, we would like to allow decisions to be made by relative majority under certain protective conditions that ensure the alignment of these decisions with the absolute majority of the DAO. We designate by boosting the transition of a proposal from an absolute majority vote to a relative majority vote, and by boosting conditions the conditions that a proposal must meet to be boosted.

Monetizing Attention

Boosting a proposal allows agents to consume the scarce resource of collective attention—perhaps the rarest resource in a DAO—and impose a decision on their behalf in a finite time. Therefore, it must be monetized.

The simplest boosting condition would be payment in DAO tokens. Do you want the DAO to consider your proposal? Easy, for one hundred DAO tokens, the collective attention is (mostly) yours. You cannot buy a decision, but you can buy consideration. Anyone can promote a proposal, not just the proposer, and once a certain threshold of promotion is reached, the proposal is boosted.

Dynamic Threshold

A fixed boosting rate will not work well, just like quorums. Setting it too high would make the decision-making process too exclusive, while setting it too low would lower the price of the attack. Furthermore, for resilience reasons, collective attention should be priced based on supply and demand; if twenty proposals are boosted and under the control of the DAO, collective attention should be significantly more expensive to dilute further than when only one decision is at stake. An appropriate boosting threshold must have a protective character and must be solidly exponential in the number of proposals already boosted. Note that the DAO tokens paid can be redistributed to voters for their efforts, which is useful for incentivizing voting.

The exponential boosting threshold closely limits the number of decisions made by relative majority at any given time. The advantage is that it makes the system fairly resilient—all collective attention is channeled into the same few proposals, allowing decisions to reflect the global opinion. The volume of simultaneously boosted proposals can be slightly stretched while voters are (exponentially) well-rewarded to maintain their presence. The downside of the exponential threshold is that it quickly makes the cost of boosting very, very high. Again, the system is not as scalable as we would like.

To make the last point more concrete, let us define the internal value of a proposal as the total value perceived by the promoters of a proposal upon its approval. By definition, promoters would be willing to pay at most the total internal value of a proposal for its boosting. The exponential boosting threshold means an "exponential filtering" of proposals regarding their internal value. If the boosting threshold is 1000 DAO tokens, only proposals with an internal value greater than 1000 DAO tokens will be brought to collective attention.

Two Tokens

To improve the boosting conditions, we need to separate two different economies. Voter attention is deployed to make decisions, and this must be monetized, rewarding their efforts. To consume this scarce resource, it must be paid with another scarce resource related to its network effect, the DAO token. A second monetization need relates to boosting, transitioning a proposal from a condition of absolute majority approval to relative and penetrating it into the collective attention of the DAO. This latter monetization has two purposes:

- **Efficiency** - protect and reduce voters' attention expenditures by filtering the overwhelming number of online proposals into those on which voters should decide in a targeted manner and within a limited time.
- **Alignment** - ensure the sufficient and impartial presence of voters for each boosted proposal to guarantee the alignment of resulting decisions made by relative majority with the opinion of the absolute majority.

Note that the two monetizations are different and should not be linked as in the previous example. In particular, they are best produced by two different network effects and two different tokens. DAO decisions require the attention of the DAO's voters and are related to the DAO's network effect and its symbolic monetization; while filtering proposals and protecting the decision-making process around them are better executed by an open, economic, and permissionless network, a network of predictors.

Predictor Network

Consider a predictor network placing predictions on the fate of proposals in different DAOs and staking tokens to back them. Successful predictors who bet on "good proposals" are rewarded—good in the only sense that the DAO ultimately approved them. And those who fail lose (perhaps part of) their stake. Likewise, predictors are compensated for staking against boosted proposals that were ultimately rejected, and the bare loss in the opposite case.

Now, a proposal is only boosted when it reaches sufficient staking in its favor, reflecting the predictors' confidence that it will pass in the DAO. The exponential threshold—this time of staked tokens—is still important for protection, which means the amount of staking necessary to boost a proposal is roughly exponential in the number of proposals already boosted. To incentivize predictions, a bonus is offered to retained predictors from retrospectively approved proposals. The bonus can come from the proposal promotion fees or be provided by the DAO itself.

Predictors have three roles in enhancing DAO decision-making, in line with the objectives of the aforementioned boosting conditions:

1. **Searching for a “good proposal”** - predictors filter a long list of proposals for voters to focus on. The generosity of predictors creates an economic incentive to discover these proposals, and an open market will seize this opportunity.
    
2. **Signal and balance the propagation of a “bad proposal”** - once a proposal is boosted, predictors are even more incentivized to reduce it if they believe it will ultimately not be approved by voters. Again, an economic market should saturate this arbitrage.
    
3. **Finally, once predictors are staked on both sides of the future, they are incentivized to maintain the voting process and its alignment with the global opinion. Those who observe a misalignment would attract collective attention, and more particularly the voters who they believe will fix it.
    

Note that predictors simply serve as routing agents and do not need to have any vested interest in the success of the decisions or be linked to the DAO in any way. In particular, they do not need to hold voting rights within the DAO. Predictors are motivated by pure short-term economics, like traders, and they can be anyone who believes they have an informational advantage they can exploit and is confident enough to stake tokens for this purpose. This is an open and permissionless network, unlike the DAO voting system which is generally permissioned. This new global network is the DAOstack predictor network: predictors of collective social behavior. This network has its own token used for predictions, the GEN token, and a strong network effect to protect its utility. For a deeper examination of this angle, see this recent article.

Large DAOs with a large number of active members may need to make a large number of decisions at a high frequency.

For the sake of resilience, all these decisions must be representative of the overall opinion of the DAO, that is, in accordance with the majority of votes of the DAO.

On the other hand, as attention is scarce and decisions so frequent, only a small fraction of the DAO's voting power can observe each decision.

Thus, we arrive at a conflict: naive governance protocols cannot allow for frequent and aligned decisions at the same time.

Holographic Consensus (HC) provides a solution to this tension by allowing local decisions, thus scalable, that ensure the representation of the whole group.

The proposed implementation below is not perfect, but we believe it is free from major failure modes and sufficiently good to be put into service with the launch of DAOs on Alchemy Earth. A complete technical specification of the protocol can be found here.

Genesis Protocol

The goal of the Genesis v0.2 protocol is to achieve a minimally viable version of holographic consensus that largely extends the governance throughput of crowd organizations while avoiding major resilience issues. We will begin by describing the basic mechanisms of the protocol.

The DAO system consists of smart contracts that can control assets and internal or external rights. It is activated by agents, identified by Ethereum addresses, who interact with the DAO by submitting proposals for actions, then voting and staking on them:

It is important to note that the DAO itself reduces a specific amount of GEN on each proposal to help incentivize the initial predictors to seek out good proposals. This payment reflects the cost of outsourcing the navigation of collective attention to an efficient market, and thus the effective cost of scalable and resilient governance. Even with this outsourcing, decisions are nevertheless made solely by the DAO’s reputation holders.

The decision-making flow within the DAO unfolds as follows:

Boosted proposals allow for greater scalability; they still have some voter support for approval, but they are exempt from any voting quorum requirement (relative majority). The boosting condition is designed to also maintain the representativeness of outcomes and thus the resilience of decision-making:

Each proposal has its market confidence, equal to its total stakes divided by total stakes, C=S+/S-. Confidence represents the probability that predictors believe it will pass. For example, a confidence of four means that predictors think the proposal has four chances against one of being approved by voters. Predictors who disagree with the market and believe that a proposal’s confidence is higher (lower) relative to its actual chances of being approved by voters are incentivized to place a downstake (upstake) until the forecasts finally reach saturation.

A proposal becomes boosted if its confidence score is above the boosting threshold continuously for a certain time, similarly to the silent end function. This additional time requirement gives dissenting predictors a chance to place their stakes, preventing instant boosting (which opens an attack surface) and allows for the saturation of predictions in the interest of maximizing security.

The boosting threshold we use in Genesis v0.2 grows exponentially with the number of currently boosted proposals, N_B, with α > 1 being the boosting difficulty parameter. The dependence on N_B ensures defense against the dilution of voter attention.

Although the Genesis protocol may seem complicated at first glance, after dozens of protocol explorations, we found that it was the simplest HC implementation that properly mitigates the tension between scalability and resilience in large DAOs. In the upcoming articles, we will explain in more detail how and to what extent the Genesis protocol resolves this tension, and provide a deeper discussion of the possible failure modes of the protocol.

Learn More and Get Involved

Dive into Alchemy and embark on the Genesis DAO.

Are you a governance researcher or a decentralization advocate? Join the conversation on DAOtalk.

Join the DAOstack community on Discord and follow it on Twitter.

Decisions made locally contain the preference information of the entire group, resembling a hologram.

With the decline of the DAO, predictors are incentivized to seek out good proposals. Once a proposal is accepted, predictors are also incentivized to reduce it if they believe it will fail.

If waiting times allowed the prediction game to resolve normally, whales would be economically incentivized to attempt a censorship attack: placing stakes on proposals so significant that they would never be amplified, making it highly likely that these proposals expire (and giving whales their stake back plus a portion of the stake, if the waiting times are resolved as "fail"). If profitable, whale censorship attacks are dangerous. An unprofitable whale censorship attack, on the other hand, is risky enough that the attacker would not be viable. We will cover a detailed analysis of the protocol's failure modes elsewhere.

#Governance 