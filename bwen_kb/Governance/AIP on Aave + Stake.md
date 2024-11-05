**On the Aave Website: Governance Proposals Overview**

On the Aave website under the [Governance](https://app.aave.com/governance) tab, you can find all AIPs (Aave Improvement Proposals), which are all the proposals for improving the protocol that have been submitted. By clicking on one of these proposals, you will notice:

- A gauge showing the votes "for" and "against,"
- A summary of the proposal,
- Specifications indicating what improvements will be made to the protocol,
- Necessary implementations (especially regarding oracles),
- Addresses of the protocol audits,
- Addresses of the tokens and contracts that will be created by the proposal,
- Links to GitHub, Whitepapers, etc.

On the right side, you can view:

- The status of the proposal: 
  - "Voting" if it is ongoing, 
  - "Executed" if it has been implemented,
  - "Canceled" if it has been withdrawn,
- Whether the proposal has reached quorum (the minimum number of votes required for the proposal to be accepted),
- The number of addresses that participated in the voting,
- The total voting power,
- The block number and date at the time of the vote's creation,
- The block number and date indicating the start of the vote,
- The block number and date indicating the execution of the proposal.

Aave is a mature and truly decentralized DAO. When a vote is conducted and the majority response is "Yes," the contract will automatically deploy on the blockchain and will come into effect when the specified execution date and block number are reached. Once the proposal is voted on and accepted, it is immediately deployed on the protocol. This ensures that an accepted vote cannot fail to be deployed simply because a member of the development team does not want to put the proposal into production.

However, before reaching the voting stage, the proposal must first be submitted on the [Aave Governance Forum](https://governance.aave.com/). This site encompasses different categories, including governance, risks, and grants programs for developers. It is in the "Governance" category that drafts of proposals can be found, where individuals discuss and potentially suggest improvements to the proposal or discuss its technical realization. As mentioned in previous videos, this part of the governance is centralized because the site is managed by the Aave protocol itself, and it is entirely possible to simply delete a topic if it does not please the developers.

Here are the steps to submit an AIP:

1. The proposal is posted on the forum and is subject to discussions with governance members.
2. At the end of the discussions, an AIP is created by the community, usually by a team of 1 to 3 people working on the proposal.
3. This team submits the AIP to the community for evaluation and potential refinement.
4. A vote is set up on the [Snapshot](https://snapshot.org/#/aave.eth) site to create a "sign signal," which allows for an initial vote to determine whether the proposal should be maintained and submitted to the community or if it is not relevant at all. At this stage, all AAVE tokens can vote (AAVE, aAAVE, stkAAVE, amAAVE, AAVE in stkBPT, etc.) on Snapshot.
5. Once the proposal is validated on Snapshot, the vote is implemented on the official Aave governance site and submitted for governance votes.
6. Governance will then vote to approve or reject the proposal. If approved, it will be automatically implemented; if rejected, it will need to be reviewed and improved before possibly being submitted again later.

Proposers submit the AIP with the assistance of the Aave development team in the "governance contract," then will be "whitelisted," and the tokens can vote to implement the proposal or not. It is important to note that in terms of decentralization, there are "Guardians" who can intervene and interact directly with the governance contract, meaning it is not entirely 100% decentralized. These guardians can cancel a proposal if it presents a major implementation problem or even roll back changes if absolutely necessary. To cancel a proposal, at least 3 out of 5 guardians must vote with their wallets; they serve as a last line of defense in case of issues, and they are all known within the ecosystem, with their wallet addresses also being public and known to avoid any risk of manipulation. These guardians should be seen as a final safety net in case of problems.

## Staking AAVE on the Protocol

The "Staker" tab on the protocol allows users to stake AAVE to protect the protocol in case of major issues such as a "shortfall event" (for example, if USDT loses its peg, these AAVE will help settle debts). Users who stake their governance tokens help protect the protocol and are likely to incur a maximum loss of 30% of their deposit in case of significant problems, but they are rewarded with an APR of 6.89%.

The voting page for Compound, available [here](https://compound.finance/governance/proposals?target_network=mainnet), consolidates all proposals that have been made, canceled, or rejected by the protocol's governance. For example, [Proposal 32](https://compound.finance/governance/proposals/32), submitted in December 2020, proposed distributing COMP tokens to users affected by the oracle issue and the increase of DAI to $1.30 on Coinbase, which led to a wave of liquidations. There is a vote with "For" or "Against," where "Against" overwhelmingly prevailed at 76%.

On these proposals, you will find:

- The voters, some of whom are identified wallets (Polychain Capital, Argent, Kyber Network, ConsenSys), while others remain pseudonymous (0x…)
- The addresses of the concerned contracts,
- A description of what will be done, the necessary mechanisms to achieve it, the methodology, etc.,
- A link to the topic published on the Compound governance forum.

The site [comp.xyz](http://comp.xyz) is the discussion forum for the governance of the Compound protocol (similar to Aave's) where users can engage in real-time discussions about proposals (including the one regarding Proposal 32 [read here](https://www.comp.xyz/t/compensation-proposal-distribute-comp-to-affected-users-in-the-dai-liquidations/801)), freely express their views, and explain their upcoming votes. Before proposals are voted on, they are subject to scrutiny on the forum, which helps improve a proposal or shed light on a direct consequence it could have on the rest of the protocol and synergies with other protocols.

#Governance 