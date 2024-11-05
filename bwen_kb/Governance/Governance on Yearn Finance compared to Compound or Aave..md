- **[Forum](https://gov.yearn.finance/)**: This forum includes several sections, most notably the proposals listed in the “YIP” section. These proposals, once developed and sufficiently documented, are sent to Snapshot.

- **[Snapshot](https://snapshot.org/#/yearn)**: All proposals, after being discussed on the forum, are voted on by YFI holders directly on this platform. The proposal is fully detailed to ensure that voters understand the motivations behind the proposed improvements to the protocol.

Anecdote: During the liquidity mining week of Yearn at the very beginning of the protocol, all YFI were minted and distributed, but Yearn had none left in its treasury. Thus, the team decided to mint 6,666 new YFI to ensure they had treasury funds. This action was received poorly by the Yearn community, as it meant diluting the protocol, but the proposal was accepted, and the increase in supply occurred because it allowed the Yearn team to pay salaries, hire new developers, secure the protocol, and contribute to the Yearn ecosystem.

The governance on Curve closely resembles others but operates exclusively on the Curve website. All governance is managed from this address: [dao.curve.fi/dao](http://dao.curve.fi/dao). First, it is worth noting the existence of “Emergency DAO members,” which includes 9 members and their Twitter accounts, including:

- Julien Bouteloup, Rekt / Stake DAO / Curve...
- Kain, CEO of Synthetix
- Stani, CEO of Aave
- Curve Finance, the development team

These members are recognized within the DeFi ecosystem and can, in emergencies, ensure a vote of 59.999% and gather 51% of the quorum, allowing them to intervene in case of a real threat of fund loss. They can invoke the “kill_me” function on Curve contracts, thus enabling direct interaction and preventing a fatal issue. These members are equivalent to Guardians on Aave.

### Voting on the Curve DAO: 
Regarding votes, we have already seen that it is mandatory to hold veCRV, which means CRV locked for a certain period in the protocol (from 7 days to 4 years). Unlocked CRV does not allow voting or participation in governance. Currently, 68% of the circulating CRV is locked in the protocol as veCRV for an average duration of 3.67 years.

On the page [dao.curve.fi/dao](http://dao.curve.fi/dao), all proposals currently being voted on are listed, where only veCRV holders will be able to vote. For example, in proposal #72, it is necessary that:

- 60% of voters approve the proposal for it to be accepted
- 15% of veCRV votes favorably for the proposal to be accepted

The quorum, here at 15%, varies according to the importance of the proposal; generally, a token listing or a minor parameter change (for a given pool or token) will not require a very high quorum. For instance, on Aave, listing a token (RAI for our previous example) only requires a 2% quorum. However, if a proposal suggests drastically changing the fundamentals of a protocol (like AAVE V3 or Curve V2), the required quorum is significantly higher, around 51% for Curve, for example. All these parameters are decided beforehand during the proposal submission on the forum.

As of July 21, 2021, Curve has 74 proposals, all contributing to having a living, evolving protocol and an open and transparent governance process.

### Curve on Snapshot: 
Curve also has a Snapshot page that allows it to submit proposals before publishing them on the dedicated DAO page for a final vote. Here, and only here, it will be possible to vote for or against using a Metamask wallet containing CRV.

### Governance Attacks via Vote Buying: 
One could easily imagine users borrowing governance tokens to execute a “governance attack.” On Aave, following a proposal approved by a governance vote, it is no longer possible to borrow such tokens. For instance, imagine someone (a user/protocol/crypto project) decides to borrow all the MKR deposited on Aave, about 44,000 MKR; they could then submit a proposal and vote in favor of it on MakerDAO. This process is called a “governance attack via vote buying,” where it is possible, by holding sufficient voting power, to impose a proposal's passage even if it is not in the community's best interest.

For example, imagine a protocol that desperately wants to be listed on Maker. If community sentiment is generally negative towards this proposal, the protocol wishing to be listed could execute a sort of coup by borrowing a large quantity of MKR to push through the proposal, overriding the voting power of the rest of the community. This is why it is now absolutely impossible to borrow governance tokens on Aave (YFI, BAL, KNC, CRV, MKR, UNI, SNX… all are disabled for borrowing).

#Governance 