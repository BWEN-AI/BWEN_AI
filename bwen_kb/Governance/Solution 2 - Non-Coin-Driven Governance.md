## Solution 2: Non-Coin-Driven Governance

A second approach involves using forms of governance that are not based on coin voting. If coins do not determine an account's weight in governance, what does? There are two natural alternatives:

1. **Proof of Personhood Systems**: These systems verify that accounts correspond to unique human individuals, allowing governance to allocate one vote per person. For a review of some developing techniques, see [here](https://arxiv.org/abs/2008.05300), and check out [ProofOfHumanity](https://app.proofofhumanity.id/), [BrightID](https://www.brightid.org/), and [Idena Network](https://github.com/idena-network) for three implementation attempts.

2. **Proof of Participation**: These systems attest that an account corresponds to a person who has participated in an event, undergone educational training, or performed useful work within the ecosystem. An example of this is [POAP](https://poap.xyz/), which attempts to implement this concept.

There are also hybrid possibilities: one example is [quadratic voting](https://vitalik.ca/general/2019/12/07/quadratic.html), which makes the power of a single voter proportional to the square root of the economic resources they commit to a decision. Preventing people from gaming the system by spreading their resources across many identities requires proof of personhood, while the remaining financial component allows participants to credibly signal how much they care about an issue and its importance to the ecosystem. Gitcoin's quadratic funding is a form of quadratic voting, and quadratic voting DAOs are [currently being built](https://gitcoin.co/issue/DemocracyEarth/DemocracyDAO/1).

Proof of participation is less well understood. The main challenge is that determining what counts as a degree of participation in itself requires a fairly robust governance structure. It may be simplest to start the system with a curated selection of 10 to 100 initial contributors, then decentralize over time as selected participants from round N set the criteria for participation for round N + 1. The possibility of a fork helps pave the way for recovery and resists governance drift.

Both proof of personhood and proof of participation require some form of anti-collusion (see [this article explaining the issue here](https://vitalik.ca/general/2019/04/03/collusion.html) and [the MACI documentation here](https://github.com/appliedZKP/maci)) to ensure that the non-monetary resource used to measure voting power remains non-financial and does not end up in smart contracts that sell governance power to the highest bidder.

#Governance 