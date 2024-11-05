## Hybrid Solutions

There are also solutions that combine elements from the above techniques. Here are a few possible examples:

- **Delays and Governance by Specialists**: This is a potential solution to the old puzzle of how to create a crypto-collateralized stablecoin where the locked funds can exceed the value of the profit-taking token without risking governance capture. The stablecoin uses a price oracle built from the median values submitted by N (for example, N = 13) elected providers. Coin voting selects the providers, but it can only remove one per week. If users notice that the coin voting calls upon unreliable price providers, they have N/2 weeks before the stablecoin pauses to switch to another.

- **Futarchy + Anti-Collusion = Reputation**: Users vote with "reputation," a non-transferable token. Users gain reputation if their decisions lead to desired outcomes and lose reputation if their decisions result in undesirable outcomes. See [here](https://medium.com/daostack/reputation-vs-tokens-6d7642c7a538) for an article advocating a reputation-based system.

- **Weakly Coupled Monetary Votes (Advisory Votes)**: A monetary vote does not directly implement a proposed change but exists solely to make its result public, thereby strengthening the [legitimacy](https://vitalik.ca/general/2021/03/23/legitimacy.html) of off-chain governance to implement that change. This can provide the benefits of monetary votes with less risk, as the legitimacy of a monetary vote automatically diminishes if it appears that the vote has been bribed or otherwise manipulated.

But these are just a few possible examples. There is much more to be done in researching and developing non-coin-based governance algorithms. **The most important thing that can be done today is to move away from the idea that coin voting is the only legitimate form of decentralized governance**. Coin voting is attractive because it _appears_ neutral and credible: anyone can seek units of the governance token on Uniswap. In practice, however, **coin voting can only seem secure today precisely because of the imperfections in its neutrality** (namely, large portions of the supply remaining in the hands of a closely coordinated clique of insiders).

We must remain very cautious about the notion that current forms of coin voting are "safe bets." There is still much to learn about how they function under conditions of increased economic stress and in mature financial ecosystems and markets, and now is the time to start experimenting simultaneously with alternatives.

[https://vitalik.ca/general/2021/08/16/voting3.html](https://vitalik.ca/general/2021/08/16/voting3.html)

#Governance 