## Starting Point: DAO and Web3

In Web3, often referred to as the "ownership economy," community is king. DAOs are a core feature of this economy. While in Web2, platforms aim to lock in users and rely on network effects to build a moat around themselves, Web3 enables users to easily leave. The focus is on consent and ownership.

Those who choose to invest in DAOs are rewarded with equity in the form of tokens. They can hold these tokens, participate in the project’s governance, and benefit if the DAO succeeds. Theoretically, these stakes are liquid, allowing members to sell part of their holdings if needed.

## Net Community Score Calculation Methodology

We collected readily available data online, including registered and active members (online) on Discord servers and Telegram chats, Twitter follower counts, and voter turnout rates for DAOs. Among these four variables, we assigned more weight to active community engagement metrics and less to those tracking the size of these communities.

We weighted voter turnout at 1.75x, online member counts on Discord servers and Telegram group chats at 1.5x, registered membership on these community platforms at 1x, and Twitter followers at 0.75x.

Our net community score is on a 5-point scale. For each variable, data was normalized on a zero-to-one scale (by dividing all data points by the highest sampled value for that variable). Since voter turnout is naturally a zero-to-one ratio, it was left as is.

To calculate voter turnout, we reviewed voting statistics for the last five governance proposals in each DAO and the average number of votes cast by individual addresses. We then divided the output by the number of eligible voters according to governance rules and the number of token holders identified by block explorers.

At **_The Business of Business_**, we want you to verify our work and reproduce the research if desired. Therefore, each "input" value is listed with its corresponding source on [KgBase](https://www.kgbase.com/data/othmanezizi/daos-net-community-scores/ctx-MrYrV0XNgTOI6Ugt0uu/overview).

For this project, we examined 50 of the largest DAOs by market capitalization, ranked by CoinMarketCap (defined as the product of circulating supply and token prices). We then excluded DAOs with private governance, governance not yet distributed, or where we couldn't find public data. Our final results include 38 DAOs.

## Here Are the DAOs with the Highest Net Community Scores

The five DAOs with the highest net community scores are [Illuvium](https://www.illuvium.io/), [Uniswap](https://uniswap.org/), [PIVX](https://pivx.org/?hl=en), [Ethereum Name Service](https://ens.domains/), and [Rarible](https://rarible.org/). The five DAOs with the lowest net community scores are [Edgeware](https://edgewa.re/), [PowerPool](https://powerpool.finance/), [DFI.Money](https://dfi.money/#/), [Akropolis](https://www.akropolis.io/), and [BarnBridge](https://barnbridge.com/).

[Illuvium](https://www.illuvium.io/) has the highest net community score with 4.03 out of 5 points. With a market cap of $715 million, it is the tenth most valuable DAO. It also has the largest number of registered members on Discord, with 233,000.

One reason for its high voter engagement at 76% is the existence of an elected council regularly selected by ILV token holders to manage governance decisions. As a result, the Illuvinati Council, as it's called in the DAO’s [official documentation](https://docs.illuvium.io/whitepaper/dao), is expected to respond when invited to vote on DAO proposals. Play-to-earn is a gaming trend started by Axie Infinity, where token-based games allow participants to earn crypto while playing.

## There Are Outliers When Examining Net Community Score and Market Cap

Most DAOs we reviewed fall within a range, but there were notable outliers – Illuvium and Uniswap.

[Illuvium](https://www.illuvium.io/), appearing in the upper left corner, has an unusual voting system where an elected council votes on governance proposals, significantly contributing to its outlier status. [Uniswap](https://uniswap.org/), shown on the right, has a net community score of 1.48 and a market cap of $9.3 billion. The decentralized exchange has the second largest number of registered Discord members and 757,600 Twitter followers. However, with 344 active voters, its active voter count is below the average of 1,428, although it’s above the median value of 45 active voters.

Meanwhile, [Decred](https://www.decred.org/), the green dot with a net community score of 0.27, has the most active voters in this study, averaging 28,748. It also has a market cap of just under $1 billion. Compared to its eligible voters, it’s the seventh-highest voter engagement rate. Voting here requires a ticket costing a hefty 197 DCR (around $14,000). Ticket voting is unusual for DAOs. The colors on our graph correspond to voting types: red for DAOs where only “masternodes” can vote, blue where votes are weighted by a member’s token holdings, green for DAOs requiring a ticket to vote, and purple for DAOs where “politician” members are elected to vote.

## Larger DAOs Don’t Necessarily Have Higher Net Community Scores

Zooming in on scores that aren’t outliers reveals more consistent trends. In particular, there is a slight positive correlation between market cap and net community score. However, it doesn’t appear statistically significant. (The correlation coefficient of 0.067 suggests a rather weak relationship.) This is due to the voter turnout rate, which was an essential factor in generating net community scores. Larger DAOs, by market cap, tend to have large pools of eligible voters. Although they may have more active voters, this often still translates into lower turnout rates.

[Article link](https://www-businessofbusiness-com.translate.goog/articles/ranking-daos-we-computed-their-net-community-score-to-see-how-they-stack-up/?_x_tr_sl=en&_x_tr_tl=fr&_x_tr_hl=fr&_x_tr_pto=wapp)

#DAO 