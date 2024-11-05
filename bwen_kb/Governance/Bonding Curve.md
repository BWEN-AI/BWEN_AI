## Capital Formation - Bonding Curve

For a capital pool to smartly deploy resources and provide a value-added portfolio, it needs sufficient assets under management (AUM) to cover operational and management costs. However, too much capital for the target investment market size can also be a problem. As with any investment, early entrants into this reserve take on more risk than later entrants as the probability of success becomes clearer over time, especially with new models such as the one described here. Early entrants should be rewarded for this asymmetric risk profile supporting the DAO. Therefore, we propose an initial capital formation period during which early contributors receive a DAO token bonus compared to subsequent contributors. The cumulative bonus should reflect the reduced liquidity for departing during the initial bonding curve period.

A DAO requires a mechanism that allows DAO members to reasonably withdraw if they need liquidity and potential new entrants to join DAO membership even after the initial capital formation period to align with incentives. Thus, we propose a bonding curve in which one can buy into the DAO with ETH, WBTC, or another approved ERC20, known as "underlying capital," in exchange for DAO tokens in a bonding curve function with increasing costs. Conversely, a DAO member can burn DAO tokens and receive the underlying liquid tokens minus a liquidity penalty corresponding to illiquid positions in the DAO plus the capital allocated in the current mandate.

A bonding curve controls the inflation of the token supply in a way that meets both of these requirements.

- ![Graph plotted on appropriate polar coordinates](https://camo.githubusercontent.com/6187d5c656600e48ddd4339cd2a93174a7d8b8a336d9cfa0b0919466299c61fa/68747470733a2f2f692e696d6775722e636f6d2f32326f525530662e706e673d32353078)

### Phase 1 - Encourage Early Capital Integration

The initial funding period will allow for the issuance of a DAO token bonus in exchange for the underlying capital (ETH, WBTC, etc.). This bonus period will decrease according to milestones over a 3-month initial funding period, thereby incentivizing early deployment.

The token strike price will increase at each interval after reaching a specified block height.

|**Time ≈**|**Block Height**|**Token Price**|
|---|---|---|
|0 weeks|0|$50|
|2 weeks|89000|$65|
|4 weeks|178000|$77|
|6 weeks|267000|$85|
|3 months|534000|$89|

At the launch of the curve, `a = $89.00`, the final token price.

The integration phase is a long enough period to allow awareness to develop a broad global participation and short enough to add value to early participants.

#### Security

Due to the nature of mandate selection, if a single holder owns the majority of the tokens, they can theoretically decide on their own the recipient of a mandate and act maliciously. The inclusion of a function whereby the acquisition cost increases geometrically offers sufficient protection against attacks from a well-capitalized malicious actor, and thus the bonding curve is necessary to control both a rush on the underlying capital in the event of a market downturn as well as an attack from a well-capitalized actor. If the community sells its tokens, this is a form of consent whereby the individual relinquishes their voting rights in the selection of mandates. It should simply become progressively prohibitive to attack the DAO via the minting of DAO tokens.

For a takeover via monetization, as participation is not identity-restrictive, it is not possible to purely prevent the case where an individual takes majority control of the token supply. However, capital is resistant to Sybil if it is directed via a bonding curve.

## Exit

After the initial funding period, DAO members can withdraw via the bonding curve with a predefined penalty corresponding to the illiquid position of the DAO plus the capital envisioned in the current mandate. For example, if the DAO holds 5% of its AUM in illiquid assets and the current mandate is to deploy 5% of AUM, the illiquidity penalty would be 10% in addition to falling on a bonding curve and increasing as more DAO tokens are subsequently burned over a short given time period.

### Last Man Standing

Inherent in the design of the DAO is a principle favoring the last man standing. Redeeming parties effectively exit their private allocation plus the current mandate as a penalty, thus leaving the remaining members of the DAO with a greater share of the private assets. If the Venture League does its job properly over time, these assets should be the most profitable assets in the long term. Thus, the onboarding and exit structure favors first-in, last-out members of the DAO. If a member maintains their position through all the ups and downs and liquidity crises of crypto, that member should be rewarded with the best-performing position.

### Token Burning/Exchange

The Treasury provides the liquidity ratio via an oracle. Thus, the burn function is not activated until a mandate has been selected and there is an entity to fill that role.

#### Example

One might imagine a scenario where the Treasury holds $100 million and 10%, or $10 million, is held in illiquid assets. At this point, the Burn Price is 90% of the NAV. If one sells at the curve, they receive 90% of their share of the NAV. After all, a token represents the fractional ownership of all underlying assets. The rest is burned in the purest sense, and the remaining token holders now own a higher proportion of illiquid assets than before. In other words, the same token represents more illiquid assets than before.

If individuals continue to sell, liquid assets are exchanged to provide liquidity for redemptions. This, in turn, increases the liquidity discount as the inverse of the liquidity ratio.

## Community

The central element of the Venture League's value proposition is to leverage the community both for deal flow sourcing and for verifying potential deals. Leveraging new social tools that allow teams to maintain a certain level of privacy when establishing a specific deal, while also allowing community feedback and proposing new opportunities.

One solution could be a multi-tiered structure where the ultimate responsibility for investment decision-making is held by a small core team, but community members are rewarded for providing information, leads, and working on behalf of all token holders. This reward can be a direct value add for portfolio companies or through the verification of potential investments, increasing the deal flow available for the core team, using SourceCred for peer review weighted by reputation.

#Governance 