## DAO Structure

The DAO is a decentralized software stack based on the [Aragon framework](https://aragon.org) that enables the secure pooling of capital and its intelligent deployment to fund crypto networks, dapp projects, businesses, and commercial opportunities. The underlying capital pool will be a balanced portfolio of whitelisted assets such as ETH, WBTC, and other high-quality crypto assets. The pool should be protected and efficiently managed by the leagues operating under strict, time-limited mandates with safety nets regarding performance.

Note: Although trustless cross-chain value transfer has not yet matured, over time, members would expect the DAO to also hold off-chain assets and crypto assets that are not native to Ethereum, potentially via the service of a bonded custody league.

At the genesis of the DAO, capital may be integrated during an initial funding period. After the closure of the initial funding, entry and redemption from the DAO will occur relative to a bonding curve that regulates the price of the DAO token in terms of the underlying capital.

The DAO will make capital deployments based on a set of permissions defined in a multi-sig between the Venture League, the Treasury League, and the Compliance League, subject to the permissions outlined in each group's respective mandates. For capital to be sent by the DAO to a potential investment, the Venture League, the Treasury League, and the Compliance League must all sign a multi-signature transaction, and the given deployment must satisfy the permissions and restrictions of the mandates issued to each league.

### DAO Specification

The DAO consists of a series of sub-DAOs of the League using a specific hierarchy of permissions within Aragon. These DAOs are organized by assigning roles and permissions relative to each other. We categorize the entire system as a DAO, where the top-level DAO is called the **Parent** and the sub-DAOs are called **Leagues**, which are technically full DAOs, subject to nested permissions. In principle, the Parent DAO will push all activities during the Mandates down to the League level. Each DAO will require its own design process and specifications, but we propose a brief illustrative overview below.

### Parent_DAO

The Parent_DAO is ultimately responsible for the members' funds in the DAO, and this is where all governing members vote on the selection of mandates. Voting in the Parent_DAO is limited to once per mandate (subject to the Benny Switch mechanism described below). Daily maintenance is delegated to the underlying DAOs of the League: Venture League, Treasury League, Compliance League, Dev League, etc., depending on the permissions of their mandates. This is similar to how a board of directors has authority in a company, but the day-to-day operations are managed by functional departments. The Parent_DAO does not have direct access to spend its own funds; rather, this functionality is managed via the multi-sig between the Treasury League, the Venture League, and the Compliance League.

### Spend_Funds

A multi-sig that has the power to execute token transfers of the DAO tokens via DAO token inflation. The only members are the Treasury League, the Venture League, and the Compliance League. The Treasury League is authorized to vote to mint tokens and transfer them, while the Compliance League will simply ensure that the values are correct but does not have the capacity to vote. 3/3 of the votes are required to transfer funds. Each payment must have a delay before execution (e.g., 14 days) to allow for an appeal process within the Venture League.

### Treasury_League

Responsible for managing the underlying capital of The DAO, specifically rebalancing the underlying capital portfolio and executing payments based on inflation. This would be a bonded league (i.e., it displays a guaranteed position relative to the current mandate's capital) that would have specific permissions regarding maintaining portfolio balance, co-signing for capital deployments in agreements proposed by the Venture League, and other management of liquid tokens such as third-party custody of certain assets.

### Venture_League

The Venture League would be the team responsible for sourcing deal flow and identifying attractive investment opportunities. This League could be well suited for open community involvement using appropriate social coordination tools while ensuring the confidentiality of certain transactions (see Community, below).

### Compliance_League

Responsible for ensuring that the Spend_Fund votes are accurate, that the Venture_League and Treasury_League adhere to their expected permissions, and that the capital deployment in question is compliant with DAO principles. The Compliance_League will also be responsible for whitelist mandate proposals with Aragon Court after the initial mandate. It would be the third signature in the multi-sig to deploy capital during the mandate.

**_Note on Compliance:_** Questions are naturally raised regarding DAO compliance and the potential classification of the token as a security by the SEC in the United States. We note that having a team work with regulators toward acceptable solutions is a compliance effort far greater than what the industry has seen from DAOs before. Given that the DAO token is just a basket of tokens that are not already considered securities, the launch of the DAO network and token issuance are a) executed with a sufficient level of decentralization as per the [SEC](https://www.sec.gov/news/speech/speech-hinman-061418), b) not a joint enterprise or an expectation of profits from the efforts of others because the leagues do not yet exist or are formed, and c) the inflation mechanism equates to a payment token under FINMA's Swiss regulations. At the time of conceptualizing the DAO, most of the authors of this specification are optimistic that the work can be done by a competent and well-meaning compliance league to achieve reasonable compliance without sacrificing the above principles, namely that the DAO remains an "equity capital" pool into which participants can freely enter and exit reasonably without discrimination.

### Dev_League

This league would be responsible for due diligence on potential acquisitions, developing software coordination between leagues, and value-added services for portfolio projects in which the Dev_League would naturally be compensated.

### Oracle_League

Initially, an Oracle will be required by the DAO for relevant price feeds and striking NAV, especially given future illiquid assets. This oracle would also be linked, and there is a range of decentralized oracle providers now available, such as [Chainlink](https://chain.link), which could potentially provide this service today with minimal integration available for Aragon DAOs.

### Original_Guild

During the first mandate, the Original_Guild (OG) operates as a proxy using Aragon Court to whitelist original proposals. The OG has pseudo-privileges over the other leagues, meaning they should be able to freeze a DAO to prevent any malicious behavior. These activities can be executed by Aragon Court while ensuring appropriate decentralization in launching and executing the DAO. The Parent_DAO has similar privileges over the OG so that the Parent_DAO can freeze the OG_DAO in case of malicious behavior. After the initial mandate, the OG will be dissolved, and the whitelist of future mandate proposals will be executed by Aragon Court in collaboration with the Compliance League.

## Mandates & Proposals

A mandate is:

1. A set of permissions for executing specified activities
2. Use of a specific set of resources
3. Over a specifically defined time period
4. Aiming to create a specific outcome.

For example, a mandate could consist of deploying 5% of the DAO's capital (via inflation) over a year, and the leagues would share a service payment of around 2% of total AUM plus 20% of the profit generated by the investments made during the given mandate, at the time of their harvest. Thus, a capital pool of $25 million would generate fixed fees of $500,000, which is sufficient to cover salaries for some teams operating the leagues as well as financial incentives for community participants, with considerable upside potential if the given mandate produces compelling investments that lead to aberrant returns. Once a mandate is approved, the portion of capital allocated is added to illiquid assets for liquidity provisions.

### Proposals

Proposals are an offer from one or more leagues to execute a mandate, with specific deliverables, permissions, a timeline, and an outcome. Proposals can be made by any League or group of Leagues acting in concert. During the proposal era, anyone can make a mandate proposal. However, only a set of 4 to 5 potential whitelisted mandates will actually be voted on by the DAO members.

### Whitelist Proposals

Any group can eventually self-organize into a league and make a proposal to the DAO to execute a mandate. A League or group of Leagues proposing a Mandate must also be able to meet the necessary conditions to execute their proposal. For the first mandate decision, the Original_Guild will be formed using Aragon Court to whitelist proposals, reducing the number of proposals to a reasonable multiple-choice question.

### Benny Switch

A Benny Switch (named after [Benedict Arnold](https://en.wikipedia.org/wiki/Benedict_Arnold)) is a means for DAO members to revoke permissions from a malicious league or group of leagues in conditions of emergency or attack. The Benny Switch engages if a minimum quorum of DAO members signals with their tokens that a Benny Switch vote is necessary. Then a subsequent vote occurs whereby a minimum quorum votes in favor of revoking permissions from a specific league or terminating the entire mandate immediately. Given the issues related to obtaining a minimum quorum in crypto networks associated with on-chain governance in the past, we would propose an adaptive quorum bias in which the minimum quorum required changes based on voting participation.

#DAO 