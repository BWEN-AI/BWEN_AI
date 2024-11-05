
The rise of DeFi in 2020 led to rapid innovation and experimentation in decentralized governance within DAOs. [Compound Finance’s Governor Alpha and Bravo framework](https://compound.finance/) introduced ready-made on-chain governance, helping drive the proliferation of DAOs. On-chain governance offers benefits like transparency and censorship resistance, but a purely on-chain governance structure can be challenging for protocols that need agility in decision-making. Yearn Finance is a case study of a project that successfully built a governance model that primarily operates off-chain to meet the needs of a fast-growing startup protocol.

### **Yearn Finance**

Most crypto projects follow the [progressive decentralization model](https://a16z.com/2020/01/09/progressive-decentralization-crypto-product-management/), where the founding team achieves product-market fit for a protocol and eventually decentralizes the organization via a DAO. Yearn is an interesting case because its founder, Andre Cronje, launched the protocol and token (YFI) and handed complete control of the protocol to the community from the beginning. This required the community to quickly establish governance to make both high-level protocol decisions and manage daily operations.

#### **Constrained Delegation Governance**

Early in Yearn’s history, the community realized that the organization’s speed required more localized decision-making. Yearn’s first governance version was called [Constrained Delegation Governance](https://yips.yearn.finance/YIPS/yip-41), which involved a [multisig wallet](https://www.coindesk.com/tech/2020/11/10/multisignature-wallets-can-keep-your-coins-safer-if-you-use-them-right/) that enabled elected individuals to approve a defined set of protocol decisions.

The multisig was authorized to perform essential day-to-day operations decisions on personnel and budgets, including:

- Determining and distributing protocol grants
- Allocating community grants
- Approving legal and DAO consultation grants
- Hiring for key roles as budgeted
- Engaging security firms
- Hiring analytics firms
- Continuing strategy changes in yVault until a dedicated team could take over
- Facilitating UI and front-end development
- Supporting business development and integrations

The team took care to limit the power of the multisig, which could not make major protocol-level decisions, such as token emissions or adopting governance proposals.

Yearn began with minimal viable governance—a single multisig model with limited powers. Starting with a basic structure allowed Yearn to observe how the multisig model met community needs, then adjust governance to respond to those needs, leading to its more complex model today.

Building governance is an iterative process. Governance tools evolve, DAO goals may shift, and most importantly, the community changes. It’s the operators' job to understand the community’s needs as they inevitably change and shape the governance model to accommodate those changes.

#### **Transitioning to a Multi-DAO Structure**

Eventually, Yearn outgrew the single multisig model and transitioned to a [multi-DAO structure](https://gov.yearn.finance/t/yip-61-governance-2-0/10460) where YFI token holders delegate decision-making powers to teams that function as DAOs. In this new structure, protocol governance is organized around three main groups: YFI holders, yTeams, and the Multisig.

**YFI Holders**

YFI holders are community members who own the token. They can submit proposals for high-level protocol decisions not covered by existing yTeams, such as treasury spending or adjusting protocol fees charged to users.

Proposals are discussed on forums before a formal vote, similar to Compound’s process. Instead of on-chain voting, Yearn uses [Snapshot](https://docs.snapshot.org/), an off-chain platform for simulating on-chain votes.

**yTeams**

Yearn’s protocol has contributors working in yTeams, which are smaller DAOs focused on specific operational areas. For example, yDevs is a team of engineers handling protocol engineering, while yPeople manages DAO compensation. These teams operate quite autonomously and have decision-making power in their areas of expertise.

Each yTeam has a budget approved by YFI holders, managed through a [multisig wallet](https://www.coindesk.com/tech/2020/11/10/multisignature-wallets-can-keep-your-coins-safer-if-you-use-them-right/) controlled by the team. yTeams were designed to enable contributors to form organic teams around protocol needs and operate independently. Contributors within these teams can make quick day-to-day decisions, allowing the organization to scale faster.

**Multisig**

All on-chain transactions from Yearn proposals or yTeams must be approved by multisig wallet signers. Yearn has nine multisig members elected by YFI holders, with six out of nine signatures required to approve a transaction. The multisig also has the power to veto any transaction if further review is deemed necessary.

[Learn more about Yearn's governance](https://davincifi.substack.com/p/examining-how-we-govern-collectives?s=r)

#DAO 