# DAOs Are Not Corporations: Where Decentralization in Autonomous Organizations Matters

### September 20, 2022

_Thanks to Karl Floersch and Tina Zhen for their feedback and critiques on earlier drafts of this article._

Recently, there has been much discourse around the idea that highly decentralized DAOs do not function effectively and that DAO governance should start to resemble that of traditional corporations in order to remain competitive. The argument is always the same: highly decentralized governance is inefficient, and traditional corporate governance structures with boards of directors, CEOs, and others have evolved over hundreds of years to optimize decision-making and deliver shareholder value in a changing world. DAO idealists are naïve to assume that the egalitarian ideals of decentralization can surpass this, especially when attempts in the traditional corporate sector have had at best marginal success.

This article will explain why this position is often mistaken and will offer a different, more nuanced perspective on the areas where various types of decentralization are important. In particular, I will focus on three types of situations where decentralization matters:

- **Decentralization for better decision-making in concave environments**, where pluralism and even naive forms of compromise are likely to outperform the types of consistency and concentration that arise from centralization.
- **Decentralization to resist censorship**: applications that must continue to operate while resisting attacks from powerful external actors.
- **Decentralization as credible fairness**: applications where DAOs assume state-like functions such as providing basic infrastructure, hence traits like predictability, robustness, and neutrality are valued over efficiency.

## Centralization is Convex, Decentralization is Concave

A way to categorize decisions that need to be made is to look at whether they are **convex** or **concave**. In a choice between A and B, we wouldn’t first look at the question of A vs B itself, but rather at a higher-order question: would you prefer a _compromise_ between A and B, or a _coin flip_? In terms of expected utility, we can express this distinction using a graph:

![Graph illustrating convex vs concave decisions](file:////Users/sebaastiencailhol/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image001.png)

If a decision is concave, we would prefer a compromise, and if it is convex, we would prefer a coin flip. Often, we can answer the higher-order question of whether a compromise or a flip is better much more easily than we can answer the first-order question of A versus B itself.

Here are examples of convex decisions:

- **Pandemic response**: a 100% travel ban might prevent a virus from entering, a 0% travel ban won’t stop viruses but at least won’t hinder people, but a 50% or 90% travel ban is the [worst of both worlds](https://mobile.twitter.com/lymanstoneky/status/1321254322064236544).
- **Military strategy**: attacking on front A may make sense, attacking on front B may make sense, but splitting your army in two and attacking on both means the enemy can easily [handle the two halves one by one](https://en.wikipedia.org/wiki/Defeat_in_detail).
- **Technological choices in encryption protocols**: using technology A may make sense, using technology B may make sense, but a hybrid between the two often leads to unnecessary complexity and even risks of [interference between](https://eprint.iacr.org/2022/289.pdf) the two.

Here are examples of concave decisions:

- **Judicial decisions**: an average between two independently chosen judgments is probably fairer, and less likely to be completely ridiculous, than a random choice of one of the two judgments.
- **Funding public goods**: generally, giving $X to each of two promising projects is more effective than giving $2X to one and nothing to the other. Having money provides a much bigger boost to a project's ability to achieve its mission than moving from $X to $2X.
- **Tax rates**: due to the [quadratic mechanics of deadweight loss](https://en.wikipedia.org/wiki/Deadweight_loss), a tax rate of X% is often only a _quarter_ as damaging as a tax rate of 2X%, and at the same time _twice as effective_ at generating revenue. Therefore, moderate taxes are better than a coin flip between low or zero taxes and high taxes.

When decisions are convex, decentralizing the decision-making process can easily lead to confusion and poor compromises. When decisions are concave, however, relying on the wisdom of crowds can yield _better_ answers. In these cases, DAO-type structures with large amounts of diverse input into decision-making can make a lot of sense. Indeed, those who see the world as a more concave place in general are more likely to see a need for decentralization across a broader variety of contexts.

### Should VitaDAO and Ukraine DAO Be DAOs?

Many of the newest DAOs differ from earlier DAOs, like MakerDAO, in that while earlier DAOs are organized around _providing infrastructure_, the newer DAOs are organized around _executing various tasks centered on a particular theme_. [VitaDAO](https://vitadao.com) is a DAO funding early-stage longevity research, and [UkraineDAO](https://ukrainedao.love) is a DAO organizing and funding efforts to help Ukrainian war victims and support the Ukrainian defense effort. Does it make sense for these to be DAOs?

This is a nuanced question, and we can get a sense of a possible answer by understanding the internal workings of UkraineDAO itself. Typical DAOs tend to "decentralize" by pooling large amounts of capital into a single pool and using token-holder voting to fund each allocation. UkraineDAO, on the other hand, operates by dividing its functions into [multiple pods](https://ukraine-dao.notion.site/Ukraine-DAO-Governance-eb34313505834095ad1d54e9f464e222), where each pod operates as independently as possible. A higher governance layer can create new pods (in principle, governance can also fund pods, although thus far funding has only gone to external organizations related to Ukraine), but once a pod is created and resourced, it operates largely on its own. Internally, individual pods have leaders and operate in a more centralized manner, although they still try to adhere to a philosophy of personal autonomy.

![Image illustrating UkraineDAO structure](file:////Users/sebaastiencailhol/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image002.png)

A natural question to ask is: **does this type of "DAO" simply rename the traditional concept of a multi-layer hierarchy?** I would argue that it depends on the implementation: it is certainly possible to take this model and turn it into something that feels authoritarian in the same way that stereotypical large corporations do, but it is also possible to use the model in a very different way.

Two things that can help ensure that an organization built in this way will turn out to be meaningfully decentralized:

1. **A truly high level of autonomy for pods**, where pods accept resources from the core and are sometimes checked for alignment and competence if they want to continue receiving those resources, but otherwise act completely on their own and do not "take orders" from the core.
2. **Highly decentralized and diverse core governance**. This [does not require a "governance token"](https://vitalik.ca/general/2021/08/16/voting3.html), but it does require broader and more diverse participation within the core. Normally, wide and diverse participation heavily weighs on efficiency. But if (1) is satisfied, the pods are therefore highly autonomous and the core needs to make fewer decisions, the effects of less efficient high-level governance become weaker.

Now, how does this fit into the "convex vs concave" framework? Here, the answer is roughly as follows: **the upper level (more decentralized) is concave, the lower level (more centralized within each pod) is convex**. Giving $X to a pod is generally better than a coin flip between giving it $0 and giving it $2X, and there is no major loss in having compromises or "incoherent" philosophies guiding different decisions. But within each individual pod, it becomes much more important to have a clear and opinionated perspective guiding decisions and to be able to insist on many choices that have synergies with each other.

## Decentralization and Censorship Resistance

The most often cited public reason for decentralization in crypto is censorship resistance: a DAO or protocol must be able to operate and defend itself despite external attacks, including from large corporations or even state actors. This has already been [extensively discussed publicly](https://www.google.com/search?q=censorship+resistance+decentralization), and therefore deserves less elaboration, but there are still a few important nuances.

[The Pirate Bay](https://thepiratebay.org/) and [Sci-Hub](https://sci-hub.se/) are two of the most high-performing censorship-resistant services that many people use today. The Pirate Bay is a hybrid system: it is a search engine for BitTorrent, which is a highly decentralized network, but the search engine itself is centralized. It has a small core team dedicated to its operation, and it defends itself with the whack-a-mole strategy: when the hammer comes down, duck and reappear elsewhere. The Pirate Bay and Sci-Hub have both frequently changed domain names, relied on arbitrage between different jurisdictions, and used all sorts of other techniques. This strategy is centralized, but it has allowed both to succeed in both _defense_ and agility in product improvement.

DAOs do not operate like The Pirate Bay and Sci-H

ub; DAOs operate like BitTorrent. And **there is a reason why BitTorrent must be decentralized: it requires not only censorship resistance but also long-term investment and reliability**. If BitTorrent were shut down once a year and forced all its seeders and users to switch to a new provider, the quality of the network would degrade quickly. DAOs demanding censorship resistance should also be in the same category: they should provide a service that avoids not only permanent censorship but also mere instability and disruptions. MakerDAO (and [Reflexer DAO](https://reflexer.finance/) that manages RAI) are excellent examples of this. A DAO executing a decentralized search engine probably does not do so: you can simply create a standard search engine and use Sci-Hub-like techniques to ensure its survival.

## Decentralization as Credible Fairness

Sometimes the primary concern of DAOs is not the need to _resist_ nation-states but rather the need to _assume some functions_ of nation-states. This often involves tasks that can be classified as "maintenance of basic infrastructure." Because governments have less capacity to oversee DAOs, DAOs need to be structured to assume greater capacity to _self-oversee_. And that requires decentralization.

![Image illustrating infrastructure maintenance](file:////Users/sebaastiencailhol/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image003.jpg)

_Of course, it is impossible to come close to eliminating hierarchy and inequality of information and decision-making power altogether, etc._

Let’s take three motivating examples: algorithmic stablecoins, the [Kleros court](https://kleros.io/) and the [Optimism retroactive funding mechanism](https://medium.com/ethereum-optimism/retroactive-public-goods-funding-33c9b7d00f0c).

- **An algorithmic stablecoin DAO** is a system that uses on-chain financial contracts to create a crypto asset whose price follows a stable index, often but not necessarily the US dollar.
- **Kleros is a "decentralized court"**: a DAO whose function is to rule on arbitration issues such as "does this GitHub submission meet acceptable criteria for this on-chain bounty?"
- **The Optimism retroactive funding mechanism** is a component of the [Optimism DAO](https://community.optimism.io/docs/governance/) that retroactively rewards projects that have added value to the Ethereum and Optimism ecosystems.

In all three cases, there is an inevitable need to make subjective judgments, which cannot be done automatically via a piece of on-chain code. In the first case, the goal is simply to get reasonably accurate measurements of a certain price index. If the stablecoin follows the US dollar, you just need the ETH/USD price. In the case of hyperinflation or some other reason to abandon the US dollar, the DAO stablecoin may need to manage a reliable calculation of CPI on-chain. Kleros involves making inevitably subjective judgments on any arbitrary question it is presented with, including whether questions submitted should be [dismissed as "unethical."](https://snapshot.org/#/kleros.eth/proposal/QmeJjVUgF2o9vrmjx8szvher8Ca6iR2TuTwHGR9KbM1gC9). The retroactive funding of Optimism is tasked with one of the most open subjective questions: which projects have done the most useful work for the Ethereum and Optimism ecosystems?

All three cases have an inevitable need for "governance," and quite robust governance at that. In all cases, governance being attackable, from the outside or the inside, can easily lead to very large problems. Finally, governance not only needs to be _robust_, it must _credibly convince_ a broad and skeptical public that it is robust.

### The Achilles' Heel of Algorithmic Stablecoins: The Oracle

Algorithmic stablecoins depend on oracles. For a smart contract on-chain to know whether to target the value of DAI at 0.005 ETH or 0.0005 ETH, it needs a mechanism to learn the information (off-chain) of what the ETH/USD price is. And in fact, this "oracle" is the main place where an algorithmic stablecoin can be attacked.

This leads to a security conundrum: an algorithmic stablecoin cannot safely hold more collateral, and thus cannot issue more units, than the market capitalization of its speculative token (e.g., MKR, FLX...), because if it does, it becomes profitable to buy half of the supply of speculative tokens, use those tokens to control the oracle, and steal funds from users by feeding it bad oracle values and liquidating them.

Here is a possible alternative design for a stablecoin oracle: [add a layer of indirection](https://ethresear.ch/t/a-not-quite-cryptoeconomic-decentralized-oracle/6453). Quoting the message from ethresear.ch:

We set up a contract where there are 13 "providers"; the response to a request is the median of the responses returned by these providers. Each week, there is a vote, where oracle token holders can replace one of the providers...

The security model is simple: if you trust the voting mechanism, you can trust the output of the oracle, unless 7 providers are corrupted at the same time. If you trust the current set of oracle providers, you can trust the output for at least the next six weeks, even if you absolutely do not trust the voting mechanism. Therefore, if the voting mechanism is corrupted, participants in all applications relying on the oracle will have time to make an orderly exit.

Notice the very uncorporate nature of this proposal. It involves _removing_ from governance the ability to act quickly and intentionally spreading responsibility for the oracle across many participants. This is valuable for two reasons. First, it makes it harder for outsiders to attack the oracle and for new token holders to quickly take control of the oracle. Second, it makes it harder for _the oracle participants themselves_ to collude to attack the system. It also mitigates the _oracle extractable value_, where a single provider can intentionally delay publication to personally profit from a liquidation (in a multi-provider system, if one provider does not publish immediately, others will soon).

### Fairness in Kleros

The "decentralized court" system Kleros is a truly valuable and important piece of infrastructure for the Ethereum ecosystem: [Proof of Humanity](https://www.proofofhumanity.id/) uses it, various "smart contract bug insurance" products use it, and many other projects connect to it as a sort of "last resort decision."

Recently, the public has been concerned about whether the platform's decision-making is fair or not. Some participants have made arguments, trying to claim payment from decentralized smart contract insurance platforms that they believe they deserve. The most famous of these cases is perhaps [Mizu's report on case #1170](https://ipfs.kleros.io/ipfs/QmYr4e5jpNSjeUBw9BbM2G156eMmxcaMaknm17JHm8WP8G/1170-6a.pdf). The case escalated from a minor language interpretation dispute to a larger scandal due to the accusation that insiders from Kleros itself were coordinating efforts to cast a large number of tokens to push the decision in their desired direction. One participant in the debate wrote:

The incentive-based decision-making process of the court... is apparently corrupted by a single developer with a very large stake (25%) in the courts.

Of course, this is only one side of an issue in a broader debate, and it is up to the Kleros community to determine who is right or wrong and how to respond. But zooming in on the issue of this individual case, what is important here is the extent to which _the entire proposition_ of value of something like Kleros depends on its ability to convince the public that it is strongly protected against this type of centralized manipulation. For something like Kleros to be trustworthy, it seems necessary that no single individual holds a 25% stake in a top-level court. Whether through a more broadly distributed token offer or by increasing the use of non-token-based governance, a form of more credible decentralized governance could help Kleros avoid such concerns altogether.

### Retroactive Funding Optimism

The retroactive funding outcomes of the first foundation round of Optimism were chosen by quadratic voting among 24 "badge holders." The second round will likely use a larger number of badge holders, and the ultimate goal is to move to a system where [a much larger number of citizens](https://community.optimism.io/docs/governance/) control the allocation of retro funds, probably through a multi-level mechanism involving sorting, subcommittees, and/or delegation.

There has been internal debate on whether to have more or fewer citizens: should "citizen" really mean something closer to "senator," an expert contributor who deeply understands the Optimism ecosystem, should it be a post assigned to _almost anyone_ who has significantly participated in the Optimism ecosystem, or somewhere in between? **My personal position on this issue has always been in favor of a larger number of citizens, solving governance inefficiencies with second-level delegation rather than adding centralization embedded in the governance protocol. One of the main reasons for my stance is the possibility of insider trading and conflicts of interest.**

The retroactive funding mechanism of Optimism has always been intended to be coupled with a prospective speculative ecosystem: public goods projects that need funding _now_ could sell "project tokens," and anyone who buys project tokens becomes eligible for significant compensation funded retroactively later. But this mechanism works well primarily depending on the proper functioning of the retroactive funding part, and is _very_ vulnerable to corruption of the retroactive funding mechanism. A few examples of attacks:

- If

 a group of people decides how it will vote on a project, it can buy (or if it is too expensive, short) its project token before publishing the decision.
- If a group of people knows it will vote on a specific project later, it can buy the project token earlier and then intentionally vote in favor of it even if the project does not actually deserve funding.
- Funding decision-makers can accept bribes from projects.

Generally, there are three ways to address these types of corruption and insider trading issues:

- _Retroactively punish_ malicious decision-makers.
- Proactively filter for _higher-quality decision-makers_.
- Add _more_ decision-makers.

The corporate world typically focuses on the first two, using financial oversight and judicious sanctions for the first, and in-person interviews and background checks for the second. The decentralized world has less access to such tools: project tokens are likely to be traded anonymously, DAOs have at best limited recourse to external judicial systems, and the remote and online nature of projects and the desire for global inclusion make it harder to conduct background checks and informal "smell tests" in person for character. Therefore, the decentralized world must place more weight on the third technique: spreading decision-making power among _more_ decision-makers so that each individual decision-maker has less power, and collusions are more likely to be reported and revealed.

## Should DAOs Learn More from Corporate Governance or Political Science?

Curtis Yarvin, an American philosopher whose main "big idea" is that companies are much more efficient and optimized than governments and therefore we should improve governments by making them more like companies (for example, moving away from democracy and closer to monarchy), recently wrote an article expressing [his thoughts on how DAO governance should be designed](https://graymirror.substack.com/p/optimal-autonomous-organizations). Unsurprisingly, his answer involves borrowing ideas from traditional corporate governance. From his introduction:

Instead, the basic design of the Anglo-American limited liability company has remained largely unchanged since the dawn of the industrial revolution—which, according to a contrarian historian, might have actually been a corporate revolution. If the design of shareholding is not perfectly optimal, one can expect it to be almost optimal.

If there is a categorical difference between these two types of organizations—we might call them first-order (sovereign) and second-order (contractual) organizations—it seems that this year’s society has very effective second-order organizations, but not very effective first-order organizations.

Therefore, we probably know more about second-order organizations. Thus, when designing a DAO, we should start from corporate governance, not political science.

Yarvin's message is quite correct in identifying the key difference between "first-order" (sovereign) and "second-order" (contractual) organizations—in fact, this exact distinction is precisely the subject of the section of my own message above on credible justice. However, Yarvin's message makes an important and surprising error immediately afterward, turning to say that corporate governance is the best starting point for how DAOs should operate. The error is surprising because the logic of the situation seems to almost directly imply the exact opposite conclusion. **Because DAOs have no sovereign above them and are often explicitly in the realm of providing services (like currency and arbitration) that are generally reserved for sovereigns, it is precisely the design of sovereigns (political science), rather than corporate governance, from which DAOs have more to learn.**

To Yarvin's credit, the second part of his article advocates for a "hourglass" model that combines a layer of decentralized alignment and accountability with a layer of centralized management and execution, but that is already an admission that DAO design must learn at least as much from first-order organizations as it does from second-order organizations.

Sovereigns are inefficient, and corporations are efficient for the same reason that group theory can prove many things, but abstract group theory can prove much less: **corporations fail less and accomplish more because they can make more assumptions and have more powerful tools to work with**. Corporations can rely on their local sovereign to stand up for them if need be, as well as to provide an external legal system to stabilize their incentive structure. In a sovereign country, on the other hand, the greatest challenge is often figuring out what to do when the incentive structure is under attack and/or at risk of collapsing entirely, with no external leviathan ready to support it.

The greatest problem in designing successful governance systems for sovereigns is perhaps what [Samo Burja calls "the succession problem"](https://samoburja.com/the-succession-problem/): how to ensure continuity as the system moves from being led by one group of humans to another when the first group retires. Corporations, Burja writes, often simply do not resolve the problem:

Silicon Valley gets excited about "disruption" because we are so accustomed to the succession problem remaining unresolved within discrete institutions like corporations.

DAOs will eventually need to resolve the succession problem (in fact, given the prevalence of the "get rich and retire" model among early crypto users, some DAOs are already facing succession problems). Monarchies and corporate-type forms often struggle to resolve the succession problem because the institutional structure is deeply tied to the habits of a specific person, and either proves difficult to pass on, or there are very high-stakes struggles over who to hand it over to. More decentralized political forms like democracy at least have a theory of how smooth transitions can occur. Therefore, I would argue that for this reason as well, DAOs have more to learn from more liberal and democratic political science schools than from corporate governance.

Of course, DAOs will need to accomplish specific complex tasks in certain cases, and using corporate-type forms to accomplish those tasks could indeed be a good idea. Additionally, DAOs must manage unexpected uncertainty. A system that was supposed to operate stably and immutably around a set of assumptions, when faced with an extreme and unexpected change in those circumstances, needs some sort of courageous leader to coordinate a response. A prototypical example of this is stablecoins managing a collapse of the US dollar: what happens when a stablecoin DAO that has evolved around the assumption that it is simply trying to peg itself to the US dollar suddenly faces a world where the US dollar is no longer a viable asset to be pegged to, and a rapid shift to some sort of CPI is necessary?

![Stylized diagram of internal experience of the RAI ecosystem undergoing an unexpected transition to an CPI-based regime if the USD ceases to be a viable benchmark asset](file:////Users/sebaastiencailhol/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image004.png)

Here, corporate governance-inspired approaches may seem better, as they provide a ready-made model for responding to such a problem: the founder orchestrates a pivot. But it turns out that the history of political systems also offers _a scheme well-suited to this situation_, which overlaps with the question of reverting to a decentralized mode once the crisis has passed: the Roman republican custom of [electing a dictator](https://en.wikipedia.org/wiki/Roman_dictator) for a temporary mandate to respond to a crisis.

**Realistically, we probably only need a small number of DAOs that resemble constructs from political science more than anything from corporate governance. But those are the ones that really matter.** A stablecoin does not need to be efficient; it must above all be stable and decentralized. A decentralized court is similar. A system directing funding for a particular cause—whether it is Optimism retroactive funding, VitaDAO, UkraineDAO, or something else—is optimized for a purpose that is much more complicated than maximizing profit, and therefore a form of alignment other than shareholder profit is necessary to ensure that it keeps using funds for the intended purposes.

By far, the largest number of organizations, even in a crypto world, _will be_ second-order "contractual" organizations that ultimately rely on these first-order giants for support, and for these organizations, much simpler and leader-focused governance emphasizing agility will often make sense. But this should not distract from the fact that the ecosystem would not survive without certain forms of decentralized _non-corporate_ structures keeping everything stable.

[https://vitalik-eth-limo.translate.goog/general/2022/09/20/daos.html?_x_tr_sl=en&_x_tr_tl=fr&_x_tr_hl=fr&_x_tr_pto=wapp](https://vitalik-eth-limo.translate.goog/general/2022/09/20/daos.html?_x_tr_sl=en&_x_tr_tl=fr&_x_tr_hl=fr&_x_tr_pto=wapp)

#Governance 