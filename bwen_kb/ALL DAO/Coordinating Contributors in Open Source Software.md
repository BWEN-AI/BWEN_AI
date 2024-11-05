## **Coordinating Contributors in Open Source Software**

#### **What is Open Source Software?**

Open Source Software (OSS) is software published publicly for anyone to view, copy, or use. Since the 1980s, online communities have formed around various open-source projects to coordinate contributions and advance these projects. Most contributors are unpaid, but open-source projects receive sustained contributions and have an impact that often surpasses that of the world’s largest companies.

Open-source projects sometimes have thousands of community members contributing to the source code and millions of users who rely on the software. Let’s explore the challenges OSS communities face, how they coordinate contributors, and how they make decisions.

#### **Coordination Challenges in OSS**

The most common issue faced by OSS projects is not a shortage of contributors but rather new contributors overwhelming project maintainers who onboard individuals, triage contributions, and sift through opinions. As Nadia Eghbal recounts in her book [Working in Public: The Making and Maintenance of Open Source Software](https://translate.google.com/website?sl=en&tl=fr&hl=fr&client=webapp&u=https://www.amazon.ca/Working-Public-Making-Maintenance-Software/dp/0578675862), the main problem faced by open source is the lack of attention from its main maintainers.

“The problem maintainers face today is not how to get more contributors, but how to manage a high volume of frequent, low-touch interactions. These developers are not creating communities; they are air traffic controllers… it’s not code bloat but user bloat competing for the maintainer’s attention that has made the work untenable for maintainers today.”

The advantage of open source is that it is permissionless, allowing professionals with diverse talents from all over the world to work together to advance the project. On the other hand, permissionless means you don’t get the filtering that regular companies would do to select individuals with the right experience to participate. The result is a cacophony of voices that core contributors must spend their time sifting through, which often does more harm than good.

There are two approaches by which OSS projects overcome this coordination problem. One is the more common solution of limiting the time and attention that core contributors spend triaging community contributions. The second is the opposite—to consciously spend resources onboarding and upgrading as many contributors as possible so they can eventually help bring new community members on board.

#### **Public but Non-Participatory**

Python founder Guido van Rossum stepped down in 2018 after 27 years as the leader of the Python community. [Van Rossum’s departure](https://translate.google.com/website?sl=en&tl=fr&hl=fr&client=webapp&u=https://www.techrepublic.com/article/programming-languages-pythons-pep-572-feature-that-drove-founder-to-quit-edges-nearer/) was driven by a very controversial decision the community was trying to make regarding syntax changes in the language. The decision involved a mix of voices from a myriad of individuals with limited context, proving taxing for the project’s core contributors.

After his departure, [Van Rossum argued](https://translate.google.com/website?sl=en&tl=fr&hl=fr&client=webapp&u=https://lwn.net/Articles/759557/) that participatory decision-making, where all community members can weigh in on the decision, doesn’t scale. Instead, a project can maintain the OSS ethos of building in public but should restrict decision-making participation to a small group of individuals with the appropriate context. Otherwise, uninformed voices will slow down decision-making, lead to poor decisions, and, worse, exhaust valuable contributors who have the most context in the project.

Van Rossum’s suggestion to limit the amount of participatory decision-making is often the most popular way for open-source projects to avoid being overwhelmed by the flood of contributions.

#### **Hero Contributors**

If we look at contribution patterns in OSS, it suggests that most projects are not highly participatory. In a study by [Majumeder et al.](https://translate.google.com/website?sl=en&tl=fr&hl=fr&client=webapp&u=https://arxiv.org/pdf/1904.09954.pdf) examining over 1,000 open-source projects, they found that in 85% of analyzed projects, 5% of contributors account for nearly 95% of all code commits and communications. The reality is that in most open-source projects, a few “hero contributors” do the majority of the work, while there is a long tail of occasional contributors.

[Bootstrap](https://translate.google.com/website?sl=en&tl=fr&hl=fr&client=webapp&u=https://getbootstrap.com/) is an open-source project that is a great example of hero contributors carrying the project. It is one of the most popular projects on the internet, used by [22.1% of all websites](https://translate.google.com/website?sl=en&tl=fr&hl=fr&client=webapp&u=https://w3techs.com/technologies/details/js-bootstrap), but if we look at the number of commits per contributor, we see that a handful of contributors represent the majority of the project’s contributions.

[](https://translate.google.com/website?sl=en&tl=fr&hl=fr&client=webapp&u=https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%253A%252F%252Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%252Fpublic%252Fimages%252Fecccae4c-6210-499d-8771-44896094f247_549x270.png)

Source: [Working in Public by Nadia Eghbal](https://translate.google.com/website?sl=en&tl=fr&hl=fr&client=webapp&u=https://www.amazon.ca/Working-Public-Making-Maintenance-Software/dp/0578675862)

Making the project less participatory might be the most common way for projects to avoid overburdening the core team contributors, but it’s not without drawbacks.

#### **Side Effects of Reducing Community Participation**

The most obvious side effect of not having a large and diverse group of core contributors is a lack of project resilience. When knowledge around a project is localized with a few select developers, the project is suddenly threatened if a few of those developers leave. In fact, the [bus factor](https://translate.google.com/website?sl=en&tl=fr&hl=fr&client=webapp&u=https://en.wikipedia.org/wiki/Bus_factor), defined by the number of core contributors who get hit by a bus before the project is in trouble, is a heuristic that projects use to measure a project’s resilience.

A more insidious effect of having a small number of core contributors is security issues. If a project does not actively grow its contributor base, when team members leave, certain parts of the library do not receive the maintenance they need. As a result, it’s easy for security risks to start appearing and even for saboteurs to actively exploit projects, to the detriment of the software’s large user base.

In 2018, Dominic Tarr, an active open-source developer, gave an unknown individual commit rights to a popular software module library, [EventStream](https://translate.google.com/website?sl=en&tl=fr&hl=fr&client=webapp&u=https://github.com/dominictarr/event-stream), which he had developed but was no longer actively maintaining. The unknown person asked to take over maintenance of the library but then inserted a malicious bug that [targeted users](https://translate.google.com/website?sl=en&tl=fr&hl=fr&client=webapp&u=https://thehackernews.com/2018/11/nodejs-event-stream-module.html) using the bitcoin wallet Copay. Projects without deep contributors and/or core contributors with active commitments to other projects make such backdoor exploits much more likely to occur.

#### **Node.js: Public and Participatory**

[Node.js](https://translate.google.com/website?sl=en&tl=fr&hl=fr&client=webapp&u=https://nodejs.org/en/about/) is a project that has adopted a liberal contribution policy to keep pace with its growing contributor base. Instead of limiting the number of contributors, Node’s strategy is to spread context as widely as possible so that new contributors quickly gain the context to make contributions and onboard others. Through this strategy, Node has become one of the most vibrant OSS user and contributor ecosystems.

#### **Node.js Governance**

With a large contributor base like Node, more formal organizational hierarchies emerge. For technical contributions, the Node.js project uses a simple governance hierarchy to guide the project and make technical decisions.

**Organizational Structure:**

[Structure Image](https://translate.google.com/website?sl=en&tl=fr&hl=fr&client=webapp&u=https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%253A%252F%252Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%252Fpublic%252Fimages%252F2f0e1e1a-1a80-469d-9623-2361d0028e8b_1294x662.png)

**Roles:**

- **Contributors** are anyone who creates or comments on an issue or pull request.
- **Collaborators** are contributors who have made at least one non-trivial contribution and who have received write access to the project’s repository.
- **Technical

 Steering Committee (TSC) members** are collaborators who have been elected to the TSC and have voting rights at TSC meetings.

**Working Groups**

Most technical contributions merged into the codebase are done by groups of collaborators who work together in working groups. Each working group focuses on a particular area and makes decisions quite autonomously. There are several working groups, all working independently on initiatives that move the Node project forward.

Decision-making within working groups is done by [lazy consensus](https://translate.google.com/website?sl=en&tl=fr&hl=fr&client=webapp&u=https://medlabboulder.gitlab.io/democraticmediums/mediums/lazy_consensus/). In short, with lazy consensus, proposals within a group can be presumed accepted unless explicit objections arise. Objections must be presented with concrete measures to address that objection. Since it is easier for team members to agree by doing nothing rather than proposing alternatives, inertia favors advancing the discussion.

**Technical Steering Committee (TSC)**

The TSC helps guide the overall direction of the Node project by providing guidance on items such as voting on projects that can form a working group, setting release deadlines, and making technical decisions when working groups are unable to decide.

The TSC makes decisions through a [consensus-seeking process](https://translate.google.com/website?sl=en&tl=fr&hl=fr&client=webapp&u=https://en.wikipedia.org/wiki/Consensus-seeking_decision-making%23:~:text%3DConsensus%252Dseeking%2520decision%252Dmaking%2520(,procedure%2520if%2520consensus%2520appears%2520unattainable) where the group attempts to reach consensus; if consensus cannot be reached, a majority vote is then called. The possibility of voting encourages team members to work toward consensus.

#### **Characteristics of a Healthy Open Source Project**

In his blog post [Healthy Open Source](https://translate.google.com/website?sl=en&tl=fr&hl=fr&client=webapp&u=https://medium.com/the-node-js-collection/healthy-open-source-967fa8be7951), Mikeal Rogers, who has contributed to Node.js since its early days, describes what a healthy open-source project should look like. He argues that when a healthy OSS project grows, the users, contributors, committers (called “collaborators” in the Node project), and the technical committee should grow proportionately to one another.

In contrast, a project’s community looks like this if the project does not experience healthy growth:

[Image of Unhealthy Community Growth](https://translate.google.com/website?sl=en&tl=fr&hl=fr&client=webapp&u=https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%253A%252F%252Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%252Fpublic%252Fimages%252F62d31526-cc14-4208-b78f-be2fbb6ed093_596x571.png)

In this latter scenario, the project’s committers do not make a concerted effort to continuously onboard new contributors, which ultimately leads to more serious problems for the project and its users. As Rogers says,

“We know what happens to unhealthy projects over a long enough period; more maintainers leave, contributions eventually drop, and if we’re lucky, users abandon it. When we’re not so lucky, adoption continues, and years later we are left with security and stability issues in widely adopted software that can’t be effectively maintained.”

#### **Liberal Onboarding Policies**

To maintain a healthy open-source community, Node diverges from most OSS projects by adopting a more liberal contribution policy. Their philosophy is to onboard contributors to become collaborators with commit access after their first non-trivial contribution. Additionally, Node seeks to invest in onboarding a diverse set of talents. Due to git’s ability to roll back code, review code, and restrict access for high-level decisions, any mistakes that occur can be avoided or easily reversed. Over time, onboarding new talent becomes easier as the organization does not rely on a handful of maintainers with all the context.

When liberal onboarding is successful, a project has independent teams with the skills and context needed to make decisions. The project does not need to require senior project members to weigh in on every hard decision and can operate more decentrally.

#### **Summary of Lessons Learned from OSS**

- In OSS, open contributions often overwhelm core contributors, so the most common solution is for core contributors to build in public but restrict community contributions.
- Onboarding new contributors is costly, but it increases the number of individuals with context and can help grow the project sustainably.
- Context and expertise are crucial to productive discussion, especially when working openly. Without this, it results in slow execution, poor decisions, and contributor burnout.
- Lazy consensus and consensus-seeking decision-making help reduce decision bottlenecks and encourage discussions to move toward resolution.
- A small group of core contributors reduces the burden of onboarding and coordination, but there is a trade-off with the project’s resilience and the long-term security of the software.

#DAO 