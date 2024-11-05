Solution 1 - Limited Governance

One possible mitigation of the above problems, which is already being tried to varying degrees, is to impose limits on what coin-based governance can do. There are several ways to achieve this:

- **Use on-chain governance only for applications, not base layers**: Ethereum already does this, as the protocol itself is governed by off-chain governance, while DAOs and other applications are sometimes (but not always) governed by on-chain governance.

- **Limit governance to fixed parameter choices**: Uniswap does this, as it only allows governance to affect (i) token distribution and (ii) a 0.05% fee in the Uniswap exchange. Another good example is the [“non-governance” roadmap of RAI](https://docs.reflexer.finance/ungovernance/governance-minimization-guide), where governance controls fewer and fewer features over time.

- **Add time delays**: A governance decision made at time T only takes effect at, e.g., T + 90 days. This allows users and applications that find the decision unacceptable to switch to another application (potentially a fork). Compound has a [timing mechanism](https://compound.finance/docs/governance) in its governance, but in principle, the delay can (and should eventually) be much longer.

- **Be more fork-friendly**: Facilitate quick coordination and execution of a fork for users. This reduces the benefits of governance capture.

The Uniswap case is particularly interesting: it is an intentional behavior that on-chain governance funds teams, who can develop future versions of the Uniswap protocol, but _it is up to the users to opt_ for upgrading to those versions. This is a hybrid of on-chain and off-chain governance that leaves only a limited role for the on-chain side.

However, limited governance is not an acceptable solution in itself; the areas where governance is most needed (e.g., distributing funds for public goods) are among the most vulnerable to attacks. Funding public goods is particularly vulnerable to attacks because there is a very direct way for an attacker to profit from bad decisions: they can try to pass a bad decision that sends them funds. Therefore, we also need techniques to improve governance itself...

#Governance 