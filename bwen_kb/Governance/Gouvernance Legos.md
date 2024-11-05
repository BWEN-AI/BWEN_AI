Gouvernance Legos

Les éléments constitutifs de la gouvernance DeFi

Chez Tally, nous passons beaucoup de temps à parler aux projets DeFi de leur approche de la gouvernance. Nous voyons des projets recombiner les mêmes primitives de gouvernance de différentes manières.

J'ai pensé qu'il serait utile d'écrire un peu sur les trois "légos de gouvernance" les plus largement utilisés et sur la façon dont nous les voyons utilisés aujourd'hui. Je couvrirai le vote par signal de style Snapshot, le vote en chaîne de style Governor Alpha et les signatures multiples de style Gnosis.

Nous avons vu des projets qui n'utilisent qu'un seul de ces blocs de construction. D'autres projets comme Uniswap et Compound utilisent les trois. Je vais creuser dans les compromis d'un, puis parler de la façon dont ils peuvent s'emboîter.

Voter par signal

Le vote par signal a lieu lorsque les détenteurs de jetons signent leurs votes avec la paire de clés publique-privée. De cette façon, toute personne disposant d'une liste de votes signés peut vérifier la signature, voir qu'elle a des jetons sur la blockchain et additionner le nombre de votes. Habituellement, le vote par signal est pondéré en jeton en donnant aux électeurs un vote par jeton.

Le vote par signal est hors chaîne, ce qui signifie qu'il est gratuit ! Les utilisateurs n'ont pas à payer d'essence. Cela signifie également que le vote ne peut pas, à lui seul, affecter la blockchain. Les projets ont besoin d'un autre lego pour prendre le résultat d'un vote et changer les choses en chaîne.

Il est important de noter que signer des messages hors chaîne ne garantit pas que tout le monde verra le message. Le vote par signal dépend d'un serveur comme Snapshot (tk) ou d'un réseau de diffusion peer-to-peer comme libp2p pour diffuser les votes.

Vote en chaîne

Le vote en chaîne est ce que cela ressemble. Les détenteurs de jetons peuvent créer et voter sur des propositions proportionnellement à leur solde de jetons. Les votes se déroulent en chaîne, de sorte qu'un contrat intelligent de gouvernance peut totaliser les votes et déterminer quelles propositions sont acceptées.

Certains projets ont mis en œuvre des versions personnalisées du vote en chaîne, mais depuis que Compound a lancé Governor Alpha l'année dernière, nous avons vu d'autres projets l'adopter comme norme de facto. J'en ai parlé davantage dans The Case for Governor Alpha.

Le vote en chaîne coûte de l'essence, mais payer pour mettre des votes en chaîne rend le système plus sûr et décentralisé. La gouvernance ne dépend pas d'un tiers pour diffuser les votes, car elle hérite simplement de la disponibilité des données de la blockchain elle-même.

Signature multiple

Les multisigs existent depuis avant même que les blockchains n'existent. Un multisig est lorsqu'un nombre fixe d'adresses partagent le pouvoir administratif. Un sous-ensemble d'entre eux - par ex. 3 sur 5, tous les 4 sur 4, ou même 1 sur 2 - ont le pouvoir d'exécuter des commandes d'administration.

L'équipe Gnosis a construit Safe, un contrat intelligent largement utilisé et testé au combat qui implémente des multisigs de taille arbitraire sur Ethereum. Ce contrat est largement utilisé pour administrer les contrats de protocole et les trésoreries en chaîne.

Tableau des compromis de gouvernance lego

🛠 Assemblage des pièces

Nous avons vu des projets commencer par choisir l'une de ces primitives pour administrer leurs contrats, mais les choses deviennent intéressantes lorsque les gouvernances les combinent.

Signal vote + Admin multisig = Instantané

Vous connaissez peut-être Snapshot, un outil populaire permettant aux projets de gérer leur gouvernance. Parce que l'outil est construit sur le vote par signal, tout détenteur de jeton peut voter gratuitement sur Snapshot.

Comme je l'ai mentionné ci-dessus, le vote par signal n'exécute rien sur la chaîne. Habituellement, les votes instantanés dépendent d'un multisig administrateur pour exécuter avec précision et rapidité la volonté des électeurs en chaîne. Récemment, Snapshot a annoncé qu'il allait utiliser l'oracle reality.eth pour exécuter les propositions en chaîne.

Vote par signal + Vote en chaîne = Cycle de vie de l'idée à la proposition

Radicle est l'un des nombreux projets qui utilisent un cycle de vie de proposition avec à la fois le vote par signal et le vote en chaîne. Dans le cas de Radicle, ils utilisent Snapshot comme contrôle de température non contraignant pour voir quelles idées semblent avoir du soutien. Une fois qu'une idée passe le contrôle de température, les membres de la communauté Radicle peuvent affiner l'idée en une proposition formelle sur leurs forums. Enfin, ils soumettent la proposition sous forme de code exécutable pour un vote sur Radicle Governor Alpha.

Ce cycle de vie utilise l'étape gratuite - le vote par signal - pour faciliter la génération de nouvelles idées et utilise l'étape coûteuse mais sécurisée - le vote en chaîne - pour assurer la sécurité du contrat et de la trésorerie.

Vote en chaîne + Admin multisig = Délégation de pouvoir à un comité

Uniswap et Compound ont récemment adopté des votes en chaîne pour déléguer le pouvoir de dépenser aux multisigs. Uniswap a proposé le programme de subventions Uniswap, et Compound a suivi peu de temps après avec le programme de subventions Compounds. Une fois adoptées, ces propositions ont chacune financé un multisig de personnes dont la mission est de financer le travail dans l'écosystème de ce projet.

Je m'attends à ce que nous voyions de plus en plus ce modèle, où les gouvernances en chaîne lentes mais sécurisées délèguent des fonds et une autorité exécutive limités aux multisigs parce qu'ils peuvent prendre des décisions plus rapidement.

🧱Plus de pièces à venir bientôt!

Nous commençons à entendre parler d'équipes construisant des legos de gouvernance encore plus intéressants ! J'espère écrire bientôt une autre version de cet article de blog avec des modèles de conception plus intéressants.

[https://blog.tally.xyz/governance-legos-6559f2234a3a](https://blog.tally.xyz/governance-legos-6559f2234a3a)

#Governance 