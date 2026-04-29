# Enhancing Reinforcement Learning With Cross-Domain Knowledge Transfer via Seeded Graph Matching

Author: Geoffrey Murray
University: Pennsylvania State University


## Abstract

Transfer reinforcement learning (TRL) aims to boost the e GLYPH&lt;14&gt; ciency of reinforcement learning (RL) agents by leveraging knowledge from related tasks. Prior research primarily focuses on intradomain transfer, overlooking the complexities of transferring knowledge across tasks with di GLYPH&lt;11&gt; ering state and action spaces. Recent e GLYPH&lt;11&gt; orts in cross-domain TRL aim to bridge this gap by establishing mappings between disparate source and target spaces, thereby enabling knowledge transfer across RL tasks with varied state and action configurations. However, existing studies often rely on strict prior assumptions about the relationships between state spaces, which limits their practical generality. In this article, we propose a novel approach to cross-domain TRL based on seeded graph matching, which enables alignment between source and target tasks regardless of di GLYPH&lt;11&gt; erences in their state-action spaces. In particular, we model RL tasks as directed graphs, identify seed node pairs based on common RL properties, and devise a graph matching algorithm to align the source and target tasks by leveraging their structural characteristics. Building on this alignment, we introduce a policy-based transfer algorithm that improves the performance of the target RL task as its RL process progresses. Finally, we conduct comprehensive empirical studies on both discrete and continuous tasks with diverse state-action spaces. The experimental results validate the e GLYPH&lt;11&gt; ectiveness of the proposed algorithm.

Index Terms -Cross-domain, graph matching, reinforcement learning (RL), transfer learning.

## I. INTRODUCTION

R EINFORCEMENT learning (RL) is a paradigm in machine learning where an agent learns to make decisions by interacting with its environment. This interaction allows the agent to receive feedback in the form of rewards or penalties, prompting it to adapt its behavior to maximize the cumulative reward over time [1], [2]. Recent progress in RL has achieved remarkable successes in complex decisionmaking areas, including arcade games [3], [4], robotic control [5], [6], and autonomous driving [7], [8]. However, due to unknown environmental dynamics, RL agents often necessitate extensive exploration for performance improvement, incurring computational expenses. In response, transfer RL (TRL) emerges as a solution by utilizing knowledge from similar tasks to bolster RL performance. TRL has shown significant success in enhancing RL performance across various practical applications [9], [10], [11], demonstrating its potential in reducing the learning time and computational resources required.

Recent surveys [12], [13] categorize TRL algorithms into two main categories: same-domain and cross-domain, di GLYPH&lt;11&gt; erentiated by the similarities or di GLYPH&lt;11&gt; erences in the state-action spaces of the source and target tasks. Same-domain TRL focuses on knowledge transfer either within the same domain or between domains with identical state-action spaces. In the existing literature, the majority of TRL research predominantly focuses on knowledge transfer across tasks within the same domain. In particular, certain strategies from the source task, such as the learned policy [14], [15] or collected demonstrations [16], can be directly applied to the target task. Alternatively, some methods introduce auxiliary rewards based on external knowledge to facilitate the learning process in the target task [17], [18]. Moreover, advanced neural networkbased techniques, including policy distillation [19], [20] and progressive networks [21], [22], are employed to boost the RL model performance. For a deeper exploration of same-domain TRL, interested readers can refer to [12] and [13]. In crossdomain TRL algorithms, the discrepancy between the state and action spaces of di GLYPH&lt;11&gt; erent tasks presents a significant hurdle for the smooth transfer of knowledge from a source task to a target task. To bridge this gap, a mapping between the source and target domains is crucial for enabling the transfer of knowledge across tasks from divergent domains. A notable challenge in existing research is the autonomous derivation of this intertask mapping. Ammar et al. [23] and Joshi and Chowdhary [24]

See https: // www.ieee.org / publications / rights / index.html for more information.

approached this by conceptualizing the mapping relationship between the state feature representations of the source and target tasks as a linear model. They then applied manifold alignment techniques to derive this mapping from positive trajectories, facilitating cross-domain transfer. Nonetheless, the assumption that the state feature representations between tasks must align linearly severely restricts the broad applicability of this model. On another front, You et al. [25] proposed a crossdomain TRL method that learns state-action correspondences between source and target tasks through embeddings optimized to capture shared task-relevant features. This method, however, requires known task relevance and similar task representations to facilitate the learning of mapping, limiting its versatility across varied task configurations.

From the discussions above, it is evident that existing automatic cross-domain TRL methods come with inherent limitations related to the disparities or relationships between the state spaces of source and target tasks. These limitations stem from the requirement to learn a comprehensive mapping of state spaces across RL tasks, which necessitates paired states that cover the entirety of both source and target state spaces for training. However, in the context of cross-domain TRL, only localized state relationships are available prior to e GLYPH&lt;11&gt; ectively learning the mapping. As a result, some form of constraint or prior knowledge is essential to deduce the global mapping from this limited dataset. Unfortunately, the ability to establish a relationship model based on these constraints is limited because the mapping relationship between the source and target tasks is often inconsistent with a single rule. Therefore, these methods can only apply to specific situations and cannot be applied to more complex scenarios. In this study, we thus aim to explore more generalized knowledge transfer across disparate RL domains without the constraints typically associated with the relationship of state-action spaces between source and target tasks.

Specifically, in this article, we propose a new cross-domain TRL algorithm designed to enable smooth knowledge transfer between tasks, irrespective of variations in their state-action spaces. The key components of the proposed algorithm are organized as follows.

- 1) Intertask Mapping via Seeded Graph Matching: We model source and target tasks as graphs using the structural properties of Markov decision processes (MDPs), construct rare matched node pairs based on inherent RL property, and infer the global node mapping between RL tasks as intertask mapping. A cost-e GLYPH&lt;14&gt; cient seeded graph matching algorithm is presented to ascertain node correspondences. To accommodate large-scale and heterogeneous graphs, we propose a graph clustering procedure to preprocess the source and target graphs before the seeded graph matching process.
- 2) Policy-Based Cross-Domain Knowledge Transfer: Due to the heterogeneity among RL tasks, it is crucial to address the potential mismatch between transferred knowledge and the target task. To tackle this challenge, Zhang et al. [26] presented an online policy-based TRL algorithm, which utilizes advantage values to guide the target task's learning process. This enables the transferred knowledge to be seamlessly utilized in target tasks with di GLYPH&lt;11&gt; erent state-action spaces while mitigating its negative e GLYPH&lt;11&gt; ects through self-learning from the
3. environment, thereby facilitating adaptation to crossdomain knowledge transfer. However, this approach imposes constraints on the relevance of state representations across RL tasks. In this study, we propose to eliminate these limitations by leveraging the node mapping between source and target graphs for a more generalized cross-domain knowledge transfer.
- 3) Dynamic Mapping Update During Learning: Existing cross-domain TRL algorithms often learn the intertask mapping based on random trajectories sampled from the target task before its learning. This kind of approach may cause the target agent to overlook important areas of the task space for learning intertask mapping, leading to inaccuracies or even negative transfer of knowledge. To address this issue, we propose to update the intertask mapping dynamically as the target agent explores the environment during the learning process, thereby providing high-quality transferred knowledge to facilitate progress. To evaluate the performance of the proposed algorithm, comprehensive empirical studies are conducted on both discrete and continuous tasks with di GLYPH&lt;11&gt; erent state-action spaces.

The rest of this article is organized as follows. Section II introduces the background of RL, the related works, and the motivation behind the study. Section III presents the details of our proposed algorithm, which includes the process of graph construction, the seeded graph matching algorithm, and the cross-domain knowledge transfer algorithm. Section IV provides comprehensive empirical studies, which are conducted on both discrete and continuous tasks with di GLYPH&lt;11&gt; erent state-action spaces to evaluate the proposed algorithm. Finally, Section V concludes this article with a few remarks.

## II. PRELIMINARY

In this section, we first provide a brief introduction to the background of RL. Next, we present the reviews of existing studies of cross-domain TRL. The motivation behind the proposed algorithm is also discussed at the end of this section.

## A. Background

In the RL process, an agent dynamically engages with its environment using a behavior policy GLYPH&lt;25&gt; , which maps a given state to a probability distribution across possible actions. The agent assesses the current state st and decides upon an action at . Following this decision, the environment evolves into a subsequent state st + 1 guided by the transition probabilities inherent to the current state st and the chosen action at . The agent then perceives this new state st + 1 and is awarded a reward rt , based on the outcome of its action. This cyclical process repeats itself, adhering to a set of predefined termination criteria. As can be observed, RL is a sequential decisionmaking process; it satisfies the Markov property and is thus typically formulated as an MDP, which is represented by the tuple M = &lt; S ; A ; P ; R &gt; . Here, S represents the set of states; A represents the set of actions; T : S × A ! S is the state transition function, mapping state-action pairs to a probability distribution over states; and P : S × A ! R is the reward function, determining the reward received by the agent from the environment. MDPs are often depicted as state transition graphs, where nodes correspond to states and (directed) edges

Fig. 1. Illustration of representing RL as an MDP graph.

<!-- image -->

denote transitions. Fig. 1 illustrates a simple MDP with three states (i.e., S 0, S 1, and S 2) and five actions (i.e., a 01, a 02, a 10, a 12, and a 20), along with four rewards (i.e., r 01, r 02, r 12, and r 20).

The goal of an RL agent is to learn a stochastic policy GLYPH&lt;25&gt; that maximizes the cumulative reward function. This function is generally represented as the expected discounted sum of rewards extended over an infinite horizon. Mathematically, this is expressed as

<!-- formula-not-decoded -->

where GLYPH&lt;13&gt; 2 (0 ; 1) is the discount factor used to calculate the long-term objective.

To maximize the objective defined in (1), as outlined in [27], existing approaches in the literature are primarily categorized into two main groups: value-based [3], [28], [29] and policybased algorithms [30], [31], [32]. On the one hand, valuebased algorithms aim to build a value function to provide a prediction of how good each state or each state / action pair is and then derive the policy according to this value function. Policy-based algorithms, on the other hand, seek to directly ascertain an e GLYPH&lt;11&gt; ective policy that aligns with the overarching goal of RL. Owing to their adaptability in addressing both discrete and continuous problem spaces within RL, policybased algorithms are widely employed across a diverse array of RL scenarios.

The mainstream of policy-based algorithms is policy gradient, where the policy GLYPH&lt;25&gt; is parameterized by a neural network GLYPH&lt;18&gt; and optimized using gradient ascent. The gradient of the policy is often given as follows:

<!-- formula-not-decoded -->

where A GLYPH&lt;18&gt; ( s ; a ) is the advantage function, which is defined as the di GLYPH&lt;11&gt; erence between the state-action value Q ( s ; a ) and the state value V ( s )

GLYPH&lt;18&gt;

<!-- formula-not-decoded -->

GLYPH&lt;18&gt;

GLYPH&lt;18&gt;

GLYPH&lt;18&gt;

where Q GLYPH&lt;25&gt; GLYPH&lt;18&gt; ( s ; a ) = E GLYPH&lt;2&gt;P 1 k = 0 GLYPH&lt;13&gt; k rt + k j st = s ; at = a ; GLYPH&lt;25&gt; GLYPH&lt;18&gt; GLYPH&lt;3&gt; denotes the expected reward following policy GLYPH&lt;25&gt; GLYPH&lt;18&gt; in a given state-action pair and V GLYPH&lt;25&gt; GLYPH&lt;18&gt; ( s ) = E GLYPH&lt;2&gt;P 1 k = 0 GLYPH&lt;13&gt; k rt + k j st = s ; GLYPH&lt;25&gt; GLYPH&lt;18&gt; GLYPH&lt;3&gt; estimates the expected reward following policy GLYPH&lt;25&gt; GLYPH&lt;18&gt; when in a given state. Therefore, the advantage function quantifies the advantage of taking action a in state s under policy GLYPH&lt;25&gt; GLYPH&lt;18&gt; .

Moreover, the actor-critic algorithm [33], a widely recognized policy-based algorithm, employs a learned model to estimate the advantage value. In particular, the actor learns the policy GLYPH&lt;25&gt; GLYPH&lt;18&gt; , parameterized by GLYPH&lt;18&gt; , and it is updated according to (2). The critic estimates the value function V GLYPH&lt;25&gt; GLYPH&lt;18&gt; , parameterized by GLYPH&lt;17&gt; , which is employed to compute the temporal di GLYPH&lt;11&gt; erence

(TD) error, serving as an unbiased estimate of the advantage function. The TD error is expressed as

<!-- formula-not-decoded -->

where the critic is updated by minimizing the squared loss of TD error

<!-- formula-not-decoded -->

GLYPH&lt;18&gt;

In this study, we consider the actor-critic method as the foundational RL approach within our proposed TRL algorithm.

## B. Related Work

1) Cross-Domain TRL: The task of transferring knowledge in cross-domain TRL presents considerable di GLYPH&lt;14&gt; culties, primarily due to variations in state-action spaces across domains. Some methods leverage strong task correlations and achieve cross-domain transfer through robust algorithmic designs without explicitly learning intertask mappings [25], [34]. However, when faced with larger domain discrepancies, constructing intertask mappings becomes an unavoidable challenge.

Initially, Taylor and Stone [35] manually defined transfer rules that map state and action spaces between the source and target tasks, enabling the adaptation and reuse of knowledge from the source task in the target domain. Subsequently, various algorithms have been proposed to automatically learn mappings of state (or state-action) spaces between source and target tasks. For instance, Gupta et al. [36] proposed learning invariant feature spaces between morphologically similar RL agents by training them on common auxiliary tasks, thereby enabling knowledge transfer through the learned invariant features. Ammar et al. [23] and Joshi and Chowdhary [24] utilized unsupervised manifold alignment to learn linear mappings between the state spaces of source and target tasks, facilitating transfer through these mappings. However, these representative cross-domain TRL approaches to learning intertask mappings either rely on manually defined structures or su GLYPH&lt;11&gt; er from limitations stemming from the complexity and diversity of state-action spaces. For example, the method in [35] required manually establishing mappings, while the method in [36] depended on human-designed auxiliary tasks to learn common representations between morphologically similar agents. The approaches by Ammar et al. [23] and Joshi and Chowdhary [24] assume a linear relationship between source and target state spaces, which is often unrealistic for capturing their correspondence in real-world scenarios. Moreover, You et al. [25] proposed a cross-domain TRL method that learns state-action correspondences via aligning trajectory data. This approach requires shared trajectory features between source and target tasks, along with strong task similarity, to ensure e GLYPH&lt;11&gt; ective mapping. Shoeleh and Asadpour [37] introduced a skillbased heterogeneous TRL framework that employs transfer component analysis (TCA) to identify a shared latent space across tasks, enabling high-level knowledge transfer. However, its e GLYPH&lt;11&gt; ectiveness depends on the assumption of similar environmental distributions for successful domain adaptation. Franzmeyer et al. [38] proposed a cross-domain imitation learning framework that jointly learns a policy and a domaininvariant state embedding through adversarial training and mutual information maximization to facilitate the transfer of expert demonstrations. This method focuses on maximizing

shared or partially similar state representations and environmental dynamics across tasks to enable optimal policy transfer. In addition, as robots with di GLYPH&lt;11&gt; erent morphologies often share similar dynamics, several cross-domain TRL algorithms have been specifically developed for robotic applications to address challenges such as motion retargeting and skill generalization [39], [40], [41]. In contrast, the method proposed in this article makes more relaxed assumptions about task similarity, thereby enabling its applicability to a broader range of task settings.

2) Graph-Based Reinforcement Learning: As discussed in Section II-A, RL is commonly framed within the context of MDPs, which are frequently visualized as state transition graphs. This conceptualization allows for the abstraction of RL tasks into MDP graphs, leveraging their structural attributes to tackle various RL challenges. For example, in RL scenarios involving low-level actions, navigating large action spaces poses challenges in exploring feasible trajectories for policy learning. To address this, the structure information of state transition graphs is employed to autonomously generate subgoals [42]. Sun et al. [43] enhance the interpretability of RL by modeling the RL problem as a graph structure. They infer causal relationships among nodes and edges in the source domain and leveraged these relationships to enable knowledge transfer from the source to the target domain. However, this method is only applicable when the source and target domains share identical state and action spaces. Graph neural networks (GNNs) [44], [45], as an e GLYPH&lt;11&gt; ective framework for capturing structural characteristics, have been employed to enhance the performance of RL [46]. However, these methods primarily focus on constructing graph models at the agent level or task level and thus are not applicable to learning interdomain task mappings.

## C. Motivation

Considering di GLYPH&lt;11&gt; erent domains of RL tasks and their unknown relationships, it is necessary for the paired state data to encompass the entire state space for learning the mapping between source and target tasks. However, the reality is that only a limited selection of paired states is typically accessible for this purpose. In such cases, the structural properties of MDP graphs o GLYPH&lt;11&gt; er a valuable resource, enabling the extrapolation of comprehensive state-space mappings from merely a fraction of paired state instances, thereby enhancing the e GLYPH&lt;14&gt; ciency and accuracy of cross-domain TRL strategies. This objective can be achieved through seeded graph matching, which enables the recovery of the full natural correspondence between the source and target graphs based on a given subset of matched node pairs [47].

Despite the existence of various seeded graph matching algorithms, their applicability within the realms of RL and cross-domain TRL remains limited. Existing algorithms typically exploit structural information [48] or semantic features [49]. In cross-domain TRL, due to the heterogeneity of different RL tasks and the randomness inherent in the graph construction process, the structures of graphs are nonidentical, making them unsuitable for calculating node relevance. As for the semantic features, it is also unfeasible to compute the similarity of nodes and edges between source and target tasks based on di GLYPH&lt;11&gt; erent state and action representations. Therefore, we propose a seeded graph matching algorithm that enables

Fig. 2. Overall structure of the proposed algorithm.

<!-- image -->

e GLYPH&lt;14&gt; cient matching between two nonisomorphic graphs using only a set of seed nodes. Since the proposed task alignment method is based on the graph structure of MDPs, which serve as the fundamental model for RL, it imposes no constraints on the similarity of state-action representations between RL tasks. The seed node set is defined based on the optimal and iteratively optimized trajectories in the source and target tasks, respectively. Consequently, the proposed method constitutes an unsupervised intertask alignment algorithm.

## III. PROPOSED ALGORITHM

In this section, we present the details of the proposed crossdomain TRL algorithm. We begin by outlining the proposed algorithm. Subsequently, we provide the details of constructing source and target graphs from their respective tasks. Then, we present a seeded graph matching algorithm for learning the node mapping between source and target graphs. Finally, the proposed cross-domain knowledge transfer algorithm is discussed.

## A. Outline of the Proposed Algorithm

Fig. 2 illustrates the overall structure of the proposed method, consisting of two phases: the former is before the learning process of the target task and the latter is during the target task learning process. We use circular symbols with di GLYPH&lt;11&gt; erent colors, i.e., green, purple, and blue, to represent the three important components of the proposed algorithm, including graph construction, graph update, seeded graph matching, and knowledge transfer. The products produced by these components in the learning process are also signed with the corresponding colors. Specifically, during the pretraining phase, both the source and target tasks gather trajectories from their respective environments to construct source and target graphs. As the target policy has not been learned yet, the target agent collects trajectories using a random policy. The source agent employs a combination of a random policy and a well-learned policy to ensure the collection of high-quality and overall trajectories. Subsequently, these trajectories are used to construct source and target graphs. In the learning phase, as the target agent explores its environment, the trajectories sampled in the environment are utilized to continually update the target graph. Then, the node mapping between source and target tasks is learned as the intertask mapping via seeded graph matching at the beginning of the target task training. Due to the dynamic update of the target graph, the node mapping between the source and target graphs is periodically updated at certain

## Algorithm 1 Outline of the Proposed Algorithm

Parameter: Episode counter e = 0; Maximal episode number E ; Number of trajectories Tn ;

Input: Source task o S ; Target task o T ; Source optimal policy GLYPH&lt;25&gt; S ; GLYPH&lt;3&gt; ; Node mapping update internal U ;

Output: Target optimal policy GLYPH&lt;25&gt; T ; GLYPH&lt;3&gt;

;

## 1 begin

- 2 Sample trajectories GLYPH&lt;28&gt; ( o S ) = f GLYPH&lt;28&gt; S 0 ; : : : ; GLYPH&lt;28&gt; S Tn S g in source task o S using mixture policy of random policy GLYPH&lt;25&gt; S ; rand and optimal source policy GLYPH&lt;25&gt; S ; GLYPH&lt;3&gt; .
- 3 Sample trajectories GLYPH&lt;28&gt; ( o T ) = f GLYPH&lt;28&gt; T 0 ; : : : ; GLYPH&lt;28&gt; T Tn T g in target task o T using random policy GLYPH&lt;25&gt; T ; rand .
- 4 Construct the source graph G S and the target graph G T according to Alg. 2.
- 5 while e 2 f 0 ; : : : ; E g do
- 6 Sample trajectories GLYPH&lt;28&gt; from the environment using target policy GLYPH&lt;25&gt; T .
- 7 Update the target graph G T according to Alg. 4.
- 8 if e = 0 or episode interval exceeds U then
- 9 Learn the node mapping according to Alg. 5.
- 10 end if
- 11 Learn the target agent model with the transferred knowledge according to Alg. 6.
- 12 e e + 1
- 13 end while
- 14 end

intervals to obtain the up-to-date node mapping covering the entire source and target graphs. With each update of the node mapping, the transferred knowledge, which is based on the node mapping, is also updated to improve its quality. Finally, the advantage value computed based on the source critic serves as transferred knowledge to guide the target task learning. In this way, the target agent learns from both the environment and the source task simultaneously during the learning process.

Algorithm 1 summarizes the pseudocode for the proposed cross-domain TRL algorithm. Here, o S and o T represent the source and the target tasks, respectively. Initially, source and target agents sample trajectories in their respective tasks (Lines 2 and 3 in Algorithm 1). Trajectories for the source task, denoted GLYPH&lt;28&gt; S = GLYPH&lt;28&gt; S 0 ; : : : ; GLYPH&lt;28&gt; S Tn S , are obtained using the well-learned source policy GLYPH&lt;25&gt; S with an GLYPH&lt;15&gt; -greedy strategy. To ensure diversity in the sampled data and to include optimal trajectories, the GLYPH&lt;15&gt; parameter transitions from 0 to 1 during the sampling process. Trajectories of the target task, denoted GLYPH&lt;28&gt; T = GLYPH&lt;28&gt; T 0 ; : : : ; GLYPH&lt;28&gt; T Tn T , are obtained by using a random policy GLYPH&lt;25&gt; T , where Tn represents the maximal number of trajectories. Subsequently, based on the obtained trajectories from the source and target tasks, the source graph G S and the target graph G T are constructed (Line 4 in Algorithm 1), as elaborated in Section III-B. Following this, the target agent undergoes training through the learning loop; during an episode, the target agent samples a trajectory GLYPH&lt;28&gt; from the environment (Line 6 in Algorithm 1). The target graph is then updated based on the newly sampled trajectory (Line 7 in Algorithm 1), with details provided in Section III-B. If the episode number equals 0, node mapping between the source and target graphs is learned using the proposed seeded graph matching algorithm

(Line 9 in Algorithm 1), introduced in Section III-C. Throughout subsequent learning, once the episode interval exceeds a certain threshold U , the node mapping between source and target graphs is relearned using the seeded graph matching algorithm (Line 9 in Algorithm 1). With the obtained node mapping, knowledge is transferred from the well-learned source task to the target task by providing guidance for policy update and exploration during the target agent's learning process (Line 11 in Algorithm 1), which is discussed in Section III-D. This learning loop persists until the target policy converges or certain predefined stopping criteria are met.

## B. Graph Construction

Graph construction involves creating an overall state distribution graph for RL tasks, constrained by a predefined number of nodes. This process is divided into two parts: initial graph construction and graph updating. During the initial graph construction process, MDP graphs of source and target tasks are generated by mapping nodes to states and edges to state transitions. However, translating an RL task into a graph representation often yields a graph of substantial size, which in turn escalates the computational demands associated with seeded graph matching. A further complication is encountered when the source and target graphs di GLYPH&lt;11&gt; er markedly in size, a disparity that can diminish the precision of node mapping learning. Additionally, altering the connections between nodes in a graph may lead to inaccuracies in modeling RL tasks and further a GLYPH&lt;11&gt; ect the accuracy of intertask mapping. To address these challenges, a cost-e GLYPH&lt;14&gt; cient graph clustering algorithm is proposed to control the graph size and preserve the connections between nodes, ensuring that the global distribution of states and transitions remains intact. These steps are completed during the pretraining phase. The second part, graph updating, occurs during the training phase. As the target agent discovers new areas within the environment, it incorporates new states and transitions into the target graph. Concurrently, graph clustering may also be executed on the target graph to maintain the graph's size within the predefined node number threshold.

During the initial graph construction process, we generate MDP graphs from trajectories sampled in source and target tasks, serving as the source and target graphs. Each node in the graph corresponds to a state in the trajectories, and edges represent transitions between these states, following prior research [50]. A directed graph G is constructed according to the pseudocode presented in Algorithm 2. Specifically, each node GLYPH&lt;23&gt; corresponds to a state s , and each edge " corresponds to a state transition. Each node GLYPH&lt;23&gt; is associated with the corresponding state s , denoted C ( GLYPH&lt;23&gt; ), and the number of states associated with a node GLYPH&lt;23&gt; is denoted as Cn ( GLYPH&lt;23&gt; ), which is often set to 1 initially (Lines 2 and 3 in Algorithm 2). The total number of nodes in graph G is denoted as Vn ( G ), and the maximal allowable number of nodes in graph G is denoted as Va ( G ). If Vn ( G ) exceeds Va ( G ), graph clustering is performed (Line 5 in Algorithm 2), which is summarized in Algorithm 3. Initially, the maximal allowable node number Va ( G ) is provided as input. The minimum allowable number of states associated with a node in graph G , denoted as Ci ( G ), is set to one greater than the minimum number of associated states among all nodes: Ci ( G ) min f Cn ( GLYPH&lt;23&gt; 0) ; : : : ; Cn ( GLYPH&lt;23&gt; Vn ) g + 1 (Line 3 in Algorithm 3). Next, each node GLYPH&lt;23&gt; i in the graph

## Algorithm 2 Algorithm of the Graph Construction

Parameter: Graph G ; Node set V ; Edge set E ; Node GLYPH&lt;23&gt; ; Edge " ; Node number Vn ( G ); Number of associated state Cn ( GLYPH&lt;23&gt; ); Input: Trajectories GLYPH&lt;28&gt; ( o ) = f GLYPH&lt;28&gt; 0 ; : : : ; GLYPH&lt;28&gt; Tn g ; Maximal number of node Va ( G );

## Output:

Graph G ;

- 1 begin
- 2 Construct the directed graph G = ( V ; E ) from trajectories GLYPH&lt;28&gt; ( o ): states correspond to the nodes GLYPH&lt;23&gt; , and transitions between states correspond to the edges " .
- 3 Cn ( GLYPH&lt;23&gt; i ) 1 ; i 2 [0 ; Vn ( G )].
- 4 if Vn ( G ) &gt; Va ( G ) then
- 5 Execute graph clustering according to Alg. 3.
- 6 end if
- 7 end

## Algorithm 3 Algorithm of the Graph Clustering

Parameter: Graph G ; Node number Vn ( G ); Associated states set C ( GLYPH&lt;23&gt; ); Number of associated state Cn ( GLYPH&lt;23&gt; ); Minimum

```
number of associated state Ci ; Input: Maximal number of node Va ( G ); Output: Graph G ; 1 begin 2 repeat 3 Ci min f Cn ( GLYPH<23> 0) ; : : : ; Cn ( GLYPH<23> Vn ( G ) ) g + 1. 4 for node GLYPH<23> in graph G do 5 if Cn ( GLYPH<23> ) < Ci then 6 Node GLYPH<23> merge a random neighbor node GLYPH<23> 0 . 7 C ( GLYPH<23> ) C ( GLYPH<23> 0 ) [ C ( GLYPH<23> ). 8 Cn ( GLYPH<23> ) Cn ( GLYPH<23> 0 ) + Cn ( GLYPH<23> ). 9 end if 10 end for 11 until Vn ( G ) > Va ( G ) 12 end
```

is visited. If its associated states number Cn ( GLYPH&lt;23&gt; i ) is less than Ci ( G ), node GLYPH&lt;23&gt; i merges with a random neighbor node GLYPH&lt;23&gt; 0 i . Node GLYPH&lt;23&gt; i inherits all connections and associated states of both nodes: C ( GLYPH&lt;23&gt; i ) C ( GLYPH&lt;23&gt; 0 i ) [ C ( GLYPH&lt;23&gt; i ). The number of associated states Cn ( GLYPH&lt;23&gt; i ) of node GLYPH&lt;23&gt; i is updated to the sum of the associated state number of the two merging nodes: Cn ( GLYPH&lt;23&gt; i ) Cn ( GLYPH&lt;23&gt; 0 i ) + Cn ( GLYPH&lt;23&gt; i ) (Lines 6-8 in Algorithm 3). Node GLYPH&lt;23&gt; 0 i is then removed from graph G . After visiting every node and performing node merging, if the number of nodes Vn ( G ) in graph G still exceeds Va ( G ), Ci ( G ) is updated to one greater than the minimum number of associated states among all nodes again (Line 3 in Algorithm 3), and the graph clustering process is repeated. When the number of nodes Vn ( G ) in graph G is less than Va ( G ), the graph clustering process terminates (Lines 11 and 12 in Algorithm 3). After the graph clustering process, we obtain a graph satisfying the predefined node number threshold.

In the graph updating process, new states and actions are incorporated into the graph. Similar to the initial graph construction process, each new node GLYPH&lt;23&gt; is initially associated with one state, with the number of associated states, denoted Cn ( GLYPH&lt;23&gt; ), set to 1 (Lines 2 and 3 in Algorithm 4). If the total

## Algorithm 4 Algorithm of the Graph Updating

Parameter: Graph G ; Node GLYPH&lt;23&gt; ; Edge " ; Node number Vn ( G ); Number of state associated with node Cn ( GLYPH&lt;23&gt; ); Input: Trajectory GLYPH&lt;28&gt; ; Maximal number of node Va ( G ); Output: Graph G ;

## 1 begin

- 2 Supplement new nodes and edges into the graph G from trajectory GLYPH&lt;28&gt; : new states correspond to the nodes GLYPH&lt;23&gt; , and new transitions of states correspond to the edges " .
- 3 Cn ( vnew ) 1.
- 4 while Vn ( G ) &gt; Va ( G ) do
- 5 Execute graph clustering according to Alg. 3.
- 6 end while
- 7 end

Fig. 3. Illustration of the graph clustering process.

<!-- image -->

number of nodes Vn ( G ) in the graph G exceeds the maximal allowable node count Va ( G ), graph clustering is performed (Line 5 in Algorithm 4). Algorithm 4 provides the pseudocode for the graph updating algorithm.

Moreover, we illustrate the graph clustering process in Fig. 3. This figure displays a graph comprising six nodes labeled A, B, C, D, X, and Y, each associated with an individual state. The objective is to condense the graph to contain only two nodes. Initially, a node with the fewest associated states is merged with its a random neighboring node. If multiple nodes meet the criteria, one is randomly chosen for merging. Here, node A and its neighboring node X are selected, with node A inheriting all connections and associated states of both nodes and node X removing from the graph. Subsequently, nodes C, B, D, and Y, now holding the fewest associated states, are considered for merging. Y is randomly chosen for merging with node B, preserving Y and removing B. This iterative process continues until the graph is condensed to the maximal allowed number of nodes. We can find that the proposed graph clustering algorithm prioritizes preserving the global distribution of states and transitions, even though it necessitates sacrificing some local transition details. Given the often imprecise nature of state correspondence across di GLYPH&lt;11&gt; erent RL task domains, the omission of local state transition information does not markedly detract from the accuracy of overall alignment across task domains.

## C. Seeded Graph Matching Algorithm

In the graph matching phase, we propose a seeded graph matching algorithm to establish a correspondence between nodes in two graphs, serving as the intertask mapping between the source and target tasks. As discussed in Section II-C, this algorithm focuses solely on utilizing seed nodes for

Algorithm 5 Algorithm of the Proposed Seeded Graph Matching

Parameter: Seed set GLYPH&lt;10&gt; Seed ; Candidate seed set GLYPH&lt;10&gt; Cand ; Assessment seed set GLYPH&lt;10&gt; Assm ; Seed node pair pSeed ; Candidate seed node pair pCand ; Maximal order ! ; Matching score M ;

```
Correlation value Cor ; Input: Source graph G S ; Target graph G T ; Output: Seed set GLYPH<10> Seed ; 1 begin 2 Construct seed node pair set GLYPH<10> Seed . 3 Construct candidate node pair set GLYPH<10> Cand . 4 repeat 5 for pCand = [ GLYPH<23> S Cand ; GLYPH<23> T Cand ] in GLYPH<10> Cand do 6 Obtain the assessment node set GLYPH<10> S Assm ;! ( GLYPH<23> S Cand ). 7 Obtain the assessment node set GLYPH<10> T Assm ;! ( GLYPH<23> T Cand ). 8 for GLYPH<23> S Assm in GLYPH<10> Assm ;! ( GLYPH<23> S Cand ) do 9 for GLYPH<23> T Assm in GLYPH<10> Assm ;! ( GLYPH<23> T Cand ) do 10 if pAssm = [ GLYPH<23> S Assm ; GLYPH<23> T Assm ] 2 GLYPH<10> Seed then 11 Compute correlation value Cor ( pCand , pAssm ) via Eq. 7. 12 end if 13 end for 14 end for 15 Compute matching score M ( pCand ) via Eq. 6. 16 end for 17 Add the candidate node pair with the highest matching score to the seed set GLYPH<10> Seed . 18 Update candidate node pair set GLYPH<10> Cand . 19 until GLYPH<10> Cand = GLYPH<30> 20 end
```

graph matching. Benefiting from the straightforward nature of seed-based calculations, this method has high computational e GLYPH&lt;14&gt; ciency, enabling dynamic updates of the intertask mapping during the target agent's learning process. To enhance node mapping quality, we incorporate higher order information of seeds when computing the matching score for each node pair while also considering the influence of di GLYPH&lt;11&gt; erent orders on the matching score.

The proposed seeded graph matching algorithm builds upon the state-of-the-art paradigm known as percolation graph matching (PMG) [51], [52]. PMG iteratively maintains a set of matched pairs starting from the seed set and comprises four key steps in each round: 1) each pair in the seed set percolates its neighboring pairs to constructing a candidate seed set; 2) calculating the matching score for every candidate seed; 3) evaluating the eligibility of each candidate seed against specific criteria and confirming matches if they qualify; and 4) newly matched pairs are then designated as seeds for subsequent iterations. The process repeats until no further candidates are identified.

For a comprehensive overview of the proposed seeded graph matching algorithm, Algorithm 5 summarizes its workflow. First, we leverage potential correspondences among states in the source and target tasks to establish an initial seed set (Line 2 in Algorithm 5). The starting and goal states of the source task correspond to the starting and goal states of the target task, respectively, and the optimal trajectory in the source task is chronologically aligned with that in the target task. Within the graph context, nodes associated with the start and goal states in the source and target graphs constitute two pairs of seeds, and node sequences associated with states in the optimal trajectories of both tasks are aligned to form additional seeds.

With the obtained seed set, we introduce the proposed seeded graph matching algorithm following the PMG procedure. First, we consider the first-order neighboring nodes of seeds as candidate nodes for their respective graphs. The candidate nodes from the source graph are permuted and combined with the candidate nodes from the target graph to form pairs, constituting candidate seeds p Cand (Line 3 in Algorithm 5). The set containing these candidate seeds is denoted as GLYPH&lt;10&gt; Cand.

Subsequently, we evaluate the matching score of each candidate seed (Lines 5 and 16 in Algorithm 5). To begin, we collect the neighboring nodes of candidate seeds within the ! breadthfirst search (BFS) distance in the source and target graphs, referred to as assessment nodes, and record the order number of these assessment nodes. Next, the assessment nodes from the source task are arranged and combined with the assessment nodes from the target task to form node pairs. Within these node pairs, if a node pair is also a seed, it is considered an assessment seed p Assm ;! = [ GLYPH&lt;23&gt; S Assm ;! ; GLYPH&lt;23&gt; T Assm ;! ]. The collection of assessment seeds of the candidate seed p Cand is denoted as GLYPH&lt;10&gt; Assm ;! ( p Cand) (Lines 6 and 7 in Algorithm 5). Then, we compute the matching score of candidate seeds from two factors: the number of assessment seeds and the correlation between the candidate seeds and corresponding assessment seeds (Line 15 in Algorithm 5). For a candidate seed p Cand = [ GLYPH&lt;23&gt; S Cand ; GLYPH&lt;23&gt; T Cand ], we define the matching score function M ( p Cand) as follows:

<!-- formula-not-decoded -->

where Cor( p Cand ; p Assm ;! ) represents the correlation value between the candidate seed and one of its assessment seeds, as the contribution of each assessment seed to the matching score. This equation indicates that a higher number of assessment seeds in the assessment seed set GLYPH&lt;10&gt; Assm ;! ( p Cand) leads to a higher matching score.

Then, we evaluate the correlation value between the candidate seed and its assessment seeds, which is based on two factors. First, the distance between the candidate seed and its assessment seed is considered. As the basis for node matching, the influence of the assessment seed diminishes with increasing distance, leading to a decrease in correlation. Second, we consider the di GLYPH&lt;11&gt; erence in order number between nodes in the assessment seed. Imagine merging the source and target graphs and treating seeds as overlapping points, a disparity in order between two nodes in the assessment seed p Assm ;! indicates a di GLYPH&lt;11&gt; erence in the relative position in the merged graph of the two nodes in the corresponding candidate seed p Cand. Therefore, the correlation decreases with an increase in the di GLYPH&lt;11&gt; erence in order between nodes GLYPH&lt;23&gt; S Assm ;! and GLYPH&lt;23&gt; T Assm ;! . The correlation value between the candidate seed p Cand and the assessment seed p Assm ! is computed as follows

(Line 11 in Algorithm 5):

<!-- formula-not-decoded -->

where GLYPH&lt;21&gt; is a decay factor, determining the rate at which the correlation value decreases with an increase in the order. Typically, it is set to a relatively small value, such as 0.001, to di GLYPH&lt;11&gt; erentiate the correlation value of assessment nodes with di GLYPH&lt;11&gt; erent distances. This formula consists of two terms. The first term assesses the correlation value by measuring the distance from the candidate seed p Cand to the assessment seed p Assm ;! . Here, we use the higher order of the nodes in the assessment seed p Assm to represent the distance. The second term evaluates the correlation value based on the di GLYPH&lt;11&gt; erence in order between the two nodes in the assessment seed p Assm ;! .

Once the matching score for each candidate seed is obtained, the candidate seed with the highest matching score is selected to be added into the seed set (Line 17 in Algorithm 5). Following this update to the seed set, the candidate seeds and their corresponding assessment seed sets are also updated accordingly and proceed to the next iteration (Line 18 in Algorithm 5). The iteration process persists until all candidate seeds have been matched, and no new candidate seeds emerge (Line 19 in Algorithm 5).

## D. Knowledge Transfer Module

Last but not least, the knowledge transfer module facilitates cross-domain knowledge transfer by utilizing the node mapping between the source and target graphs to enhance the target task's performance. The advantage value serves as the key piece of transferred knowledge, directing policy updates in the target task's learning process, which was proposed by Zhang et al. [26]. According to (4), the advantage value is derived from the V value calculated by the critic network. However, in the early learning stages, the target agent lacks su GLYPH&lt;14&gt; cient data to learn an appropriate policy, and the V value estimated by the target critic may be nearly arbitrary and fail to accurately reflect the actual cumulative discounted expected value. Consequently, the advantage value is thus not able to facilitate policy learning initially. To address this, the transferred advantage value, derived from the well-learned source task, provides a positive bias in updating the target policy, which helps in determining both the step size and direction of the optimization of the target actor network. By leveraging the transferred advantage value, the target agent is able to reduce random exploration and collect high-quality trajectories more e GLYPH&lt;14&gt; ciently.

To enhance clarity, the flow of the proposed knowledge transfer algorithm is depicted in Algorithm 6. First, we acquire the V value for each node in the source graph (Lines 3-5 in Algorithm 6), considering the value distribution of the source graph as an approximation of the value distribution of the source task. The node value in the source graph is computed by averaging the V values of all associated states. To account for varying reward ranges across tasks, the node values are

```
Parameters : Matching score M ; Input : Source critic GLYPH<17> S ; Source graph G S ; Target graph G T ; Trajectories GLYPH<28> ; Seed set GLYPH<10> Seed ; Episode e ; Update internal U ; Output : Transfer critic GLYPH<19> ; Target critic GLYPH<17> T ; Target actor GLYPH<18> T ; 1 begin 2 if e = 0 then 3 for node GLYPH<23> S in G S do 4 Compute the node value W ( GLYPH<23> S ) via Eq. 8. 5 end for 6 end if 7 if e = 0 or episode interval exceeds U then 8 for seed node pair ( GLYPH<23> S ; GLYPH<23> T ) in GLYPH<10> Seed do 9 Assign the node value W ( GLYPH<23> S ) to the node GLYPH<23> T as its node value W ( GLYPH<23> T ). 10 end for 11 Train the transfer critic via Eq. 9. 12 end if 13 for each sample ( s T i ; a T i ) in target task trajectories GLYPH<28> T do 14 Compute the transfer advantage value via Eq. 10. 15 Compute the target advantage value via Eq. 11. 16 Compute the total advantage value via Eq. 12. 17 end for 18 Train the target actor via Eq. 13. 19 Train the target critic via Eq. 14. 20 end
```

Algorithm 6 Algorithm of the Knowledge Transfer

normalized to remove task-specific characteristics. Hence, the node value in source graph, denoted W ( GLYPH&lt;23&gt; S ), is computed by

<!-- formula-not-decoded -->

where V S ( Cj ( GLYPH&lt;23&gt; S )) denotes the V value of the j th associated state Cj ( GLYPH&lt;23&gt; S ) of the node GLYPH&lt;23&gt; S calculated based on the source critic and Cn ( GLYPH&lt;23&gt; S ) denotes the number of states associated with the node GLYPH&lt;23&gt; S . Since the source graph's structure is fixed postinitialization, its node values are computed only once at the beginning of the target task's learning process.

By obtaining the node values in the source graph, the values of corresponding nodes in the target graph, denoted W ( GLYPH&lt;23&gt; T ), are assigned based on the node mapping (Lines 8-10 in Algorithm 6). In this way, the value distribution of the source task is mapped to the target task space. Subsequently, a transfer critic is constructed to learn the value distribution of the target graph (Line 11 in Algorithm 6). The loss function of the transfer critic, L ( GLYPH&lt;19&gt; ), is defined as follows:

<!-- formula-not-decoded -->

As the target graph and the node mapping undergo continuous updates during the learning process, the transfer critic also keeps updating to ensure alignment with the latest node mapping.

During the target task's learning process, given trajectories GLYPH&lt;28&gt; sampled from the target task, we compute the transfer advantage value, denoted A Transfer , for each state using the transfer critic according to (3) (Line 14 in Algorithm 6). We use the

transfer V value of the next state, denoted V Transfer ( st + 1), as an estimation of the Q value of the current state and action, denoted Q Transfer ( st ; at ). Consequently, the transfer advantage value A Transfer is computed by

<!-- formula-not-decoded -->

Meanwhile, given trajectories GLYPH&lt;28&gt; , the target agent computes the self-learning advantage value A T according to (4) (Line 15 in Algorithm 6), which is given by

<!-- formula-not-decoded -->

where r ( st ; at ) is the reward received by the target agent from the environment and V T denotes the V value computed by the target critic.

Then, the total advantage value A Total for the policy learning in the target task is then determined by both the target advantage value A T and the transfer advantage value A Transfer (Line 16 in Algorithm 6), which is given by

<!-- formula-not-decoded -->

where GLYPH&lt;11&gt; is used to balance the contributions of self-learning and knowledge transfer, which is often set to 0.5 to enable an equal influence on the target agent's learning from both aspects [26]. GLYPH&lt;12&gt; is used to scale the range of A Transfer , which is defined as the amplitude of variation in self-learning advantage values over the current episode so that the range of the transfer advantage is consistent with that of target advantage.

The actor network of the agent in the target task, which is parameterized by GLYPH&lt;18&gt; T , is updated according to the total advantage value (Line 18 in Algorithm 6). The gradient of the loss is calculated as follows:

<!-- formula-not-decoded -->

Finally, the optimization of the target critic network, which is parameterized by GLYPH&lt;17&gt; T , is proceeded as routine, which is described in Section II-A (Line 19 in Algorithm 6), which is given by

<!-- formula-not-decoded -->

## E. Computational Complexity Analysis

1) Time Computational Complexity: In the graph construction and update process, computational cost arises from node creation and clustering. Let Vn new and En new represent new nodes and edges, respectively. For each of the D data entries, the algorithm checks for existing nodes and edges. With constant-time edge lookup per node, the total time complexity is O ( D GLYPH&lt;1&gt; Vn new). Next, using adjacency lists to construct the graph incurs a time complexity of O ( Vn new + En new). Then, graph clustering reduces nodes iteratively until reaching a threshold Va , with each iteration requiring O (1). Thus, the time complexity for clustering is O ( Vn new GLYPH&lt;0&gt; Va ) GLYPH&lt;25&gt; O ( Vn new). In summary, the overall computational complexity for the graph construction phase is O ( D GLYPH&lt;3&gt; Vn new + En new).

In the seeded graph matching phase, computational e GLYPH&lt;11&gt; orts involve three sequential components. The process begins with

Fig. 4. Illustration of grid tasks with di GLYPH&lt;11&gt; erent goals. (a) Grid task with goal g 1. (b) Grid task with goal g 2. (c) Grid task with goal g 3.

<!-- image -->

generating candidate seed sets by accessing first-order neighbors of initial seeds through adjacency lists, resulting in a time complexity of O ( Vn S + Vn T ). Next, the assessment node sets are constructed for each candidate seed by aggregating nodes within a ! -BFS distance, represented as V Assm ;! , which introduces a complexity of O ( Vn S GLYPH&lt;3&gt; Vn S Assm ;! + Vn T GLYPH&lt;3&gt; Vn T Assm ;! ). Finally, matching scores are calculated by correlating the assessment node sets between the source and target graphs, a step characterized by the highest computational cost with complexity O ( Vn T GLYPH&lt;3&gt; ( Vn S Assm ;! GLYPH&lt;3&gt; Vn T Assm ;! )). The total time complexity is O ( Vn T GLYPH&lt;3&gt; ( Vn S Assm ;! GLYPH&lt;3&gt; Vn T Assm ;! + Vn T Assm ;! + 1) + Vn S GLYPH&lt;3&gt; ( Vn S Assm ;! + 1)). Retaining only the highest-order term, this simplifies to O ( Vn T GLYPH&lt;3&gt; Vn S Assm ;! GLYPH&lt;3&gt; Vn T Assm ;! ), which increases as the maximum order ! increases. 1

During the policy transfer phase, the agent updates its policy through an additional forward pass of the transfer critic, which requires constant time O (1).

2) Space Complexity: The adjacency list representation contributes O ( Vn + En ) space complexity. Furthermore, the source and target assessment node sets are also stored as adjacency lists, with a space complexity of O ( Vn Cand GLYPH&lt;3&gt; Vn Assm ;! ). Additionally, the content of nodes, including the discretized state vectors, increases with the addition of new nodes to the graph, leading to a space complexity of O ( Vn new). In summary, the space complexity of the proposed method is O ( Vn new + Vn + En + Vn Cand GLYPH&lt;3&gt; Vn Assm ;! ). 2

## IV. EMPIRICAL STUDY

To demonstrate the e GLYPH&lt;14&gt; cacy and performance of the proposed algorithm, comprehensive empirical studies conducted experiments on the two experiment platforms: Grid World and Pinball [53]. The details of the experimental setups, results, and discussions are presented in this section.

## A. Grid World

1) Experimental Setups: Fig. 4 illustrates a 24 × 21 grid world, where an agent navigates from a specified grid toward the goal grid. The agent chooses from four possible actions: up, down, left, and right, each moving it one step in the corresponding direction. The state representation comprises the one-hot encoding of the agent's position relative to the target grid and the detection of surrounding walls. Within this

1 When ! is close to 1, accessing first-order neighboring nodes becomes a constant-time operation, resulting in time complexity of O ( Vn T ). Conversely, as ! increases, the assessment node set may span the entire graph, causing the time complexity to approach O ( Vn T GLYPH&lt;3&gt; Vn S GLYPH&lt;3&gt; Vn T ).

2 When ! is large, Vn Assm ;! approaches Vn , whereas for ! = 1, Vn Assm ;! approximates a constant roughly equal to the average node degree. Vn Cand represents the first-order neighboring nodes of the seeds, excluding the already matched seeds, and its maximum size does not exceed Vn . Therefore, the upper bound is O ( Vn new + Vn GLYPH&lt;3&gt; Vn + En ) and the lower bound is O ( Vn new + Vn + En ).

TABLE I COMPARISON OF THE RELATED CROSS-DOMAIN TRL METHODS

grid world, Start denotes the starting point of the agents and g denotes the goal for the grid tasks. Both source and target agents start from the same grid but may have di GLYPH&lt;11&gt; erent goals. We designed three grid tasks with distinct goal positions, i.e., g 1g 3, each presenting varying levels of di GLYPH&lt;14&gt; culty. Goals g 2 and g 3 are situated farther from the start position than goal g 1, thereby presenting more challenging tasks. The simplest grid task with goal g 1 is selected to serve as both the source and target task, whereas the more complex grid tasks with goals g 2 and g 3 are designated solely as target tasks. The task ends when the agent reaches the goal or exceeds a predefined step limit of 1000 to guarantee su GLYPH&lt;14&gt; cient exploration of the environment. Upon reaching the goal grid, the agent receives a reward of + 5 while exceeding the step limit results in a penalty of GLYPH&lt;0&gt; 1.

To di GLYPH&lt;11&gt; erentiate between the state and action spaces of the source and target tasks, a representation transition matrix Z converts the state representation in the target task into a form with a di GLYPH&lt;11&gt; erent permutation order, which is defined as: ˜ st = Z GLYPH&lt;3&gt; st , where st 2 R n denotes the original state vector, ˜ st represents the transformed state vector, and Z is a representation transition matrix satisfying: Zi ; j 2 f 0 ; 1 g n × n ; P n i = 1 Zi ; j = 1 ; P n j = 1 Zi ; j = 1. This matrix is randomly generated and remains unknown for the target task, ensuring that the welllearned source policy cannot be applied directly in the target task. It is important to note that the proposed cross-domain TRL algorithm is not confined to such specific scenarios but can be extended to address broader cross-domain TRL settings, in which source and target tasks may feature variations in dimension and physical interpretation of representation, distinct ranges and distributions of reward functions, and diverse environmental dynamics, among other factors.

To further compare related cross-domain TRL algorithms, we summarize the key di GLYPH&lt;11&gt; erences between existing works and the proposed method in Table I. As shown in the table, most existing cross-domain TRL algorithms learn mappings based on representational features, whereas the proposed algorithm exploits the structural properties of MDP graphs. This design enables the proposed algorithm to accommodate substantial di GLYPH&lt;11&gt; erences in state-action spaces. Taylor and Stone [35] require manually specifying correspondences between source and target tasks. Gupta et al. [36] rely on auxiliary tasks to facilitate transfer. Ammar et al. [23], Joshi and Chowdhary [24], and Shoeleh and Asadpour [37] adopt linear RL models, making their mappings incompatible with policy gradient methods. Moreover, the complex matrix decompositions in [23] and [24] limit scalability to more complex tasks. You et al.

TABLE II SUMMARY OF THE PARAMETER SETTING IN THE EXPERIMENTS

[25] assume shared dynamics and representations to learn embeddings in a common task-relevant feature space. Since Zhang et al. [26] employ a transfer mechanism similar to the proposed method, including the same foundational RL methods and online transfer strategies, we select it as a baseline for comparison. By contrasting it with the proposed method, we emphasize the e GLYPH&lt;11&gt; ectiveness of the proposed seeded graph matching algorithm in aligning tasks with significant di GLYPH&lt;11&gt; erences across domains. Additionally, we include a standard RL algorithm without knowledge transfer as a baseline for comparison. Specifically, we use proximal policy optimization (PPO) [32], a widely used policy gradient-based RL algorithm, as the foundational RL algorithm in this study. Moreover, regarding graph clustering and graph size management of the proposed method, we conduct experiments using various graph sizes in the proposed method to measure the sensitivity to variations in graph size.

For the proposed algorithm, the state-space size of grid tasks is relatively small, obviating the necessity for graph clustering to reduce its size. In the initial graph construction stage, we use 30 trajectories from both the source and target tasks to construct the source and target graphs. The maximal order of neighboring nodes used for constructing the assessment seed set, denoted ! , is set to 5. To ensure a fair comparison, all algorithms employ PPO with identical configurations. The policy and value networks in PPO in all the compared algorithms are configured as neural networks with three layers and 64 hidden units. These networks are trained using standard backpropagation with the ADAM optimizer, which has a learning rate of 0.001. Additionally, all compared algorithms are executed for 20 independent runs to obtain averaged results for comparison. Comprehensive details regarding the overall hyperparameter configuration are provided in Table II.

2) Results and Discussion: For the target grid task with goal g 1, Fig. 5(a) and (b) depicts the averaged results in

Fig. 5. Comparison on the grid task with g 1. (a) Averaged step number. (b) Averaged success rate.

<!-- image -->

Fig. 6. Comparison on the grid task with g 2. (a) Averaged step number. (b) Averaged success rate.

<!-- image -->

terms of averaged step number and success rate. 3 The figures reveal that the proposed algorithm exhibits faster convergence compared to the baseline algorithm PPO, which is without knowledge transfer from the source task, with respect to both averaged step number and success rate. Moreover, the proposed algorithm achieves superior asymptotic performance. These results demonstrate the e GLYPH&lt;11&gt; ectiveness of the proposed TRL algorithm in facilitating knowledge transfer to enhance RL performance across tasks with di GLYPH&lt;11&gt; erent domains. The advantage-based method, which is the compared cross-domain TRL algorithm, obtains the worst results, with the agent unable to learn any feasible strategy. This is because the method requires the source and target tasks to share similar state spaces and value function distributions. When the di GLYPH&lt;11&gt; erences in state-action spaces and value function distributions are substantial, the knowledge transferred from the source task fails to help the target task's learning and instead interfered with it. The proposed method, possessing the same knowledge transfer mechanism as the advantage-based method, leverages the seeded graph matching algorithm to align the source and target task spaces, which allows the proposed method to overcome the limitations imposed by the assumption of similar state spaces and value function distributions. This demonstrates the e GLYPH&lt;11&gt; ectiveness of the proposed seeded graph matching algorithm in cross-domain TRL scenarios with di GLYPH&lt;11&gt; erent state-action spaces.

For the target grid task with goal g 2, Fig. 6(a) and (b) presents the averaged results regarding the averaged step number and success rate. Since goal g 2 is situated farther from the starting grid than goal g 1, the agent encounters increased di GLYPH&lt;14&gt; culty in exploring and receiving positive feedback. Consequently, learning the optimal policy using PPO in the grid task with goal g 2 is slower compared to the grid task with goal g 1. As task di GLYPH&lt;14&gt; culty increases, the proposed

3 In this study, the success rate is calculated as the ratio of the number of successful goal completions to the total number of task performances in the target task within a window of 30 episodes:success rate = the number of successful goal completions the total number of task performances .

Fig. 7. Comparison on the grid task with g 3. (a) Averaged step number. (b) Averaged success rate.

<!-- image -->

method uses the transfer advantage value to guide the target agent in navigating toward the goal grid, resulting in a more pronounced improvement compared to the grid tasks with goal g 1. However, the advantage-based method achieves the worst performance, which implies that directly using the source critic network to compute transfer advantage without aligning task domains through the seeded graph matching algorithm deteriorates the target task's performance due to substantial domain di GLYPH&lt;11&gt; erences. Furthermore, the advantage-based method performs worse in the task with goal g 2 compared to goal g 1 since the negative transfer e GLYPH&lt;11&gt; ects become more severe with increased task di GLYPH&lt;14&gt; culty. These findings underscore the necessity and e GLYPH&lt;11&gt; ectiveness of employing the seeded graph matching algorithm to align task domains in cross-domain TRL.

For the target grid task with goal g 3, Fig. 7(a) and (b) presents the averaged results in terms of the averaged step number and success rate, respectively. Reaching goal g 3 entails traversing multiple narrow channels, posing significant challenges for the agent. Consequently, compared with grid tasks with goals g 1 and g 2, learning this task requires the longest training time. As shown in the figures, the agent using PPO achieves a success rate of approximately 95% by the 4000th episode, while the proposed method reaches nearly 100% success by only 3000th episode. These results further validate the e GLYPH&lt;11&gt; ectiveness of the proposed algorithm. The advantage-based method obtains the poorest performance, with the negative transfer e GLYPH&lt;11&gt; ect becoming more pronounced as the task di GLYPH&lt;14&gt; culty increases. To evaluate the statistical significance of the proposed method over the baseline, we performed analysis of variance (ANOVA) on the number of steps over 20 independent runs. Due to the clear negative transfer observed with the advantage-based method, only the proposed method and the baseline were compared. The results 4 indicate highly significant improvements over the baseline, with large e GLYPH&lt;11&gt; ect sizes, confirming the robustness and practical impact of the proposed algorithm.

Moreover, to demonstrate that the proposed method can establish mappings between RL tasks in di GLYPH&lt;11&gt; erent state-action spaces and transform the value distribution of the source task into the value distribution depended on the target state space, we visualize the value distribution of the source critic and the transfer critics of three target grid tasks in Fig. 8. In particular, Fig. 8(a) shows the value distributions of source critic, and

4 For the three grid tasks, Levene's test yielded p 1 = 0 : 0089, p 2 = 0 : 5406, and p 3 = 0 : 0000. Accordingly, Welch's ANOVA was applied to Groups 1 and 3, and standard ANOVA was applied to Group 2. The ANOVA results were: 1) F 1(1 ; 33 : 4667) = 40 : 2343, p 1 = 3 : 3 e GLYPH&lt;0&gt; 7 &lt; 0 : 0001, and GLYPH&lt;17&gt; 2 1 = 0 : 5143; 2) F 2(1 ; 38) = 32 : 7884, p 2 = 1 e GLYPH&lt;0&gt; 6 &lt; 0 : 0001, and GLYPH&lt;17&gt; 2 2 = 0 : 4632; and 3) F 3(1 ; 19 : 7938) = 81 : 4677, p 3 = 1 : 88 e GLYPH&lt;0&gt; 8 &lt; 0 : 0001, and GLYPH&lt;17&gt; 3 3 = 0 : 6819.

Fig. 8. Illustration of the value distribution of source and transfer critics in grid tasks. (a) Source critic in grid task with goal g 1. (b) Transfer critic in grid task with goal g 1. (c) Transfer critic in grid task with goal g 2. (d) Transfer critic in grid task with goal g 3.

<!-- image -->

Fig. 8(b)-(d) shows the value distributions of the transfer critics in the target grid tasks with goals g 1g 3, respectively. In these figures, the position of nodes corresponds to the grid position in the tasks. The color of nodes represents the V value computed by the transfer critic networks. Bright colors indicate high values, while dark colors indicate low values. Specifically, in Fig. 8(a) and (b), where the grid task with goal g 1 and source grid task possess the same start and goal positions, their value distributions are similar as well. The nodes in brightest color correspond to the vicinity of the goal grid g 1, while the darkest node are close to the initial state. As the distance from the goal grid increases, the color of nodes changes from bright to dark continuously. In Fig. 8(c) and (d), where the positions of the goal grids di GLYPH&lt;11&gt; er, the corresponding bright areas also change, corresponding to the vicinity of the goal grids g 2 and g 3, respectively. As the value distribution of the source critic is learned from the environment by the source agent, while the value distributions of the target critics are mapped from the value distribution of the source critic via the node mapping, these figures indicate that the proposed seeded graph matching algorithm can accurately learn the node mapping between source and target graphs, and the transfer critic can e GLYPH&lt;11&gt; ectively learn the value distribution from the source critic according to the node mapping between source and target tasks.

## B. Pinball

1) Experimental Setups: Fig. 9 illustrates the minefield navigation task (MNT), an experiment involving a variety of polygons and a ball. The objective is for the ball to navigate through the polygons to reach a specified target location within a specified time limit. In this task, the environment dynamics account for collisions between the ball and obstacles, along with motion resistance. The state vector encompasses the positions of the ball and target, as well as the velocity of the ball in both vertical and horizontal directions, all represented as continuous variables. Actions, defined as velocity increments, include moving up, down, left, right, or remaining

Fig. 9. Illustration of pinball tasks with di GLYPH&lt;11&gt; erent environments. (a) Pinball task with env1. (b) Pinball task with env2. (c) Pinball task with env3.

<!-- image -->

Fig. 10. Comparison on the pinball task with env1. (a) Averaged step number. (b) Averaged success rate.

<!-- image -->

stationary and are considered discrete. The positions of the target, polygons, and initial ball setup are predetermined. Three target tasks are designed, each with a di GLYPH&lt;11&gt; erent number of polygons and di GLYPH&lt;14&gt; culty levels, as shown in Fig. 9(a)-(c). The task with fewer polygons represents a simpler scenario, while the task with more polygons poses a greater challenge. The simple pinball task with environment env1 is chosen as the source task among these three tasks. If the ball runs out of time, it receives a negative feedback GLYPH&lt;0&gt; 1 and stops acting. If the ball reaches the goal position, it receives a positive feedback + 10. To ensure adequate exploration of the environment, the maximum step number is fixed at 5000.

Similar to the grid task, we shu GLYPH&lt;15&gt; e the state and action vectors in the source and target tasks di GLYPH&lt;11&gt; erently to construct distinct state and action spaces for each task.

For the proposed algorithm, as the state space in the pinball task is continuous, the state space undergoes a discretization process with a precision of 0.1, which implies that state value changes within 0.1 are disregarded. To manage computational costs, we control graph size through graph clustering. The initial graph size, which serves as a basis for defining the maximal graph size, can be estimated through random sampling in the environment. After su GLYPH&lt;14&gt; cient random sampling, the graph size grows slowly and stabilizes at approximately 6000 nodes. To evaluate the method's tolerance to variations in graph size, we conducted experiments with maximum graph sizes ranging from 500 to 4000. Other hyperparameters remain consistent with those in the grid task, as detailed in Table II.

2) Results and Discussion: Fig. 10(a) and (b) presents the averaged results in terms of averaged step number and success rate, which are obtained by the proposed algorithm and the compared approach on the target pinball task with env1 over 20 independent runs. The results show that the proposed method and PPO exhibit comparable learning performance due to the simplicity of the task, which limits the potential performance improvement through knowledge transfer. The proposed method demonstrates robust crossdomain knowledge transfer without negative e GLYPH&lt;11&gt; ects on performance, whereas the advantage-based method negatively

Fig. 11. Comparison on the pinball task with env2. (a) Averaged step number. (b) Averaged success rate.

<!-- image -->

impacts the performance. These findings further underscore the e GLYPH&lt;11&gt; ectiveness and necessity of the seeded graph matching algorithm in aligning tasks with significant domain disparities.

Regarding changes in graph size, Fig. 10(a) and (b) illustrates that changing the maximal graph size from 500 to 4000 has a minimal impact on the e GLYPH&lt;11&gt; ectiveness of knowledge transfer in the pinball task with env1. This is because the proposed method primarily utilizes the coarse-grained, global information of the value function mapped from the source task to guide learning in the target task. As the degree of graph clustering increases, the value function distribution mapped to the target graph loses local fine-grained information, causing the distribution to become increasingly smoother over the target state space. However, this does not alter the overall trend of the value function distribution. The knowledge transfer module can still provide the target agent with high-level guidance for policy convergence and exploration, enhancing the e GLYPH&lt;11&gt; ectiveness of the target agent's exploration. As a result, even with moderate clustering reduction, the process does not lead to a significant loss of critical information, and its impact on the e GLYPH&lt;11&gt; ectiveness of knowledge transfer remains minimal. This outcome demonstrates the adaptability of the proposed method to variations in the settings of the maximal graph size.

Fig. 11(a) and (b) presents the averaged results in terms of the averaged step number and success rate in the target pinball task with env2. As the number of polygons increases, navigating to the goal becomes inherently more challenging for the agent; it has to traverse narrow channels to reach the goal. As depicted in the figures, the proposed method reaches nearly 100% success rate, demonstrating a significant improvement in the target agent's asymptotic performance. This improvement is attributed to the transfer of advantage values from the source task, which helps the agent escape suboptimal local optima, thereby increasing its chances of reaching the goal. Changes in graph size have a negligible impact on the e GLYPH&lt;11&gt; ectiveness of knowledge transfer, which is similar to the results observed in the pinball task with env1. The advantage-based method negatively impacts the learning process in the target pinball task. Moreover, as tasks become increasingly complex, relying solely on advantage value-based knowledge transfer exacerbates the negative transfer e GLYPH&lt;11&gt; ects.

Fig. 12(a) and (b) presents the averaged results in terms of the averaged step number and success rate in the target pinball task with env3. Given the multitude of polygons and their intricate shapes, navigating through multiple narrow channels and avoiding potential collisions poses a considerable challenge for the agent. As shown in the figures, the agent using PPO fails to learn any viable strategies. Conversely, the agent using the proposed method achieves nearly a 100% success rate, highlighting the essential role of transfer learning in learning complex RL tasks. The performance of

Fig. 12. Comparison on the pinball task with env3. (a) Averaged step number. (b) Averaged success rate.

<!-- image -->

Lowvalue

SourcePinballTaskwithEnv1

<!-- image -->

<!-- image -->

Highvalue

Fig. 13. Illustration of the value distribution of source and transfer critics in pinball tasks. (a) Source critic in pinball task with env1. (b) Transfer critic in pinball task with env1. (c) Transfer critic in pinball task with env2. (d) Transfer critic in pinball task with env3.

<!-- image -->

<!-- image -->

the proposed transfer method remains nearly consistent across graph sizes ranging from 500 to 4000. This further validates the robustness and adaptability of the proposed algorithm to graph size variations, even in the complex pinball task. Finally, the advantage-based method without seeded graph matching initially outperforms PPO. In such challenging tasks, the lack of positive feedback from self-exploration often traps the agent in local optima, hindering its ability to explore e GLYPH&lt;11&gt; ectively. The bias introduced by the advantage values enables the agent to escape these local optima temporarily. However, due to negative transfer e GLYPH&lt;11&gt; ects, the agent ultimately fails to acquire a feasible strategy. This underscores the necessity of aligning task domains using seeded graph matching to mitigate negative transfer and enhance performance in crossdomain scenarios. To evaluate the statistical significance of the proposed method over the baseline, we performed ANOVA on the number of steps over 20 independent runs. Due to the clear negative transfer observed with the advantage-based method, only the proposed method and the baseline were compared. The results 5 indicate highly significant improvements over the baseline, with large e GLYPH&lt;11&gt; ect sizes, confirming the robustness and practical impact of the proposed algorithm.

5 For the three pinball tasks, Levene's test yielded p 1 = 0 : 0870, p 2 = 0 : 8603, and p 3 = 0 : 0121. Accordingly, Welch's ANOVA was applied to Group 3, and standard ANOVA was applied to Groups 1 and 2. The ANOVA results were: 1) F 1(1 ; 38) = 6 : 1710, p 1 = 0 : 0175 &lt; 0 : 05, and GLYPH&lt;17&gt; 2 1 = 0 : 1397; 2) F 2(1 ; 38) = 666 : 0448, p 2 = 1 e GLYPH&lt;0&gt; 25 &lt; 0 : 0001, and GLYPH&lt;17&gt; 2 2 = 0 : 9460; and 3) F 3(1 ; 22 : 9472) = 14891 : 5587, p 3 = 8 : 8 e GLYPH&lt;0&gt; 34 &lt; 0 : 0001, and GLYPH&lt;17&gt; 2 3 = 0 : 9975.

Finally, we visualize the value distribution of source critic and transfer critics for pinball tasks in Fig. 13. Fig. 13(a) shows the value distributions of the source critic in pinball task with env1, and Fig. 13(b)-(d) shows the value distributions of the transfer critics in the target pinball tasks with env1-env3, respectively. In these visualizations, the position of nodes corresponds to the agent's position in the environment, with node color indicating the V value computed by the critic networks. Bright colors denote high values, whereas dark colors signify low values. Notably, the highest value is observed near the goal state, while the lowest value is near the initial state. These figures provide further evidence that the proposed seeded graph matching algorithm accurately learns the node mapping relationship between source and target graphs, enabling the transfer critic to e GLYPH&lt;11&gt; ectively learn the value distribution from the source critic based on node mapping.

## V. CONCLUSION

This article proposes a cross-domain TRL algorithm that enhances the performance of a target task by leveraging knowledge from a source task with di GLYPH&lt;11&gt; erent state and action spaces. The proposed method models both tasks as graphs and learns node mappings between them to address cross-domain discrepancies. It introduces graph construction and matching algorithms, along with a policy-based transfer scheme to enable e GLYPH&lt;11&gt; ective online knowledge transfer. While the method demonstrates strong performance on tasks with exploration bottlenecks, its e GLYPH&lt;11&gt; ectiveness may decline in tasks where direct learning from the environment is su GLYPH&lt;14&gt; cient. Additionally, the algorithm relies on an initial set of automatically selected seeds (e.g., expert and target trajectories) to guide the graph matching process. However, defining optimal paths can be challenging in certain scenarios, such as when start points are randomly initialized, necessitating task-specific strategies for seed selection. To evaluate the performance of the proposed cross-domain TRL algorithm, comprehensive experiments were conducted on two RL tasks: Grid World and Pinball. The results confirm the e GLYPH&lt;11&gt; ectiveness of the proposed algorithm in improving the performance of the target task across the RL domains.
